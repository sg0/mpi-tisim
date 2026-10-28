// GPU implementations of the training and inference simulation kernels,
// organised as concurrent JOB SLOTS (one CUDA stream + cuBLAS handle each).
//
//   Training  unit (1 epoch): cublasDgemm (forward), cublasDgemm x2 (dW, dX),
//                             cublasDaxpy (update), loss reduction
//   Inference unit (chunk)  : per layer: bias broadcast,
//                             cublasDgemvStridedBatched (one GEMV per sample,
//                             shared weights: strideA = 0), ReLU; checksum
//
// cuBLAS runs in DEVICE pointer mode (alpha/beta/lr live in device memory), so
// no call blocks the host and many slots can be kept busy from one thread.
//
// cuBLAS is column-major and the host data is row-major. A row-major RxC
// matrix is the same memory as a column-major CxR matrix (its transpose), so
// every call is written in terms of the transposed problem; the mapping is
// spelled out next to each call.
//
// Reductions use fixed-shape strided partial sums (no atomics, no shared
// memory), so results are deterministic and the file also builds against the
// host emulation shim (cuda_emul.h, -DGPU_EMULATE) for testing without a GPU.
#include "gpu_kernels.h"

#ifdef GPU_EMULATE
#include "cuda_emul.h"
#define KLAUNCH(kern, grid, block, stream, ...) \
  emu::launch_async((stream), dim3(grid), dim3(block), kern, __VA_ARGS__)
#else
#include <cublas_v2.h>
#include <cuda_runtime.h>
#define KLAUNCH(kern, grid, block, stream, ...) \
  kern<<<(grid), (block), 0, (stream)>>>(__VA_ARGS__)
#if CUDART_VERSION >= 11060
#define HAVE_GEMV_BATCHED 1
#endif
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string g_err = "GPU not initialised";
int g_dev = -1;
int g_nslots = 0;
bool g_no_batched = false;  // set if the batched GEMV is unavailable / rejected

void set_err(const char* what, const char* call, const char* msg, int line) {
  char buf[512];
  std::snprintf(buf, sizeof buf, "%s error at gpu_kernels.cu:%d in %s: %s", what, line, call, msg);
  g_err = buf;
}

#define CK(call)                                                       \
  do {                                                                 \
    cudaError_t e_ = (call);                                           \
    if (e_ != cudaSuccess) {                                           \
      set_err("CUDA", #call, cudaGetErrorString(e_), __LINE__);        \
      return gpu::ERR;                                                 \
    }                                                                  \
  } while (0)
#define CB(call)                                                       \
  do {                                                                 \
    cublasStatus_t s_ = (call);                                        \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                 \
      char m_[32];                                                     \
      std::snprintf(m_, sizeof m_, "cublasStatus %d", (int)s_);        \
      set_err("cuBLAS", #call, m_, __LINE__);                          \
      return gpu::ERR;                                                 \
    }                                                                  \
  } while (0)

constexpr int RT = 256;                 // threads per block
constexpr int RB = 256;                 // blocks for level-1 partial sums
constexpr size_t P1 = (size_t)RT * RB;  // 65536 level-1 partials
constexpr size_t PART_N = P1 + RT;      // + 256 level-2 partials
constexpr int CHUNK_MAX = 256;          // inference samples per unit

// device scalar block layout (per slot)
enum { S_ONE = 0, S_ZERO = 1, S_MLR = 2, S_XN = 3, S_LOSS = 4, S_ACC = 5, S_N = 8 };
// pinned host block layout (per slot)
enum { H_RING = 0, H_ACC = gpu::RING, H_N = gpu::RING + 2 };

struct Buf { double* p = nullptr; size_t n = 0; };

struct Slot {
  bool ready = false;
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;
  cudaEvent_t ev[gpu::RING] = {};
  double* pin = nullptr;    // H_N pinned doubles
  double* stage = nullptr;  // pinned staging for W uploads
  size_t stage_n = 0;
  // training
  Buf X, T, Y, dY, dW, dX, W;
  int B = 0, K = 0, N = 0;
  // inference
  Buf Wts, Bias, In, Q0, Q1;
  int batch = 0, L = 0;
  std::vector<int> d;
  std::vector<size_t> woff, boff;
  // shared
  Buf Part, Scal;

  Buf* all[14] = {&X, &T, &Y, &dY, &dW, &dX, &W, &Wts, &Bias, &In, &Q0, &Q1, &Part, &Scal};
};
Slot g_slot[gpu::MAX_SLOTS];

// Stream-ordered allocation (cudaMallocAsync / cudaFreeAsync, CUDA >= 11.2):
// unlike cudaFree, releasing a buffer does not synchronise the whole device,
// so growing one slot's buffers never stalls the other streams.
void free_bufs(Slot& s) {
  for (Buf* b : s.all) {
    if (b->p) cudaFreeAsync(b->p, s.stream);
    b->p = nullptr;
    b->n = 0;
  }
}

bool grow(Slot& s, Buf& b, size_t n) {
  if (n == 0) n = 1;
  if (b.n >= n) return true;
  if (b.p) cudaFreeAsync(b.p, s.stream);
  b.p = nullptr;
  b.n = 0;
  n += n / 8;  // 12.5% slack: fewer re-allocations across tasks of similar size
  if (cudaMallocAsync((void**)&b.p, n * sizeof(double), s.stream) != cudaSuccess) {
    cudaGetLastError();  // clear the allocation error
    b.p = nullptr;
    return false;
  }
  b.n = n;
  return true;
}

__global__ void k_consts(double* s) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    s[S_ONE] = 1.0;
    s[S_ZERO] = 0.0;
  }
}

// Make sure every requested buffer of the slot is large enough. On failure
// the slot's pool is released and allocation retried once; if that fails too
// the caller gets OOM (and can free other idle slots or use the CPU).
int reserve(Slot& s, std::initializer_list<std::pair<Buf*, size_t>> req) {
  for (int attempt = 0; attempt < 2; ++attempt) {
    bool ok = true;
    for (auto& r : req) ok = ok && grow(s, *r.first, r.second);
    if (ok) {
      KLAUNCH(k_consts, 1, 1, s.stream, s.Scal.p);  // Scal may have been (re)allocated
      return gpu::OK;
    }
    free_bufs(s);
  }
  size_t tot = 0;
  for (auto& r : req) tot += r.second * sizeof(double);
  char m[128];
  std::snprintf(m, sizeof m, "could not allocate %.1f MB of device memory", tot / 1048576.0);
  g_err = m;
  return gpu::OOM;
}

int ensure_slot(int i) {
  if (i < 0 || i >= g_nslots) {
    g_err = "bad slot index";
    return gpu::ERR;
  }
  Slot& s = g_slot[i];
  if (s.ready) return gpu::OK;
  CK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
  CB(cublasCreate(&s.blas));
  CB(cublasSetStream(s.blas, s.stream));
  CB(cublasSetPointerMode(s.blas, CUBLAS_POINTER_MODE_DEVICE));
  for (auto& e : s.ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  CK(cudaMallocHost((void**)&s.pin, H_N * sizeof(double)));
  s.ready = true;
  return gpu::OK;
}

int blocks_for(size_t n) {
  return (int)std::max<size_t>(1, std::min<size_t>((n + RT - 1) / RT, 65535));
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------
// Same counter-based splitmix64 stream as the CPU fill_uniform(): element i of
// a stream with seed s is mix(s + (i+1)*golden), so values are bitwise
// identical to the host-generated ones.
__device__ __forceinline__ uint64_t d_mix(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

__global__ void k_fill(double* a, size_t n, uint64_t seed, double scale) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    uint64_t z = d_mix(seed + (uint64_t)(i + 1) * 0x9e3779b97f4a7c15ULL);
    double u = __dmul_rn((double)(z >> 11), 0x1.0p-53);
    a[i] = __dmul_rn(scale, __dadd_rn(__dmul_rn(u, 2.0), -1.0));
  }
}

// mlr = -(B / max(xn, 1e-12))  — same learning-rate rule as the CPU path
__global__ void k_lr(double* s, double B) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const double xn = s[S_XN];
    s[S_MLR] = -(B / (xn < 1e-12 ? 1e-12 : xn));  // == std::max(xn, 1e-12) on the CPU
  }
}

// dY = (Y - T) / B  and per-thread partial sums of (Y - T)^2 for the loss.
__global__ void k_dy_loss(const double* Y, const double* T, double* dY, size_t n, double B,
                          double* part) {
  const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  double s = 0.0;
  for (size_t i = gid; i < n; i += stride) {
    double d = Y[i] - T[i];
    s += d * d;
    dY[i] = d / B;
  }
  part[gid] = s;
}

__global__ void k_strided_sum(const double* in, size_t n, double* out) {
  const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  double s = 0.0;
  for (size_t i = gid; i < n; i += stride) s += in[i];
  out[gid] = s;
}

// Single thread: acc = (accumulate ? acc : 0) + scale * sum(in[0..n))
__global__ void k_final_sum(const double* in, int n, double* acc, int accumulate, double scale) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += in[i];
    s *= scale;
    acc[0] = accumulate ? acc[0] + s : s;
  }
}

// Y[j*m + r] = b[r] for j < c   (bias for a chunk of samples)
__global__ void k_bias_bcast(const double* b, double* Y, int m, size_t n) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    Y[i] = b[i % (size_t)m];
}

__global__ void k_relu(double* y, size_t n) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    y[i] = y[i] > 0.0 ? y[i] : 0.0;
}

// Deterministic sum of n values into *acc (scaled, optionally accumulated).
int reduce_into(Slot& s, const double* src, size_t n, double* acc, bool accumulate,
                double scale) {
  double* l1 = s.Part.p;
  double* l2 = s.Part.p + P1;
  if (n > P1) {
    KLAUNCH(k_strided_sum, RB, RT, s.stream, src, n, l1);
    src = l1;
    n = P1;
  }
  if (n > (size_t)RT) {
    KLAUNCH(k_strided_sum, 1, RT, s.stream, src, n, l2);
    src = l2;
    n = RT;
  }
  KLAUNCH(k_final_sum, 1, 1, s.stream, src, (int)n, acc, accumulate ? 1 : 0, scale);
  CK(cudaGetLastError());
  return gpu::OK;
}

int fill(Slot& s, double* a, size_t n, uint64_t seed, double scale) {
  KLAUNCH(k_fill, blocks_for(n), RT, s.stream, a, n, seed, scale);
  CK(cudaGetLastError());
  return gpu::OK;
}

}  // namespace

namespace gpu {

bool compiled() { return true; }
const char* backend() {
#ifdef GPU_EMULATE
  return "host emulation (cuda_emul.h)";
#else
  return "CUDA + cuBLAS";
#endif
}
const char* last_error() { return g_err.c_str(); }
int max_chunk() { return CHUNK_MAX; }

int init(int local_rank, int slots, DeviceInfo* info) {
  g_nslots = std::max(1, std::min(slots, MAX_SLOTS));
  // Hardware work queues the driver multiplexes streams onto (default 8).
  // Must be set before the CUDA context is created.
  if (!std::getenv("CUDA_DEVICE_MAX_CONNECTIONS")) {
    char v[16];
    std::snprintf(v, sizeof v, "%d", std::min(32, std::max(8, g_nslots)));
    setenv("CUDA_DEVICE_MAX_CONNECTIONS", v, 0);
  }
  int n = 0;
  CK(cudaGetDeviceCount(&n));
  if (n <= 0) {
    g_err = "no CUDA device visible to this process";
    return ERR;
  }
  g_dev = local_rank % n;
  CK(cudaSetDevice(g_dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, g_dev));
  int sms = 0, mclk_khz = 0, bus_bits = 0;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, g_dev));
  if (cudaDeviceGetAttribute(&mclk_khz, cudaDevAttrMemoryClockRate, g_dev) != cudaSuccess)
    mclk_khz = 0;
  if (cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, g_dev) != cudaSuccess)
    bus_bits = 0;
  cudaGetLastError();
  if (info) {
    std::strncpy(info->name, prop.name, sizeof info->name - 1);
    info->name[sizeof info->name - 1] = '\0';
    info->mem_gb = prop.totalGlobalMem / 1073741824.0;
    info->sm_count = sms;
    // DDR: 2 transfers per clock
    info->mem_bw_gbs = 2.0 * mclk_khz * 1e3 * (bus_bits / 8.0) / 1e9;
    info->device = g_dev;
  }
  return ensure_slot(0);
}

void shutdown() {
  if (g_dev < 0) return;
  for (int i = 0; i < g_nslots; ++i) {
    Slot& s = g_slot[i];
    if (!s.ready) continue;
    cudaStreamSynchronize(s.stream);
    free_bufs(s);
    for (auto& e : s.ev) cudaEventDestroy(e);
    if (s.pin) cudaFreeHost(s.pin);
    if (s.stage) cudaFreeHost(s.stage);
    cublasDestroy(s.blas);
    cudaStreamDestroy(s.stream);
    s = Slot();
  }
  g_dev = -1;
}

int mem_info(size_t* free_b, size_t* total_b) {
  CK(cudaMemGetInfo(free_b, total_b));
  return OK;
}

void release_slot(int i) {
  if (i < 0 || i >= g_nslots || !g_slot[i].ready) return;
  cudaStreamSynchronize(g_slot[i].stream);
  free_bufs(g_slot[i]);
}

// ------------------------------- training ----------------------------------
int train_begin(int slot, int B, int K, int N, uint64_t seedX, uint64_t seedT, uint64_t seedW,
                double w_scale, const double* W_host) {
  if (ensure_slot(slot) != OK) return ERR;
  Slot& s = g_slot[slot];
  const size_t BK = (size_t)B * K, BN = (size_t)B * N, KN = (size_t)K * N;
  int r = reserve(s, {{&s.X, BK}, {&s.T, BN}, {&s.Y, BN}, {&s.dY, BN}, {&s.dW, KN},
                      {&s.dX, BK}, {&s.W, KN}, {&s.Part, PART_N}, {&s.Scal, S_N}});
  if (r != OK) return r;
  s.B = B;
  s.K = K;
  s.N = N;
  if (fill(s, s.X.p, BK, seedX, 1.0) || fill(s, s.T.p, BN, seedT, 1.0)) return ERR;
  if (W_host) {
    // stage through pinned memory so the upload is truly asynchronous
    if (s.stage_n < KN) {
      if (s.stage) CK(cudaFreeHost(s.stage));
      s.stage = nullptr;
      s.stage_n = 0;
      CK(cudaMallocHost((void**)&s.stage, KN * sizeof(double)));
      s.stage_n = KN;
    }
    std::memcpy(s.stage, W_host, KN * sizeof(double));
    CK(cudaMemcpyAsync(s.W.p, s.stage, KN * sizeof(double), cudaMemcpyHostToDevice, s.stream));
  } else if (fill(s, s.W.p, KN, seedW, w_scale)) {
    return ERR;
  }
  // lr = B / ||X||^2 entirely on the device (device pointer mode)
  CB(cublasDdot(s.blas, (int)BK, s.X.p, 1, s.X.p, 1, s.Scal.p + S_XN));
  KLAUNCH(k_lr, 1, 1, s.stream, s.Scal.p, (double)B);
  CK(cudaGetLastError());
  return OK;
}

int train_enqueue_epoch(int slot, int ring) {
  Slot& s = g_slot[slot];
  const int B = s.B, K = s.K, N = s.N;
  const double* one = s.Scal.p + S_ONE;
  const double* zero = s.Scal.p + S_ZERO;
  // Row-major A (RxC) == column-major A^T (CxR, ld = C).
  //
  // forward  Y = X W            -> col-major: Y^T (NxB) = W^T (NxK) * X^T (KxB)
  //          W buf = W^T (NxK, ld N), X buf = X^T (KxB, ld K), Y buf = Y^T (NxB, ld N)
  CB(cublasDgemm(s.blas, CUBLAS_OP_N, CUBLAS_OP_N, N, B, K, one, s.W.p, N, s.X.p, K, zero, s.Y.p,
                 N));
  // dY = (Y - T) / B, loss = 0.5/B * ||Y - T||^2
  KLAUNCH(k_dy_loss, RB, RT, s.stream, s.Y.p, s.T.p, s.dY.p, (size_t)B * N, (double)B, s.Part.p);
  CK(cudaGetLastError());
  if (reduce_into(s, s.Part.p, P1, s.Scal.p + S_LOSS, false, 0.5 / B)) return ERR;
  // backward 1  dW = X^T dY     -> col-major: dW^T (NxK) = dY^T (NxB) * X (BxK)
  //          dY buf = dY^T (NxB, ld N) op N ; X buf = X^T (KxB, ld K) op T -> X (BxK)
  CB(cublasDgemm(s.blas, CUBLAS_OP_N, CUBLAS_OP_T, N, K, B, one, s.dY.p, N, s.X.p, K, zero,
                 s.dW.p, N));
  // backward 2  dX = dY W^T     -> col-major: dX^T (KxB) = W (KxN) * dY^T (NxB)
  //          W buf = W^T (NxK, ld N) op T -> W (KxN) ; dY buf op N ; dX buf = dX^T (KxB, ld K)
  CB(cublasDgemm(s.blas, CUBLAS_OP_T, CUBLAS_OP_N, K, B, N, one, s.W.p, N, s.dY.p, N, zero,
                 s.dX.p, K));
  // update   W = W - lr dW  (same element order in both layouts; alpha = -lr on device)
  CB(cublasDaxpy(s.blas, K * N, s.Scal.p + S_MLR, s.dW.p, 1, s.W.p, 1));
  CK(cudaMemcpyAsync(s.pin + H_RING + ring, s.Scal.p + S_LOSS, sizeof(double),
                     cudaMemcpyDeviceToHost, s.stream));
  CK(cudaEventRecord(s.ev[ring], s.stream));
  return OK;
}

int train_download_W(int slot, double* W_host) {
  Slot& s = g_slot[slot];
  const size_t KN = (size_t)s.K * s.N;
  // on the slot's own stream (the legacy default stream would be a device-wide
  // synchronisation point); a pageable destination makes it host-synchronous
  CK(cudaMemcpyAsync(W_host, s.W.p, KN * sizeof(double), cudaMemcpyDeviceToHost, s.stream));
  CK(cudaStreamSynchronize(s.stream));
  return OK;
}

// ------------------------------- inference ---------------------------------
int infer_begin(int slot, int batch, int L, const int* d, const uint64_t* seedW,
                const double* scaleW, const uint64_t* seedB, double scaleB, uint64_t seedIn,
                double acc0) {
  if (ensure_slot(slot) != OK) return ERR;
  Slot& s = g_slot[slot];
  s.batch = batch;
  s.L = L;
  s.d.assign(d, d + L + 1);
  s.woff.assign(L, 0);
  s.boff.assign(L, 0);
  int maxd = *std::max_element(d, d + L + 1);
  size_t nw = 0, nb = 0;
  for (int l = 0; l < L; ++l) {
    s.woff[l] = nw;
    s.boff[l] = nb;
    nw += (size_t)d[l + 1] * d[l];
    nb += d[l + 1];
  }
  const size_t q = (size_t)CHUNK_MAX * maxd;
  int r = reserve(s, {{&s.Wts, nw}, {&s.Bias, nb}, {&s.In, (size_t)batch * d[0]}, {&s.Q0, q},
                      {&s.Q1, q}, {&s.Part, PART_N}, {&s.Scal, S_N}});
  if (r != OK) return r;
  for (int l = 0; l < L; ++l) {
    if (fill(s, s.Wts.p + s.woff[l], (size_t)d[l + 1] * d[l], seedW[l], scaleW[l])) return ERR;
    if (fill(s, s.Bias.p + s.boff[l], d[l + 1], seedB[l], scaleB)) return ERR;
  }
  if (fill(s, s.In.p, (size_t)batch * d[0], seedIn, 1.0)) return ERR;
  // the previous task on this slot has fully completed, so the pinned cell is free
  s.pin[H_ACC] = acc0;
  CK(cudaMemcpyAsync(s.Scal.p + S_ACC, s.pin + H_ACC, sizeof(double), cudaMemcpyHostToDevice,
                     s.stream));
  return OK;
}

int infer_enqueue_chunk(int slot, int first, int count, int ring) {
  Slot& s = g_slot[slot];
  const double* one = s.Scal.p + S_ONE;
  const int d0 = s.d[0];
  const double* x = s.In.p + (size_t)first * d0;
  long long incx_batch = d0;  // distance between consecutive samples' input vectors
  double* Q[2] = {s.Q0.p, s.Q1.p};
  for (int l = 0; l < s.L; ++l) {
    const int m = s.d[l + 1], n = s.d[l];
    double* y = Q[l & 1];
    const double* Wl = s.Wts.p + s.woff[l];
    const size_t cm = (size_t)count * m;
    // y_j = b  for every sample j of the chunk
    KLAUNCH(k_bias_bcast, blocks_for(cm), RT, s.stream, s.Bias.p + s.boff[l], y, m, cm);
    CK(cudaGetLastError());
    // y_j += W x_j  ; W row-major (m x n) == col-major W^T (n x m, ld n) -> op T
    bool done = false;
#if defined(HAVE_GEMV_BATCHED) || defined(GPU_EMULATE)
    if (!g_no_batched) {
      cublasStatus_t st = cublasDgemvStridedBatched(s.blas, CUBLAS_OP_T, n, m, one, Wl, n,
                                                    0LL /* shared weights */, x, 1, incx_batch,
                                                    one, y, 1, (long long)m, count);
      if (st == CUBLAS_STATUS_SUCCESS) done = true;
      else g_no_batched = true;  // fall back to one GEMV per sample from now on
    }
#endif
    if (!done)
      for (int j = 0; j < count; ++j)
        CB(cublasDgemv(s.blas, CUBLAS_OP_T, n, m, one, Wl, n, x + (size_t)j * incx_batch, 1, one,
                       y + (size_t)j * m, 1));
    // a = ReLU(y) in place
    KLAUNCH(k_relu, blocks_for(cm), RT, s.stream, y, cm);
    CK(cudaGetLastError());
    x = y;
    incx_batch = m;
  }
  // checksum += sum of the chunk's outputs
  if (reduce_into(s, Q[(s.L - 1) & 1], (size_t)count * s.d[s.L], s.Scal.p + S_ACC, true, 1.0))
    return ERR;
  CK(cudaEventRecord(s.ev[ring], s.stream));
  return OK;
}

int infer_get_acc(int slot, double* acc) {
  Slot& s = g_slot[slot];
  CK(cudaMemcpyAsync(s.pin + H_ACC, s.Scal.p + S_ACC, sizeof(double), cudaMemcpyDeviceToHost,
                     s.stream));
  CK(cudaStreamSynchronize(s.stream));
  *acc = s.pin[H_ACC];
  return OK;
}

// ------------------------------- completion --------------------------------
int query(int slot, int ring) {
  cudaError_t e = cudaEventQuery(g_slot[slot].ev[ring]);
  if (e == cudaSuccess) return 1;
  if (e == cudaErrorNotReady) {
    cudaGetLastError();
    return 0;
  }
  set_err("CUDA", "cudaEventQuery", cudaGetErrorString(e), __LINE__);
  return -1;
}

double ring_value(int slot, int ring) { return g_slot[slot].pin[H_RING + ring]; }

int sync(int slot) {
  CK(cudaStreamSynchronize(g_slot[slot].stream));
  return OK;
}

}  // namespace gpu
