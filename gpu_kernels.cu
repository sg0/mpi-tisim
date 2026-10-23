// GPU implementations of the training and inference simulation kernels.
//
//   Training  : cublasDgemm (forward), cublasDgemm x2 (dW, dX), cublasDaxpy (update)
//   Inference : cublasDgemv + bias + ReLU per layer, one input vector at a time
//
// cuBLAS is column-major and the host data is row-major. A row-major RxC
// matrix is the same memory as a column-major CxR matrix (its transpose), so
// every call below is written in terms of the transposed problem. The mapping
// for each call is spelled out next to it.
//
// Reductions (loss, inference checksum) use fixed-shape strided partial sums,
// so results are deterministic run to run (no atomics). They need no shared
// memory, which also lets this file build against the host emulation shim
// (cuda_emul.h, -DGPU_EMULATE) for testing on machines without a GPU.
#include "gpu_kernels.h"

#ifdef GPU_EMULATE
#include "cuda_emul.h"
#define KLAUNCH(kern, grid, block, stream, ...) \
  emu::launch(dim3(grid), dim3(block), [&] { kern(__VA_ARGS__); })
#else
#include <cublas_v2.h>
#include <cuda_runtime.h>
#define KLAUNCH(kern, grid, block, stream, ...) \
  kern<<<(grid), (block), 0, (stream)>>>(__VA_ARGS__)
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string g_err = "GPU not initialised";
int g_dev = -1;
cudaStream_t g_stream = nullptr;
cublasHandle_t g_blas = nullptr;

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
      return -1;                                                       \
    }                                                                  \
  } while (0)
#define CB(call)                                                       \
  do {                                                                 \
    cublasStatus_t s_ = (call);                                        \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                 \
      char m_[32];                                                     \
      std::snprintf(m_, sizeof m_, "cublasStatus %d", (int)s_);        \
      set_err("cuBLAS", #call, m_, __LINE__);                          \
      return -1;                                                       \
    }                                                                  \
  } while (0)

// ---------------------------------------------------------------------------
// Device buffer pool (grow-only, reused across tasks)
// ---------------------------------------------------------------------------
struct Buf { double* p = nullptr; size_t n = 0; };
Buf bX, bT, bY, bdY, bdW, bdX, bW;      // training
Buf bWts, bBias, bIn, bA, bYv;          // inference
Buf bPart, bScal;                       // reductions / scalars
Buf* const kAll[] = {&bX, &bT, &bY, &bdY, &bdW, &bdX, &bW, &bWts, &bBias, &bIn, &bA, &bYv,
                     &bPart, &bScal};

constexpr int RT = 256;                 // threads per block
constexpr int RB = 256;                 // blocks for level-1 partial sums
constexpr size_t P1 = (size_t)RT * RB;  // 65536 level-1 partials
constexpr size_t PART_N = P1 + RT;      // + 256 level-2 partials

void free_pool() {
  for (Buf* b : kAll) {
    if (b->p) cudaFree(b->p);
    b->p = nullptr;
    b->n = 0;
  }
}

bool grow(Buf& b, size_t n) {
  if (n == 0) n = 1;
  if (b.n >= n) return true;
  if (b.p) cudaFree(b.p);
  b.p = nullptr;
  b.n = 0;
  if (cudaMalloc((void**)&b.p, n * sizeof(double)) != cudaSuccess) {
    cudaGetLastError();  // clear the sticky allocation error
    b.p = nullptr;
    return false;
  }
  b.n = n;
  return true;
}

// Make sure every requested buffer is large enough. On failure the whole pool
// is released (fragmentation / other ranks' usage) and allocation is retried once.
int reserve(std::initializer_list<std::pair<Buf*, size_t>> req) {
  for (int attempt = 0; attempt < 2; ++attempt) {
    bool ok = true;
    for (auto& r : req) ok = ok && grow(*r.first, r.second);
    if (ok) return 0;
    free_pool();
  }
  size_t tot = 0;
  for (auto& r : req) tot += r.second * sizeof(double);
  char m[128];
  std::snprintf(m, sizeof m, "could not allocate %.1f MB of device memory", tot / 1048576.0);
  g_err = m;
  return -1;
}

int blocks_for(size_t n) { return (int)std::max<size_t>(1, std::min<size_t>((n + RT - 1) / RT, 65535)); }

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------
// Same counter-based splitmix64 stream as the CPU fill_uniform(): element i
// of a stream with seed s is mix(s + (i+1)*golden), so values are bitwise
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

__global__ void k_relu(const double* y, double* a, int m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) a[i] = y[i] > 0.0 ? y[i] : 0.0;
}

// Deterministic sum of n values into *acc (scaled, optionally accumulated).
int reduce_into(const double* src, size_t n, double* acc, bool accumulate, double scale) {
  double* l1 = bPart.p;
  double* l2 = bPart.p + P1;
  if (n > P1) {
    KLAUNCH(k_strided_sum, RB, RT, g_stream, src, n, l1);
    src = l1;
    n = P1;
  }
  if (n > (size_t)RT) {
    KLAUNCH(k_strided_sum, 1, RT, g_stream, src, n, l2);
    src = l2;
    n = RT;
  }
  KLAUNCH(k_final_sum, 1, 1, g_stream, src, (int)n, acc, accumulate ? 1 : 0, scale);
  CK(cudaGetLastError());
  return 0;
}

int fill(double* a, size_t n, uint64_t seed, double scale) {
  KLAUNCH(k_fill, blocks_for(n), RT, g_stream, a, n, seed, scale);
  CK(cudaGetLastError());
  return 0;
}

// ---------------------------------------------------------------------------
// Task state
// ---------------------------------------------------------------------------
struct TrainCtx { int B = 0, K = 0, N = 0; } tc;
struct InferCtx {
  int batch = 0, L = 0, maxd = 0;
  std::vector<int> d;
  std::vector<size_t> woff, boff;
} ic;

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

int init(int local_rank, char* name, int name_len, double* mem_gb) {
  int n = 0;
  CK(cudaGetDeviceCount(&n));
  if (n <= 0) {
    g_err = "no CUDA device visible to this process";
    return -1;
  }
  g_dev = local_rank % n;
  CK(cudaSetDevice(g_dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, g_dev));
  if (name && name_len > 0) {
    std::strncpy(name, prop.name, name_len - 1);
    name[name_len - 1] = '\0';
  }
  if (mem_gb) *mem_gb = prop.totalGlobalMem / 1073741824.0;
  CK(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  CB(cublasCreate(&g_blas));
  CB(cublasSetStream(g_blas, g_stream));
  return g_dev;
}

void shutdown() {
  if (g_dev < 0) return;
  cudaStreamSynchronize(g_stream);
  free_pool();
  if (g_blas) cublasDestroy(g_blas);
  if (g_stream) cudaStreamDestroy(g_stream);
  g_blas = nullptr;
  g_stream = nullptr;
  g_dev = -1;
}

// ------------------------------- training ----------------------------------
int train_begin(int B, int K, int N, uint64_t seedX, uint64_t seedT, uint64_t seedW,
                double w_scale, const double* W_host, double* lr_out) {
  const size_t BK = (size_t)B * K, BN = (size_t)B * N, KN = (size_t)K * N;
  if (reserve({{&bX, BK}, {&bT, BN}, {&bY, BN}, {&bdY, BN}, {&bdW, KN}, {&bdX, BK}, {&bW, KN},
               {&bPart, PART_N}, {&bScal, 4}}) != 0)
    return -1;
  tc = {B, K, N};
  if (fill(bX.p, BK, seedX, 1.0) || fill(bT.p, BN, seedT, 1.0)) return -1;
  if (W_host)
    CK(cudaMemcpyAsync(bW.p, W_host, KN * sizeof(double), cudaMemcpyHostToDevice, g_stream));
  else if (fill(bW.p, KN, seedW, w_scale))
    return -1;
  // lr = 1 / trace(X^T X / B), same rule as the CPU path
  double xn = 0.0;
  CB(cublasDdot(g_blas, (int)BK, bX.p, 1, bX.p, 1, &xn));  // host pointer mode: synchronous
  *lr_out = (double)B / std::max(xn, 1e-12);
  CK(cudaStreamSynchronize(g_stream));
  return 0;
}

int train_epoch(double lr, double* loss_out) {
  const int B = tc.B, K = tc.K, N = tc.N;
  const double one = 1.0, zero = 0.0, mlr = -lr;
  // Row-major A (RxC) == column-major A^T (CxR, ld = C).
  //
  // forward  Y = X W            -> col-major: Y^T (NxB) = W^T (NxK) * X^T (KxB)
  //          W buf = W^T (NxK, ld N), X buf = X^T (KxB, ld K), Y buf = Y^T (NxB, ld N)
  CB(cublasDgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, N, B, K, &one, bW.p, N, bX.p, K, &zero, bY.p,
                 N));
  // dY = (Y - T) / B, loss = 0.5/B * ||Y - T||^2
  KLAUNCH(k_dy_loss, RB, RT, g_stream, bY.p, bT.p, bdY.p, (size_t)B * N, (double)B, bPart.p);
  CK(cudaGetLastError());
  if (reduce_into(bPart.p, P1, bScal.p, false, 0.5 / B)) return -1;
  // backward 1  dW = X^T dY     -> col-major: dW^T (NxK) = dY^T (NxB) * X (BxK)
  //          dY buf = dY^T (NxB, ld N) op N ; X buf = X^T (KxB, ld K) op T -> X (BxK)
  CB(cublasDgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_T, N, K, B, &one, bdY.p, N, bX.p, K, &zero,
                 bdW.p, N));
  // backward 2  dX = dY W^T     -> col-major: dX^T (KxB) = W (KxN) * dY^T (NxB)
  //          W buf = W^T (NxK, ld N) op T -> W (KxN) ; dY buf op N ; dX buf = dX^T (KxB, ld K)
  CB(cublasDgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, K, B, N, &one, bW.p, N, bdY.p, N, &zero,
                 bdX.p, K));
  // update   W = W - lr dW  (same element order in both layouts)
  CB(cublasDaxpy(g_blas, K * N, &mlr, bdW.p, 1, bW.p, 1));
  CK(cudaMemcpyAsync(loss_out, bScal.p, sizeof(double), cudaMemcpyDeviceToHost, g_stream));
  CK(cudaStreamSynchronize(g_stream));
  return 0;
}

int train_download_W(double* W_host) {
  CK(cudaMemcpyAsync(W_host, bW.p, (size_t)tc.K * tc.N * sizeof(double), cudaMemcpyDeviceToHost,
                     g_stream));
  CK(cudaStreamSynchronize(g_stream));
  return 0;
}

// ------------------------------- inference ---------------------------------
int infer_begin(int batch, int L, const int* d, const uint64_t* seedW, const double* scaleW,
                const uint64_t* seedB, double scaleB, uint64_t seedIn, double acc0) {
  ic.batch = batch;
  ic.L = L;
  ic.d.assign(d, d + L + 1);
  ic.woff.assign(L, 0);
  ic.boff.assign(L, 0);
  ic.maxd = *std::max_element(d, d + L + 1);
  size_t nw = 0, nb = 0;
  for (int l = 0; l < L; ++l) {
    ic.woff[l] = nw;
    ic.boff[l] = nb;
    nw += (size_t)d[l + 1] * d[l];
    nb += d[l + 1];
  }
  if (reserve({{&bWts, nw}, {&bBias, nb}, {&bIn, (size_t)batch * d[0]}, {&bA, (size_t)ic.maxd},
               {&bYv, (size_t)ic.maxd}, {&bPart, PART_N}, {&bScal, 4}}) != 0)
    return -1;
  for (int l = 0; l < L; ++l) {
    if (fill(bWts.p + ic.woff[l], (size_t)d[l + 1] * d[l], seedW[l], scaleW[l])) return -1;
    if (fill(bBias.p + ic.boff[l], d[l + 1], seedB[l], scaleB)) return -1;
  }
  if (fill(bIn.p, (size_t)batch * d[0], seedIn, 1.0)) return -1;
  CK(cudaMemcpyAsync(bScal.p + 1, &acc0, sizeof(double), cudaMemcpyHostToDevice, g_stream));
  CK(cudaStreamSynchronize(g_stream));
  return 0;
}

int infer_sample(int i) {
  const double one = 1.0;
  const double* x = bIn.p + (size_t)i * ic.d[0];
  for (int l = 0; l < ic.L; ++l) {
    const int m = ic.d[l + 1], n = ic.d[l];
    // y = b
    CK(cudaMemcpyAsync(bYv.p, bBias.p + ic.boff[l], m * sizeof(double), cudaMemcpyDeviceToDevice,
                       g_stream));
    // y = W x + y, W row-major (m x n) == col-major W^T (n x m, ld n) -> op T
    CB(cublasDgemv(g_blas, CUBLAS_OP_T, n, m, &one, bWts.p + ic.woff[l], n, x, 1, &one, bYv.p,
                   1));
    // a = ReLU(y)
    KLAUNCH(k_relu, (m + RT - 1) / RT, RT, g_stream, bYv.p, bA.p, m);
    CK(cudaGetLastError());
    x = bA.p;
  }
  if (reduce_into(bA.p, ic.d[ic.L], bScal.p + 1, true, 1.0)) return -1;  // checksum += sum(a)
  CK(cudaStreamSynchronize(g_stream));
  return 0;
}

int infer_get_acc(double* acc) {
  CK(cudaMemcpyAsync(acc, bScal.p + 1, sizeof(double), cudaMemcpyDeviceToHost, g_stream));
  CK(cudaStreamSynchronize(g_stream));
  return 0;
}

}  // namespace gpu
