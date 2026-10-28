// Host emulation of the small CUDA runtime / cuBLAS subset used by
// gpu_kernels.cu. TESTING ONLY: it lets the GPU code path (multi-stream job
// slots, cuBLAS argument mapping, kernels, reductions, checkpoint hand-off)
// be built and verified on machines without a GPU:   make EMU=1
//
// Execution model (mirrors what matters for correctness on a real device):
// * Work enqueued on a stream is executed ASYNCHRONOUSLY by emulated-device
//   worker thread(s), in order per stream; different streams interleave.
//   EMU_GPU_THREADS=<n> (default 1) worker threads = streams that can
//   progress at the same time.
// * Kernel arguments and host-pointer-mode cuBLAS scalars are captured at
//   enqueue time; DEVICE-pointer-mode scalars are read at execution time.
// * cudaMemcpyAsync from pageable host memory snapshots the source at enqueue;
//   to pageable host memory it is synchronous; pinned/device copies are deferred.
// * cudaEventQuery reports cudaErrorNotReady until the stream reaches the event.
// * cudaFree / cudaFreeHost synchronise the whole device (as CUDA does).
// * cuBLAS calls follow cuBLAS semantics (column-major, op flags, leading
//   dimension checks) and run on CBLAS in column-major mode. CBLAS is called
//   from the worker thread(s): link an OpenBLAS built with USE_LOCKING=1 (or a
//   threaded build) when the host also calls BLAS.
// * cudaMalloc fills new buffers with NaN bytes. EMU_GPU_MEM_MB=<n> caps device
//   memory (out-of-memory path); EMU_NO_GPU=1 hides the device;
//   EMU_NO_BATCHED=1 makes the batched GEMV unsupported (fallback path).
#pragma once
#include <cblas.h>

#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline

struct dim3 {
  unsigned x, y, z;
  dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {}
};
inline thread_local dim3 threadIdx(0, 0, 0), blockIdx(0, 0, 0), blockDim, gridDim;

inline double __dmul_rn(double a, double b) { return a * b; }
inline double __dadd_rn(double a, double b) { return a + b; }

namespace emu {

struct Stream {
  std::deque<std::function<void()>> q;
  bool busy = false;
};

struct Device {
  std::mutex m;
  std::condition_variable cv_work, cv_done;
  std::vector<Stream*> streams;
  std::vector<std::thread> workers;
  bool stop = false;
  size_t rr = 0;
  std::map<const char*, size_t> dev_allocs, pin_allocs;  // main thread only
  size_t in_use = 0;

  Device() {
    const char* e = std::getenv("EMU_GPU_THREADS");
    int n = e ? std::max(1, std::atoi(e)) : 1;
    for (int i = 0; i < n; ++i) workers.emplace_back([this] { loop(); });
  }
  ~Device() {
    {
      std::lock_guard<std::mutex> lk(m);
      stop = true;
    }
    cv_work.notify_all();
    for (auto& t : workers) t.join();
  }
  Stream* pick() {  // caller holds m
    size_t n = streams.size();
    for (size_t i = 0; i < n; ++i) {
      Stream* s = streams[(rr + i) % n];
      if (!s->busy && !s->q.empty()) {
        rr = (rr + i + 1) % n;
        return s;
      }
    }
    return nullptr;
  }
  void loop() {
    for (;;) {
      std::function<void()> f;
      Stream* s = nullptr;
      {
        std::unique_lock<std::mutex> lk(m);
        cv_work.wait(lk, [&] { return stop || (s = pick()) != nullptr; });
        if (!s) return;
        f = std::move(s->q.front());
        s->q.pop_front();
        s->busy = true;
      }
      f();
      {
        std::lock_guard<std::mutex> lk(m);
        s->busy = false;
      }
      cv_done.notify_all();
      cv_work.notify_all();
    }
  }
  bool idle(Stream* s) { return s->q.empty() && !s->busy; }
  void sync(Stream* s) {
    std::unique_lock<std::mutex> lk(m);
    cv_done.wait(lk, [&] { return idle(s); });
  }
  void sync_all() {
    std::unique_lock<std::mutex> lk(m);
    cv_done.wait(lk, [&] {
      for (Stream* s : streams)
        if (!idle(s)) return false;
      return true;
    });
  }
  void enqueue(Stream* s, std::function<void()> f) {
    if (!s) {  // legacy default stream: synchronous w.r.t. everything
      sync_all();
      f();
      return;
    }
    {
      std::lock_guard<std::mutex> lk(m);
      s->q.push_back(std::move(f));
    }
    cv_work.notify_one();
  }
  static bool within(const std::map<const char*, size_t>& mp, const void* p) {
    const char* c = static_cast<const char*>(p);
    auto it = mp.upper_bound(c);
    if (it == mp.begin()) return false;
    --it;
    return c < it->first + it->second;
  }
  bool is_dev(const void* p) { return within(dev_allocs, p); }
  bool is_pinned(const void* p) { return within(pin_allocs, p); }
};
inline Device& dev() {
  static Device d;
  return d;
}

template <class F>
void run_grid(dim3 g, dim3 b, F&& f) {
  gridDim = g;
  blockDim = b;
  for (unsigned bx = 0; bx < g.x; ++bx) {
    blockIdx = dim3(bx, 0, 0);
    for (unsigned tx = 0; tx < b.x; ++tx) {
      threadIdx = dim3(tx, 0, 0);
      f();
    }
  }
}
}  // namespace emu

// ----------------------------- runtime --------------------------------------
typedef int cudaError_t;
enum {
  cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorMemoryAllocation = 2,
  cudaErrorNotReady = 600
};
enum cudaMemcpyKind {
  cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice,
  cudaMemcpyDefault
};
typedef emu::Stream* cudaStream_t;
enum { cudaStreamNonBlocking = 1, cudaEventDisableTiming = 2 };
struct emu_event {
  bool done = true;  // guarded by emu::dev().m
};
typedef emu_event* cudaEvent_t;
struct cudaDeviceProp {
  char name[256];
  size_t totalGlobalMem;
};
enum cudaDeviceAttr {
  cudaDevAttrMultiProcessorCount, cudaDevAttrMemoryClockRate, cudaDevAttrGlobalMemoryBusWidth
};

namespace emu {
template <class K, class... A>
void launch_async(cudaStream_t s, dim3 g, dim3 b, K kern, A... args) {
  dev().enqueue(s, [=] { run_grid(g, b, [&] { kern(args...); }); });
}
inline size_t mem_cap() {
  const char* cap = std::getenv("EMU_GPU_MEM_MB");
  return cap ? (size_t)std::atoll(cap) << 20 : (size_t)16 << 30;
}
}  // namespace emu

inline const char* cudaGetErrorString(cudaError_t e) {
  switch (e) {
    case cudaSuccess: return "no error";
    case cudaErrorMemoryAllocation: return "out of memory";
    case cudaErrorNotReady: return "device not ready";
    default: return "invalid value";
  }
}
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int* n) {
  *n = std::getenv("EMU_NO_GPU") ? 0 : 1;
  return cudaSuccess;
}
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) {
  std::strcpy(p->name, "Emulated GPU (host)");
  p->totalGlobalMem = emu::mem_cap();
  return cudaSuccess;
}
inline cudaError_t cudaDeviceGetAttribute(int* v, cudaDeviceAttr a, int) {
  switch (a) {  // A100-80GB-like figures
    case cudaDevAttrMultiProcessorCount: *v = 108; break;
    case cudaDevAttrMemoryClockRate: *v = 1593000; break;  // kHz
    case cudaDevAttrGlobalMemoryBusWidth: *v = 5120; break;
  }
  return cudaSuccess;
}
inline cudaError_t cudaMemGetInfo(size_t* free_b, size_t* total_b) {
  *total_b = emu::mem_cap();
  *free_b = *total_b > emu::dev().in_use ? *total_b - emu::dev().in_use : 0;
  return cudaSuccess;
}
inline cudaError_t cudaMalloc(void** p, size_t n) {
  auto& d = emu::dev();
  if (d.in_use + n > emu::mem_cap()) return cudaErrorMemoryAllocation;
  *p = std::malloc(n ? n : 1);
  if (!*p) return cudaErrorMemoryAllocation;
  std::memset(*p, 0xFF, n);  // NaN poison
  d.dev_allocs[static_cast<const char*>(*p)] = n ? n : 1;
  d.in_use += n;
  return cudaSuccess;
}
inline cudaError_t cudaFree(void* p) {
  auto& d = emu::dev();
  d.sync_all();  // cudaFree synchronises the device
  auto it = d.dev_allocs.find(static_cast<const char*>(p));
  if (it != d.dev_allocs.end()) {
    d.in_use -= it->second;
    d.dev_allocs.erase(it);
  }
  std::free(p);
  return cudaSuccess;
}
// stream-ordered allocator: allocation is immediate, the free is ordered
// after the stream's prior work (no device-wide synchronisation)
inline cudaError_t cudaMallocAsync(void** p, size_t n, cudaStream_t) { return cudaMalloc(p, n); }
inline cudaError_t cudaFreeAsync(void* p, cudaStream_t s) {
  auto& d = emu::dev();
  auto it = d.dev_allocs.find(static_cast<const char*>(p));
  if (it != d.dev_allocs.end()) {
    d.in_use -= it->second;
    d.dev_allocs.erase(it);
  }
  d.enqueue(s, [p] { std::free(p); });
  return cudaSuccess;
}
inline cudaError_t cudaMallocHost(void** p, size_t n) {
  *p = std::malloc(n ? n : 1);
  if (!*p) return cudaErrorMemoryAllocation;
  emu::dev().pin_allocs[static_cast<const char*>(*p)] = n ? n : 1;
  return cudaSuccess;
}
inline cudaError_t cudaFreeHost(void* p) {
  auto& d = emu::dev();
  d.sync_all();
  d.pin_allocs.erase(static_cast<const char*>(p));
  std::free(p);
  return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind,
                                   cudaStream_t s) {
  auto& d = emu::dev();
  const bool dst_pageable = !d.is_dev(dst) && !d.is_pinned(dst);
  const bool src_pageable = !d.is_dev(src) && !d.is_pinned(src);
  if (dst_pageable) {  // D2H into pageable memory is synchronous
    if (s) d.sync(s); else d.sync_all();
    std::memmove(dst, src, n);
    return cudaSuccess;
  }
  if (src_pageable) {  // pageable source is staged at call time
    auto buf = std::make_shared<std::vector<char>>(static_cast<const char*>(src),
                                                   static_cast<const char*>(src) + n);
    d.enqueue(s, [=] { std::memcpy(dst, buf->data(), n); });
    return cudaSuccess;
  }
  d.enqueue(s, [=] { std::memmove(dst, src, n); });
  return cudaSuccess;
}
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
  emu::dev().sync_all();
  std::memmove(dst, src, n);
  return cudaSuccess;
}
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) {
  auto& d = emu::dev();
  *s = new emu::Stream;
  std::lock_guard<std::mutex> lk(d.m);
  d.streams.push_back(*s);
  return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t s) {
  auto& d = emu::dev();
  d.sync(s);
  {
    std::lock_guard<std::mutex> lk(d.m);
    for (size_t i = 0; i < d.streams.size(); ++i)
      if (d.streams[i] == s) {
        d.streams.erase(d.streams.begin() + i);
        break;
      }
  }
  delete s;
  return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t s) {
  if (s) emu::dev().sync(s); else emu::dev().sync_all();
  return cudaSuccess;
}
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) {
  *e = new emu_event;
  return cudaSuccess;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t e) {
  emu::dev().sync_all();
  delete e;
  return cudaSuccess;
}
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s) {
  auto& d = emu::dev();
  {
    std::lock_guard<std::mutex> lk(d.m);
    e->done = false;
  }
  d.enqueue(s, [e, &d] {
    std::lock_guard<std::mutex> lk(d.m);
    e->done = true;
  });
  return cudaSuccess;
}
inline cudaError_t cudaEventQuery(cudaEvent_t e) {
  auto& d = emu::dev();
  std::lock_guard<std::mutex> lk(d.m);
  return e->done ? cudaSuccess : cudaErrorNotReady;
}

// ------------------------------ cuBLAS --------------------------------------
enum cublasStatus_t {
  CUBLAS_STATUS_SUCCESS = 0, CUBLAS_STATUS_INVALID_VALUE = 7, CUBLAS_STATUS_NOT_SUPPORTED = 15
};
enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 };
enum cublasPointerMode_t { CUBLAS_POINTER_MODE_HOST = 0, CUBLAS_POINTER_MODE_DEVICE = 1 };
struct emu_blas {
  cudaStream_t s = nullptr;
  cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST;
};
typedef emu_blas* cublasHandle_t;

namespace emu {
struct Scalar {  // host mode: value captured now; device mode: read when executed
  const double* p;
  double v;
  double get() const { return p ? *p : v; }
};
inline Scalar scalar(cublasHandle_t h, const double* x) {
  return h->mode == CUBLAS_POINTER_MODE_DEVICE ? Scalar{x, 0.0} : Scalar{nullptr, *x};
}
inline int max1(int a) { return a > 1 ? a : 1; }
}  // namespace emu

inline cublasStatus_t cublasCreate(cublasHandle_t* h) {
  *h = new emu_blas;
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDestroy(cublasHandle_t h) {
  delete h;
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSetStream(cublasHandle_t h, cudaStream_t s) {
  h->s = s;
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSetPointerMode(cublasHandle_t h, cublasPointerMode_t m) {
  h->mode = m;
  return CUBLAS_STATUS_SUCCESS;
}

inline cublasStatus_t cublasDgemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                                  int m, int n, int k, const double* alpha, const double* A,
                                  int lda, const double* B, int ldb, const double* beta,
                                  double* C, int ldc) {
  if (m < 0 || n < 0 || k < 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (lda < emu::max1(ta == CUBLAS_OP_N ? m : k)) return CUBLAS_STATUS_INVALID_VALUE;
  if (ldb < emu::max1(tb == CUBLAS_OP_N ? k : n)) return CUBLAS_STATUS_INVALID_VALUE;
  if (ldc < emu::max1(m)) return CUBLAS_STATUS_INVALID_VALUE;
  auto al = emu::scalar(h, alpha), be = emu::scalar(h, beta);
  emu::dev().enqueue(h->s, [=] {
    cblas_dgemm(CblasColMajor, ta == CUBLAS_OP_N ? CblasNoTrans : CblasTrans,
                tb == CUBLAS_OP_N ? CblasNoTrans : CblasTrans, m, n, k, al.get(), A, lda, B, ldb,
                be.get(), C, ldc);
  });
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDgemv(cublasHandle_t h, cublasOperation_t t, int m, int n,
                                  const double* alpha, const double* A, int lda, const double* x,
                                  int incx, const double* beta, double* y, int incy) {
  if (m < 0 || n < 0 || lda < emu::max1(m) || incx == 0 || incy == 0)
    return CUBLAS_STATUS_INVALID_VALUE;
  auto al = emu::scalar(h, alpha), be = emu::scalar(h, beta);
  emu::dev().enqueue(h->s, [=] {
    cblas_dgemv(CblasColMajor, t == CUBLAS_OP_N ? CblasNoTrans : CblasTrans, m, n, al.get(), A,
                lda, x, incx, be.get(), y, incy);
  });
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDgemvStridedBatched(cublasHandle_t h, cublasOperation_t t, int m,
                                                int n, const double* alpha, const double* A,
                                                int lda, long long strideA, const double* x,
                                                int incx, long long stridex, const double* beta,
                                                double* y, int incy, long long stridey,
                                                int batch) {
  if (std::getenv("EMU_NO_BATCHED")) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (m < 0 || n < 0 || batch < 0 || lda < emu::max1(m) || incx == 0 || incy == 0)
    return CUBLAS_STATUS_INVALID_VALUE;
  auto al = emu::scalar(h, alpha), be = emu::scalar(h, beta);
  emu::dev().enqueue(h->s, [=] {
    for (int i = 0; i < batch; ++i)
      cblas_dgemv(CblasColMajor, t == CUBLAS_OP_N ? CblasNoTrans : CblasTrans, m, n, al.get(),
                  A + i * strideA, lda, x + i * stridex, incx, be.get(), y + i * stridey, incy);
  });
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDdot(cublasHandle_t h, int n, const double* x, int incx,
                                 const double* y, int incy, double* result) {
  if (h->mode == CUBLAS_POINTER_MODE_DEVICE) {
    emu::dev().enqueue(h->s, [=] { *result = cblas_ddot(n, x, incx, y, incy); });
  } else {  // host pointer mode: blocks until the result is available
    if (h->s) emu::dev().sync(h->s); else emu::dev().sync_all();
    *result = cblas_ddot(n, x, incx, y, incy);
  }
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDaxpy(cublasHandle_t h, int n, const double* alpha, const double* x,
                                  int incx, double* y, int incy) {
  auto al = emu::scalar(h, alpha);
  emu::dev().enqueue(h->s, [=] { cblas_daxpy(n, al.get(), x, incx, y, incy); });
  return CUBLAS_STATUS_SUCCESS;
}
