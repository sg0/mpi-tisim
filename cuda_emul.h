// Host emulation of the small CUDA runtime / cuBLAS subset used by
// gpu_kernels.cu. TESTING ONLY: it lets the GPU code path (cuBLAS argument
// mapping, kernels, reductions, checkpoint hand-off) be built and verified on
// machines without a GPU:   make EMU=1
//
// * cuBLAS calls follow cuBLAS semantics (column-major, op flags, leading
//   dimension checks returning CUBLAS_STATUS_INVALID_VALUE) and are executed by
//   CBLAS in column-major mode.
// * Kernels run sequentially over blockIdx.x / threadIdx.x (1-D launches only;
//   the kernels use no shared memory or __syncthreads).
// * cudaMalloc fills new buffers with NaN bytes to catch reads of
//   uninitialised device memory. EMU_GPU_MEM_MB=<n> caps total "device" memory
//   to exercise the out-of-memory -> CPU fallback path.
#pragma once
#include <cblas.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline

struct dim3 {
  unsigned x, y, z;
  dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {}
};
inline dim3 threadIdx(0, 0, 0), blockIdx(0, 0, 0), blockDim, gridDim;

inline double __dmul_rn(double a, double b) { return a * b; }
inline double __dadd_rn(double a, double b) { return a + b; }

namespace emu {
template <class F>
void launch(dim3 g, dim3 b, F&& f) {
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
inline std::map<void*, size_t>& allocs() {
  static std::map<void*, size_t> m;
  return m;
}
inline size_t& in_use() {
  static size_t u = 0;
  return u;
}
}  // namespace emu

// ----------------------------- runtime --------------------------------------
typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorMemoryAllocation = 2 };
enum cudaMemcpyKind {
  cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice,
  cudaMemcpyDefault
};
struct emu_stream {};
typedef emu_stream* cudaStream_t;
enum { cudaStreamNonBlocking = 1 };
struct cudaDeviceProp {
  char name[256];
  size_t totalGlobalMem;
};

inline const char* cudaGetErrorString(cudaError_t e) {
  return e == cudaSuccess ? "no error" : e == cudaErrorMemoryAllocation ? "out of memory"
                                                                        : "invalid value";
}
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int* n) {
  *n = std::getenv("EMU_NO_GPU") ? 0 : 1;
  return cudaSuccess;
}
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) {
  std::strcpy(p->name, "Emulated GPU (host)");
  const char* cap = std::getenv("EMU_GPU_MEM_MB");
  p->totalGlobalMem = cap ? (size_t)std::atoll(cap) << 20 : (size_t)16 << 30;
  return cudaSuccess;
}
inline cudaError_t cudaMalloc(void** p, size_t n) {
  const char* cap = std::getenv("EMU_GPU_MEM_MB");
  if (cap && emu::in_use() + n > ((size_t)std::atoll(cap) << 20)) return cudaErrorMemoryAllocation;
  *p = std::malloc(n ? n : 1);
  if (!*p) return cudaErrorMemoryAllocation;
  std::memset(*p, 0xFF, n);  // NaN poison
  emu::allocs()[*p] = n;
  emu::in_use() += n;
  return cudaSuccess;
}
inline cudaError_t cudaFree(void* p) {
  auto it = emu::allocs().find(p);
  if (it != emu::allocs().end()) {
    emu::in_use() -= it->second;
    emu::allocs().erase(it);
  }
  std::free(p);
  return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind, cudaStream_t) {
  std::memmove(d, s, n);
  return cudaSuccess;
}
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) {
  *s = new emu_stream;
  return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t s) {
  delete s;
  return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }

// ------------------------------ cuBLAS --------------------------------------
enum cublasStatus_t { CUBLAS_STATUS_SUCCESS = 0, CUBLAS_STATUS_INVALID_VALUE = 7 };
enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 };
struct emu_blas {};
typedef emu_blas* cublasHandle_t;

inline cublasStatus_t cublasCreate(cublasHandle_t* h) {
  *h = new emu_blas;
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDestroy(cublasHandle_t h) {
  delete h;
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t) { return CUBLAS_STATUS_SUCCESS; }

inline int emu_max1(int a) { return a > 1 ? a : 1; }

inline cublasStatus_t cublasDgemm(cublasHandle_t, cublasOperation_t ta, cublasOperation_t tb,
                                  int m, int n, int k, const double* alpha, const double* A,
                                  int lda, const double* B, int ldb, const double* beta,
                                  double* C, int ldc) {
  if (m < 0 || n < 0 || k < 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (lda < emu_max1(ta == CUBLAS_OP_N ? m : k)) return CUBLAS_STATUS_INVALID_VALUE;
  if (ldb < emu_max1(tb == CUBLAS_OP_N ? k : n)) return CUBLAS_STATUS_INVALID_VALUE;
  if (ldc < emu_max1(m)) return CUBLAS_STATUS_INVALID_VALUE;
  cblas_dgemm(CblasColMajor, ta == CUBLAS_OP_N ? CblasNoTrans : CblasTrans,
              tb == CUBLAS_OP_N ? CblasNoTrans : CblasTrans, m, n, k, *alpha, A, lda, B, ldb,
              *beta, C, ldc);
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDgemv(cublasHandle_t, cublasOperation_t t, int m, int n,
                                  const double* alpha, const double* A, int lda, const double* x,
                                  int incx, const double* beta, double* y, int incy) {
  if (m < 0 || n < 0 || lda < emu_max1(m) || incx == 0 || incy == 0)
    return CUBLAS_STATUS_INVALID_VALUE;
  cblas_dgemv(CblasColMajor, t == CUBLAS_OP_N ? CblasNoTrans : CblasTrans, m, n, *alpha, A, lda,
              x, incx, *beta, y, incy);
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDdot(cublasHandle_t, int n, const double* x, int incx,
                                 const double* y, int incy, double* result) {
  *result = cblas_ddot(n, x, incx, y, incy);
  return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasDaxpy(cublasHandle_t, int n, const double* alpha, const double* x,
                                  int incx, double* y, int incy) {
  cblas_daxpy(n, *alpha, x, incx, y, incy);
  return CUBLAS_STATUS_SUCCESS;
}
