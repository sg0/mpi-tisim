// Host-side API of the GPU training / inference kernels (gpu_kernels.cu).
// Plain C++ types only, so sim.cpp never needs CUDA headers.
//
// Each MPI rank drives at most one GPU task at a time; device buffers are a
// grow-only pool that is reused across tasks. All matrices are row-major on the
// host side (same layout as the CPU path), so checkpoints are interchangeable
// between the CPU and GPU implementations.
#pragma once
#include <cstddef>
#include <cstdint>

namespace gpu {

bool compiled();            // false in CPU-only builds (gpu_stub.cpp)
const char* backend();      // "CUDA + cuBLAS", "host emulation", "none"
const char* last_error();

// Select device (local_rank % device_count), create stream + cuBLAS handle.
// Returns the device id, or -1 on failure (see last_error()).
int init(int local_rank, char* name, int name_len, double* mem_gb);
void shutdown();

// ---- training: Y = X W ; dW = X^T dY ; dX = dY W^T ; W -= lr dW ----------
// X (BxK), T (BxN) are generated on the device from seedX / seedT exactly as
// the CPU path does. W (KxN) is uploaded from W_host, or generated from seedW
// with scale w_scale when W_host is null. Returns 0, or -1 if device memory
// could not be allocated (caller falls back to the CPU kernel).
int train_begin(int B, int K, int N, uint64_t seedX, uint64_t seedT, uint64_t seedW,
                double w_scale, const double* W_host, double* lr_out);
int train_epoch(double lr, double* loss_out);  // one epoch, synchronous
int train_download_W(double* W_host);          // K*N doubles

// ---- inference: per input vector, per layer  y = W a + b ; a = ReLU(y) ----
// d[0..L] are the layer widths. Weights, biases and inputs are generated on
// the device. acc0 seeds the running output checksum (resume support).
int infer_begin(int batch, int L, const int* d, const uint64_t* seedW, const double* scaleW,
                const uint64_t* seedB, double scaleB, uint64_t seedIn, double acc0);
int infer_sample(int i);       // push input vector i through all layers, synchronous
int infer_get_acc(double* acc);

}  // namespace gpu
