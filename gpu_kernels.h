// Host-side API of the GPU training / inference kernels (gpu_kernels.cu).
// Plain C++ types only, so sim.cpp never needs CUDA headers.
//
// The device is driven through JOB SLOTS. Each slot owns its own CUDA stream,
// cuBLAS handle (device pointer mode, so no call blocks the host), device
// buffers (grow-only, reused across tasks), pinned host scalars and events.
// One task occupies one slot; several slots run concurrently on the device,
// which is how the scheduler packs multiple tasks onto one GPU.
//
// Work is enqueued in UNITS (one training epoch, or a chunk of inference
// samples processed with a batched GEMV). Each unit records an event in ring
// position `ring` (0..RING-1); the host polls it with query(). Units of a slot
// execute in order, so "unit k done" implies all earlier units are done.
//
// All matrices are row-major on the host side (same layout as the CPU path),
// so checkpoints are interchangeable between the CPU and GPU implementations.
#pragma once
#include <cstddef>
#include <cstdint>

namespace gpu {

constexpr int RING = 4;          // max units in flight per slot
constexpr int MAX_SLOTS = 64;    // hard cap on concurrent jobs per process

struct DeviceInfo {
  char name[256];
  double mem_gb;       // total device memory
  int sm_count;        // streaming multiprocessors
  double mem_bw_gbs;   // peak DRAM bandwidth estimate (GB/s)
  int device;          // CUDA device id
};

bool compiled();        // false in CPU-only builds (gpu_stub.cpp)
const char* backend();  // "CUDA + cuBLAS", "host emulation", "none"
const char* last_error();

// Select device (local_rank % device_count). `slots` = job slots to prepare
// (streams are created lazily). Returns 0 or -1 (see last_error()).
int init(int local_rank, int slots, DeviceInfo* info);
void shutdown();
int mem_info(size_t* free_bytes, size_t* total_bytes);
int max_chunk();        // largest inference chunk (samples per unit)

// Release the device buffers held by an idle slot (to make room for others).
void release_slot(int slot);

// Return codes of the *_begin functions
constexpr int OK = 0, ERR = -1, OOM = -2;

// ---- training: Y = X W ; dW = X^T dY ; dX = dY W^T ; W -= lr dW ---------
// X (BxK), T (BxN) are generated on the device from seedX / seedT exactly as
// the CPU path does. W (KxN) is uploaded from W_host, or generated from seedW
// with scale w_scale when W_host is null. lr = B / max(||X||^2, 1e-12) is
// computed on the device. Asynchronous except for the W upload staging.
int train_begin(int slot, int B, int K, int N, uint64_t seedX, uint64_t seedT, uint64_t seedW,
                double w_scale, const double* W_host);
int train_enqueue_epoch(int slot, int ring);  // loss of the epoch -> ring value
int train_download_W(int slot, double* W_host);  // synchronous, K*N doubles

// ---- inference: per input vector, per layer  y = W a + b ; a = ReLU(y) ---
// d[0..L] are layer widths. Weights, biases and inputs are generated on the
// device. acc0 seeds the running output checksum (resume support).
int infer_begin(int slot, int batch, int L, const int* d, const uint64_t* seedW,
                const double* scaleW, const uint64_t* seedB, double scaleB, uint64_t seedIn,
                double acc0);
// Samples [first, first+count) through all layers with cublasDgemvStridedBatched
// (one GEMV per sample, shared weights); checksum += sum of outputs.
int infer_enqueue_chunk(int slot, int first, int count, int ring);
int infer_get_acc(int slot, double* acc);  // synchronous

// ---- completion -----------------------------------------------------------
int query(int slot, int ring);          // 1 = unit done, 0 = still running, -1 = error
double ring_value(int slot, int ring);  // training loss of that unit (valid once done)
int sync(int slot);                     // wait for all work of the slot

}  // namespace gpu
