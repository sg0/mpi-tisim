// CPU-only build: GPU entry points that report "not available".
#include "gpu_kernels.h"

namespace gpu {
static const char* kMsg = "mpisim was built without GPU support (rebuild with: make GPU=1)";
bool compiled() { return false; }
const char* backend() { return "none"; }
const char* last_error() { return kMsg; }
int init(int, int, DeviceInfo*) { return ERR; }
void shutdown() {}
int mem_info(size_t*, size_t*) { return ERR; }
int max_chunk() { return 1; }
void release_slot(int) {}
int train_begin(int, int, int, int, uint64_t, uint64_t, uint64_t, double, const double*) {
  return ERR;
}
int train_enqueue_epoch(int, int) { return ERR; }
int train_download_W(int, double*) { return ERR; }
int infer_begin(int, int, int, const int*, const uint64_t*, const double*, const uint64_t*,
                double, uint64_t, double) {
  return ERR;
}
int infer_enqueue_chunk(int, int, int, int) { return ERR; }
int infer_get_acc(int, double*) { return ERR; }
int query(int, int) { return -1; }
double ring_value(int, int) { return 0.0; }
int sync(int) { return ERR; }
}  // namespace gpu
