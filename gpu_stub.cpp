// CPU-only build: GPU entry points that report "not available".
#include "gpu_kernels.h"

namespace gpu {
static const char* kMsg = "mpisim was built without GPU support (rebuild with: make GPU=1)";
bool compiled() { return false; }
const char* backend() { return "none"; }
const char* last_error() { return kMsg; }
int init(int, char*, int, double*) { return -1; }
void shutdown() {}
int train_begin(int, int, int, uint64_t, uint64_t, uint64_t, double, const double*, double*) {
  return -1;
}
int train_epoch(double, double*) { return -1; }
int train_download_W(double*) { return -1; }
int infer_begin(int, int, const int*, const uint64_t*, const double*, const uint64_t*, double,
                uint64_t, double) {
  return -1;
}
int infer_sample(int) { return -1; }
int infer_get_acc(double*) { return -1; }
}  // namespace gpu
