// ============================================================================
//  mpisim — MPI training / inference workload simulator (DGEMM / DGEMV)
//
//  * Training task  : fixed #epochs of  fwd DGEMM, 2x bwd DGEMM (dW, dX), DAXPY
//  * Inference task : batch of single-vector inputs through L layers
//                     (DGEMV + bias + ReLU per layer)
//  * 100 random shape combinations each (dimensions in the thousands, default
//    [256, 4096])
//  * CPU kernels: OpenBLAS (cblas_*).  GPU kernels: cuBLAS (gpu_kernels.cu).
//    exec_mode:cpu | exec_mode:gpu | exec_mode:cpu,gpu  — with cpu,gpu every
//    task whose precalculated total FLOPs exceed 1 TFLOP (--gpu-threshold)
//    runs on the GPU, the rest on the CPU
//  * Capacity-aware packing (exec_mode gpu / cpu,gpu): GPU capacity =
//    --gpu-mult x calibrated CPU FLOP/s per process; every task gets a
//    modelled GPU demand (FLOP/s it can usefully consume, from its shape), and
//    tasks are packed onto separate CUDA streams until the device's capacity
//    or stream limit is reached. The CPU runs its own task concurrently on a
//    worker thread, so CPU and GPU are both kept busy.
//  * Dynamic self-scheduling through a global task counter (NXTVAL) implemented
//    with MPI-3 passive-target RMA (MPI_Fetch_and_op, MPI_SUM)
//  * Runtime control (priority / suspend / resume / exit) from a monitor thread
//    on rank 0, published to every rank through the same RMA window
//  * Checkpointing of partial training and inference work, per-rank files
//
//  Control tokens (initial ones on the command line, live ones via the control
//  file or stdin of rank 0):
//      p:<0..1>    probability of choosing an inference task (default 0.5)
//      s:<rank>    suspend rank (checkpoint + stop taking tasks)   s:all ok
//      r:<rank>    resume rank (drain its checkpoints first)        r:all ok
//      e:0         exit now, do NOT save checkpoints
//      e:1         exit now, save checkpoints
// ============================================================================
#include <mpi.h>
#include <cblas.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <poll.h>
#include <sys/stat.h>
#include <unistd.h>

#include "gpu_kernels.h"

// Keep OpenBLAS single-threaded per rank (MPI already provides the parallelism).
// Weak so the code still links against a non-OpenBLAS CBLAS.
extern "C" void openblas_set_num_threads(int) __attribute__((weak));

// ----------------------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------------------
enum ExecMode { EXEC_CPU = 1, EXEC_GPU = 2, EXEC_BOTH = 3 };
static constexpr int DIM_CAP = 8192;

struct Config {
  int epochs = 40;             // training epochs per task
  int nshapes = 100;           // shape combinations per workload
  int min_dim = 256;           // random dims in [min_dim, max_dim]
  int max_dim = 4096;          // hard cap DIM_CAP
  int exec_mode = EXEC_CPU;    // exec_mode:cpu | gpu | cpu,gpu
  double gpu_threshold = 1e12; // cpu,gpu: tasks above this many FLOPs go to the GPU
  int max_layers = 8;          // inference layers in [2, max_layers]
  uint64_t seed = 20261023ULL;
  std::string ctl_file = "sim_control.txt";
  std::string ckpt_dir = "checkpoints";
  std::string timeline_file = "sim_timeline.csv";
  bool fresh = false;          // ignore / delete checkpoints from a previous run
  bool use_stdin = true;
  bool selftest = false;
  int sync_every = 8;          // decisions between global-mix syncs
  double poll_ms = 0.5;        // control poll interval inside kernels
  double timeline_ms = 250.0;  // timeline sample interval
  double max_seconds = 0.0;    // >0: inject e:1 after this many seconds
  int verbose = 1;
  // ---- capacity model / packing (exec_mode gpu, cpu,gpu) ----
  double cpu_gflops = 0.0;     // CPU GFLOP/s per process (0 = calibrate at start)
  double gpu_mult = 100.0;     // GPU FLOP/s = gpu_mult x CPU FLOP/s per process
  double gpu_gflops = 0.0;     // >0: absolute GPU GFLOP/s per device (overrides mult)
  int gpu_streams = 16;        // max concurrent streams (jobs) per device
  bool pack = true;            // false: one task at a time per rank (baseline)
  bool spill = true;           // cpu,gpu: CPU-routed tasks may use spare GPU capacity
  bool rebalance = true;       // cpu,gpu: idle CPU takes a waiting GPU task if it finishes sooner
  int gpu_depth = 2;           // units in flight per GPU job
  int gpu_tile = 64;           // DGEMM output tile edge for the occupancy model
  int ctas_per_sm = 2;         // concurrent GEMM tiles per SM
  double launch_us = 5.0;      // per kernel launch overhead in the demand model
  double unit_ms = 20.0;       // target duration of one inference GPU unit (chunk)
  double gpu_poll_us = 100.0;  // orchestrator back-off when nothing progressed
  double max_wait_ms = 200.0;  // a blocked GPU task older than this stops backfilling
  std::vector<std::string> init_tokens;  // p:/s:/r:/e: given on the command line
};

static Config cfg;
static int g_rank = 0, g_np = 1;
static double g_t0 = 0.0;

#define LOG(...)                                                       \
  do {                                                                 \
    if (cfg.verbose) {                                                 \
      std::printf("[%9.3fs r%-3d] ", MPI_Wtime() - g_t0, g_rank);      \
      std::printf(__VA_ARGS__);                                        \
      std::printf("\n");                                               \
      std::fflush(stdout);                                             \
    }                                                                  \
  } while (0)

// ----------------------------------------------------------------------------
// Deterministic RNG (data for a task is a pure function of seed + task id, so
// checkpoints only need the mutable state, never the inputs)
// ----------------------------------------------------------------------------
static inline uint64_t splitmix(uint64_t& x) {
  uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static inline uint64_t mix3(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t s = a ^ (b * 0xd1b54a32d192ed03ULL) ^ (c * 0xabc98388fb8fac03ULL);
  splitmix(s);
  return splitmix(s);
}
static void fill_uniform(double* a, size_t n, uint64_t seed, double scale) {
  uint64_t s = seed;
  for (size_t i = 0; i < n; ++i)
    a[i] = scale * (((splitmix(s) >> 11) * 0x1.0p-53) * 2.0 - 1.0);
}

// ----------------------------------------------------------------------------
// Shapes
// ----------------------------------------------------------------------------
enum TaskType { TRAIN = 0, INFER = 1 };
static const char* tname(int t) { return t == TRAIN ? "train" : "infer"; }

struct TrainShape { int B, K, N; };  // X: BxK, W: KxN, Y: BxN
static constexpr int MAXL = 16;
struct InferShape { int batch, L; int d[MAXL + 1]; };  // layer l: d[l] -> d[l+1]

static std::vector<TrainShape> g_tshapes;
static std::vector<InferShape> g_ishapes;

static void make_shapes() {
  std::mt19937_64 rng(cfg.seed);
  std::uniform_int_distribution<int> dim(cfg.min_dim, cfg.max_dim);
  std::uniform_int_distribution<int> lay(2, cfg.max_layers);
  g_tshapes.resize(cfg.nshapes);
  g_ishapes.resize(cfg.nshapes);
  for (auto& s : g_tshapes) s = {dim(rng), dim(rng), dim(rng)};
  for (auto& s : g_ishapes) {
    s.batch = dim(rng);
    s.L = lay(rng);
    for (int l = 0; l <= s.L; ++l) s.d[l] = dim(rng);
  }
}
static inline double train_epoch_flops(const TrainShape& s) {
  // fwd 2BKN + dW 2BKN + dX 2BKN + DAXPY 2KN
  return 6.0 * s.B * s.K * s.N + 2.0 * s.K * s.N;
}
static inline double infer_sample_flops(const InferShape& s) {
  double f = 0;  // DGEMV 2*m*n + m (beta*y bias add)
  for (int l = 0; l < s.L; ++l) f += 2.0 * s.d[l + 1] * s.d[l] + s.d[l + 1];
  return f;
}

// Total FLOPs of a whole task of a given shape (precalculated, used for routing)
static std::vector<double> g_task_flops[2];
static void precompute_task_flops() {
  g_task_flops[0].resize(cfg.nshapes);
  g_task_flops[1].resize(cfg.nshapes);
  for (int i = 0; i < cfg.nshapes; ++i) {
    g_task_flops[0][i] = train_epoch_flops(g_tshapes[i]) * cfg.epochs;
    g_task_flops[1][i] = infer_sample_flops(g_ishapes[i]) * g_ishapes[i].batch;
  }
}
static bool g_gpu_ready = false;
// Device routing: cpu -> CPU, gpu -> GPU, cpu,gpu -> GPU iff task FLOPs > threshold
static bool route_to_gpu(int type, int shape) {
  if (!g_gpu_ready || cfg.exec_mode == EXEC_CPU) return false;
  if (cfg.exec_mode == EXEC_GPU) return true;
  return g_task_flops[type][shape] > cfg.gpu_threshold;
}

// ----------------------------------------------------------------------------
// Task / checkpoint representation
// ----------------------------------------------------------------------------
enum Origin { ORIG_NONE = 0, ORIG_PREEMPT = 1, ORIG_SUSPEND = 2, ORIG_EXIT = 3, ORIG_PREVRUN = 4 };
static const char* oname(int o) {
  static const char* n[] = {"-", "priority-preempt", "suspend", "exit", "previous-run"};
  return n[o];
}

struct Task {
  int type = TRAIN;
  int origin = ORIG_NONE;
  int64_t id = -1;
  int shape = 0;
  int progress = 0;       // epochs done (train) / samples done (infer)
  double acc = 0.0;       // train: loss at epoch 0, infer: running output checksum
  double flops_run = 0;   // flops executed for this task during *this* run
  std::vector<double> W;  // train: current weights (KxN)
};

// GPU_UNAVAILABLE: device memory could not be allocated -> run on the CPU instead
enum Status { CONTINUE, COMPLETED, PREEMPTED, SUSPENDED, EXIT_SAVE, EXIT_NOSAVE, GPU_UNAVAILABLE };

// ----------------------------------------------------------------------------
// Statistics
// ----------------------------------------------------------------------------
#define STAT_FIELDS(X)                                                        \
  X(loops) X(idle_loops) X(task_runs)                                         \
  X(train_done) X(infer_done) X(train_new) X(infer_new)                       \
  X(train_resumed) X(infer_resumed) X(preempts) X(suspends) X(resumes)        \
  X(ckpt_writes) X(ckpt_bytes) X(nxtval_calls) X(ctl_polls) X(ctl_changes)    \
  X(syncs) X(flops_train) X(flops_infer) X(exec_train) X(exec_infer)          \
  X(flops_pending) X(flops_discarded) X(t_train) X(t_infer) X(t_idle)         \
  X(t_ctl) X(t_sched) X(t_ckpt) X(wall) X(loss_ratio_sum) X(epochs_run)       \
  X(samples_run) X(pending_left) X(checksum)                                  \
  X(train_done_gpu) X(infer_done_gpu) X(flops_train_gpu) X(flops_infer_gpu)   \
  X(exec_gpu) X(t_gpu) X(gpu_fallbacks) X(gpu_dev)                           \
  X(t_busy) X(cap_cpu) X(cap_gpu) X(slot_limit) X(gpu_jobs) X(conc_int)       \
  X(max_conc) X(dem_int) X(spilled) X(sched_iters) X(sharers) X(t_cpu_occ)   \
  X(rebalanced)

struct Stats {
#define DECL(n) double n = 0;
  STAT_FIELDS(DECL)
#undef DECL
  static constexpr int count() {
    int c = 0;
#define CNT(n) ++c;
    STAT_FIELDS(CNT)
#undef CNT
    return c;
  }
  void to(double* v) const {
    int i = 0;
#define TO(n) v[i++] = n;
    STAT_FIELDS(TO)
#undef TO
  }
  void from(const double* v) {
    int i = 0;
#define FROM(n) n = v[i++];
    STAT_FIELDS(FROM)
#undef FROM
  }
};
// thread_local: in the concurrent engine the CPU kernels run on a worker thread
// with their own counters, merged into the main thread's at the end.
static thread_local Stats st;
static thread_local bool tl_in_worker = false;
static bool g_concurrent = false;  // exec_mode gpu / cpu,gpu engine active

struct Sample { double t, exec_train, exec_infer, busy, idle, susp, p, exec_gpu; };
static constexpr int SAMPLE_DOUBLES = 8;
static std::vector<Sample> g_timeline;
static double g_next_sample = 0.0;

// ----------------------------------------------------------------------------
// RMA window: global counters + published control state (lives on rank 0)
// ----------------------------------------------------------------------------
enum { W_NXT = 0, W_GTRAIN = 1, W_GINF = 2, W_VER = 3, W_PRIO = 4, W_EXIT = 5, W_SUSP0 = 6 };
static MPI_Win g_win;
static int64_t* g_winbuf = nullptr;
static constexpr double PRIO_SCALE = 1e9;

// Local view of the control state (every rank)
static double cur_p = 0.5;
static int64_t cur_exit = -1;
static bool susp_me = false;
static bool prio_event = false;  // priority moved towards inference since last check
static int64_t seen_ver = -1;
static double last_poll = -1e30;
static double poll_s = 0.0005;
static bool selftest_mode = false;
static int64_t g_inf_seen = 0, g_train_seen = 0;  // last synced global mix

// Rank-0 side: state written by the monitor thread, read by the main thread
struct CtlShared {
  std::mutex m;
  double p = 0.5;
  int64_t exit_mode = -1;
  std::vector<int64_t> susp;
  std::atomic<uint64_t> version{0};
};
static CtlShared ctl;
static uint64_t published_ver = 0;
static std::atomic<bool> monitor_stop{false};

static int64_t rma_fetch_add(int disp, int64_t inc) {
  int64_t res = 0;
  MPI_Fetch_and_op(&inc, &res, MPI_INT64_T, 0, disp, MPI_SUM, g_win);
  MPI_Win_flush(0, g_win);
  return res;
}
static int64_t rma_read(int disp) {
  int64_t res = 0;
  MPI_Fetch_and_op(nullptr, &res, MPI_INT64_T, 0, disp, MPI_NO_OP, g_win);
  MPI_Win_flush(0, g_win);
  return res;
}

// NXTVAL: atomically fetch the global task counter and add 1
static int64_t nxtval() {
  st.nxtval_calls++;
  return rma_fetch_add(W_NXT, 1);
}

// Apply one control token (rank 0 monitor thread / initial argv)
static bool apply_token(const std::string& tok, bool quiet = false) {
  if (tok.size() < 3 || tok[1] != ':') return false;
  char c = tok[0];
  std::string v = tok.substr(2);
  std::lock_guard<std::mutex> lk(ctl.m);
  char* end = nullptr;
  if (c == 'p') {
    double p = std::strtod(v.c_str(), &end);
    if (end == v.c_str() || *end || !(p >= 0.0 && p <= 1.0)) {
      std::fprintf(stderr, "[ctl] bad priority '%s' (need 0..1)\n", tok.c_str());
      return false;
    }
    if (!quiet) LOG("[ctl] priority p: %.3f -> %.3f (P(inference))", ctl.p, p);
    ctl.p = p;
  } else if (c == 's' || c == 'r') {
    int64_t val = (c == 's') ? 1 : 0;
    if (v == "all") {
      std::fill(ctl.susp.begin(), ctl.susp.end(), val);
    } else {
      long r = std::strtol(v.c_str(), &end, 10);
      if (end == v.c_str() || *end || r < 0 || r >= g_np) {
        std::fprintf(stderr, "[ctl] bad rank in '%s' (np=%d)\n", tok.c_str(), g_np);
        return false;
      }
      ctl.susp[r] = val;
    }
    if (!quiet) LOG("[ctl] %s rank %s", c == 's' ? "suspend" : "resume", v.c_str());
  } else if (c == 'e') {
    if (v != "0" && v != "1") {
      std::fprintf(stderr, "[ctl] bad exit token '%s' (e:0 or e:1)\n", tok.c_str());
      return false;
    }
    ctl.exit_mode = (v == "1") ? 1 : 0;
    if (!quiet) LOG("[ctl] exit requested (%s checkpoints)", v == "1" ? "saving" : "NOT saving");
  } else {
    std::fprintf(stderr, "[ctl] unknown token '%s'\n", tok.c_str());
    return false;
  }
  ctl.version.fetch_add(1);
  return true;
}

static void apply_line(const std::string& line) {
  std::string s = line;
  for (char& ch : s)
    if (ch == ',' || ch == ';') ch = ' ';
  std::istringstream is(s);
  std::string tok;
  while (is >> tok) apply_token(tok);
}

// Monitor thread (rank 0 only, never calls MPI): watches stdin and the
// control file for new tokens. argc/argv are fixed once a process starts, so
// "the argument" is re-read from this live channel; the initial values come
// from argv.
static void monitor_thread() {
  bool stdin_ok = cfg.use_stdin;
  std::string sbuf;
  off_t off = 0;
  double start = MPI_Wtime();
  bool timed_exit = false;
  while (!monitor_stop.load()) {
    if (stdin_ok) {
      struct pollfd pfd{0, POLLIN, 0};
      int r = poll(&pfd, 1, 20);
      if (r > 0) {
        char buf[4096];
        ssize_t n = read(0, buf, sizeof buf);
        if (n <= 0) {
          stdin_ok = false;
        } else {
          sbuf.append(buf, n);
          size_t pos;
          while ((pos = sbuf.find('\n')) != std::string::npos) {
            apply_line(sbuf.substr(0, pos));
            sbuf.erase(0, pos + 1);
          }
        }
      } else if (r < 0 && errno != EINTR) {
        stdin_ok = false;
      }
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    struct stat sb;
    if (stat(cfg.ctl_file.c_str(), &sb) == 0) {
      if (sb.st_size < off) off = 0;  // file was truncated / rewritten
      if (sb.st_size > off) {
        FILE* f = std::fopen(cfg.ctl_file.c_str(), "r");
        if (f) {
          std::fseek(f, off, SEEK_SET);
          std::string data(sb.st_size - off, '\0');
          size_t n = std::fread(&data[0], 1, data.size(), f);
          std::fclose(f);
          data.resize(n);
          size_t last = data.rfind('\n');  // only consume complete lines
          if (last != std::string::npos) {
            std::istringstream is(data.substr(0, last));
            std::string line;
            while (std::getline(is, line)) apply_line(line);
            off += last + 1;
          }
        }
      }
    }
    if (cfg.max_seconds > 0 && !timed_exit && MPI_Wtime() - start > cfg.max_seconds) {
      timed_exit = true;
      LOG("[ctl] --max-seconds reached, injecting e:1");
      apply_token("e:1", true);
    }
  }
}

// Rank 0 main thread: push the monitor's state into the RMA window.
// Fields are written first, version last, so a reader that observes a new
// version is guaranteed to read the matching fields.
static void publish_if_needed() {
  uint64_t v = ctl.version.load();
  if (v == published_ver) return;
  std::vector<int64_t> vals(2 + g_np);
  {
    std::lock_guard<std::mutex> lk(ctl.m);
    v = ctl.version.load();
    vals[0] = (int64_t)std::llround(ctl.p * PRIO_SCALE);
    vals[1] = ctl.exit_mode;
    for (int r = 0; r < g_np; ++r) vals[2 + r] = ctl.susp[r];
  }
  int n = 2 + g_np;
  MPI_Accumulate(vals.data(), n, MPI_INT64_T, 0, W_PRIO, n, MPI_INT64_T, MPI_REPLACE, g_win);
  MPI_Win_flush(0, g_win);
  int64_t vv = (int64_t)v;
  MPI_Accumulate(&vv, 1, MPI_INT64_T, 0, W_VER, 1, MPI_INT64_T, MPI_REPLACE, g_win);
  MPI_Win_flush(0, g_win);
  published_ver = v;
}

// Every rank: read the published control state (atomic RMA reads).
static void poll_control() {
  double t0 = MPI_Wtime();
  st.ctl_polls++;
  if (g_rank == 0) publish_if_needed();
  int64_t ver = rma_read(W_VER);
  if (ver != seen_ver) {
    int n = 2 + g_np;
    std::vector<int64_t> blk(n);
    MPI_Get_accumulate(nullptr, 0, MPI_INT64_T, blk.data(), n, MPI_INT64_T, 0, W_PRIO, n,
                       MPI_INT64_T, MPI_NO_OP, g_win);
    MPI_Win_flush(0, g_win);
    double np_ = blk[0] / PRIO_SCALE;
    if (seen_ver >= 0) {
      st.ctl_changes++;
      if (np_ > cur_p + 1e-12) prio_event = true;
    }
    cur_p = np_;
    cur_exit = blk[1];
    susp_me = blk[2 + g_rank] != 0;
    seen_ver = ver;
  }
  last_poll = MPI_Wtime();
  st.t_ctl += last_poll - t0;
}

// ----------------------------------------------------------------------------
// CPU worker (concurrent engine only): runs the unchanged CPU kernels on a
// separate thread so the main thread can keep feeding GPU streams. It never
// calls MPI for communication; interrupts reach it through `intr`, which
// check_interrupt() reads between epochs / samples.
// ----------------------------------------------------------------------------
struct CpuWorker {
  std::thread th;
  std::mutex m;
  std::condition_variable cv;
  bool has = false, quit = false;  // guarded by m
  std::atomic<bool> done{false};
  bool running = false;            // main thread's view
  Task task;
  Status result = CONTINUE;
  bool from_drain = false;
  std::atomic<int> intr{CONTINUE};
  std::atomic<double> pub_exec_train{0}, pub_exec_infer{0};
  Stats final_stats;               // worker's counters, handed over at exit
};
static CpuWorker g_cpu;

static void worker_publish() {
  g_cpu.pub_exec_train.store(st.exec_train, std::memory_order_relaxed);
  g_cpu.pub_exec_infer.store(st.exec_infer, std::memory_order_relaxed);
}

static double cum_exec_train() {
  return st.exec_train + (g_concurrent ? g_cpu.pub_exec_train.load() : 0.0);
}
static double cum_exec_infer() {
  return st.exec_infer + (g_concurrent ? g_cpu.pub_exec_infer.load() : 0.0);
}
static double cum_busy() { return g_concurrent ? st.t_busy : st.t_train + st.t_infer; }

static void sample_timeline(double now, bool force = false) {
  if (tl_in_worker) return;  // main thread only
  if (!force && now < g_next_sample) return;
  g_timeline.push_back({now - g_t0, cum_exec_train(), cum_exec_infer(), cum_busy(), st.t_idle,
                        susp_me ? 1.0 : 0.0, cur_p, st.exec_gpu});
  g_next_sample = now + cfg.timeline_ms * 1e-3;
}

// Called between units of work (epochs / samples) inside a task.
static Status check_interrupt(int type) {
  if (selftest_mode) return CONTINUE;
  if (tl_in_worker) {  // concurrent engine: the orchestrator decides
    worker_publish();
    int s = g_cpu.intr.load(std::memory_order_acquire);
    if (s == PREEMPTED && type != TRAIN) return CONTINUE;
    return s == CONTINUE ? CONTINUE : (Status)s;
  }
  double now = MPI_Wtime();
  sample_timeline(now);
  if (now - last_poll < poll_s) return CONTINUE;
  poll_control();
  if (cur_exit == 0) return EXIT_NOSAVE;
  if (cur_exit == 1) return EXIT_SAVE;
  if (susp_me) return SUSPENDED;
  if (type == TRAIN && prio_event) return PREEMPTED;
  return CONTINUE;
}

// ----------------------------------------------------------------------------
// Kernels
// ----------------------------------------------------------------------------
static double g_sink = 0.0;  // keeps dX "used"

// Training: single linear layer, MSE loss, SGD.  Data regenerated from task id.
static Status run_training(Task& t, int stop_at = -1) {
  const TrainShape& s = g_tshapes[t.shape];
  const int B = s.B, K = s.K, N = s.N;
  std::vector<double> X((size_t)B * K), T((size_t)B * N), Y((size_t)B * N), dY((size_t)B * N),
      dW((size_t)K * N), dX((size_t)B * K);
  fill_uniform(X.data(), X.size(), mix3(cfg.seed, t.id, 1), 1.0);
  fill_uniform(T.data(), T.size(), mix3(cfg.seed, t.id, 2), 1.0);
  if (t.W.empty()) {
    t.W.resize((size_t)K * N);
    fill_uniform(t.W.data(), t.W.size(), mix3(cfg.seed, t.id, 3), 1.0 / std::sqrt((double)K));
  }
  double xn = cblas_ddot(B * K, X.data(), 1, X.data(), 1);
  const double lr = (double)B / std::max(xn, 1e-12);  // 1/trace(X^T X / B): always stable
  const double ef = train_epoch_flops(s);
  double loss = 0.0;

  for (int e = t.progress; e < cfg.epochs; ++e) {
    if (e == stop_at) return PREEMPTED;
    Status cs = check_interrupt(TRAIN);
    if (cs != CONTINUE) return cs;  // state == "e epochs done", consistent
    double t0 = MPI_Wtime();
    // forward:  Y = X W                                    (BxK)(KxN)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, B, N, K, 1.0, X.data(), K, t.W.data(),
                N, 0.0, Y.data(), N);
    loss = 0.0;
    for (size_t i = 0; i < Y.size(); ++i) {
      double d = Y[i] - T[i];
      loss += d * d;
      dY[i] = d / B;
    }
    loss *= 0.5 / B;
    if (e == 0) t.acc = loss;
    // backward 1: dW = X^T dY                              (KxB)(BxN)
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, N, B, 1.0, X.data(), K, dY.data(), N,
                0.0, dW.data(), N);
    // backward 2: dX = dY W^T                              (BxN)(NxK)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, B, K, N, 1.0, dY.data(), N, t.W.data(),
                N, 0.0, dX.data(), K);
    // update:     W = W - lr dW
    cblas_daxpy(K * N, -lr, dW.data(), 1, t.W.data(), 1);
    g_sink += dX[0];
    t.progress = e + 1;
    t.flops_run += ef;
    st.exec_train += ef;
    st.epochs_run += 1;
    st.t_train += MPI_Wtime() - t0;
  }
  if (t.acc > 0) st.loss_ratio_sum += loss / t.acc;
  return COMPLETED;
}

// Inference: each sample is one vector pushed through L layers (DGEMV+bias+ReLU).
static Status run_inference(Task& t, int stop_at = -1) {
  const InferShape& s = g_ishapes[t.shape];
  std::vector<std::vector<double>> W(s.L), b(s.L);
  for (int l = 0; l < s.L; ++l) {
    W[l].resize((size_t)s.d[l + 1] * s.d[l]);
    b[l].resize(s.d[l + 1]);
    fill_uniform(W[l].data(), W[l].size(), mix3(cfg.seed, t.id, 100 + l),
                 std::sqrt(3.0 / s.d[l]));
    fill_uniform(b[l].data(), b[l].size(), mix3(cfg.seed, t.id, 200 + l), 0.1);
  }
  std::vector<double> in((size_t)s.batch * s.d[0]);
  fill_uniform(in.data(), in.size(), mix3(cfg.seed, t.id, 99), 1.0);
  std::vector<double> a(cfg.max_dim), y(cfg.max_dim);
  const double sf = infer_sample_flops(s);

  for (int i = t.progress; i < s.batch; ++i) {
    if (i == stop_at) return PREEMPTED;
    Status cs = check_interrupt(INFER);
    if (cs != CONTINUE) return cs;
    double t0 = MPI_Wtime();
    std::copy(in.begin() + (size_t)i * s.d[0], in.begin() + (size_t)(i + 1) * s.d[0], a.begin());
    for (int l = 0; l < s.L; ++l) {
      const int m = s.d[l + 1], n = s.d[l];
      std::copy(b[l].begin(), b[l].end(), y.begin());  // y = b
      cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, W[l].data(), n, a.data(), 1, 1.0,
                  y.data(), 1);                       // y = W a + y
      for (int j = 0; j < m; ++j) a[j] = y[j] > 0.0 ? y[j] : 0.0;  // ReLU
    }
    double sum = 0.0;
    for (int j = 0; j < s.d[s.L]; ++j) sum += a[j];
    t.acc += sum;
    t.progress = i + 1;
    t.flops_run += sf;
    st.exec_infer += sf;
    st.samples_run += 1;
    st.t_infer += MPI_Wtime() - t0;
  }
  return COMPLETED;
}

// ----------------------------------------------------------------------------
// GPU capacity model
//
//   capacity per device  C = gpu_mult x (CPU FLOP/s per process)   [or --gpu-gflops]
//   share per rank         = C / (#ranks sharing that device), same for streams
//
// Every task shape gets a modelled DEMAND: the FLOP/s it can usefully consume
// on the device, from its shape and operations:
//   training  (DGEMM-bound)  occupancy = output tiles (gpu_tile^2) of its GEMMs
//                            / (SMs x ctas_per_sm), demand = C x min(1, occ)
//   inference (GEMV-bound)   memory-bound: each weight (8 B) feeds 2 FLOPs, so
//                            demand <= DRAM_BW/4; occupancy = rows in flight /
//                            (SMs x 2048 threads)
//   both                     demand <= FLOPs per unit / (launches x launch_us)
// The scheduler packs jobs while  sum(demand) <= share, #jobs <= streams share,
// sum(GEMV demand) <= bandwidth share, and the memory estimate fits.
// ----------------------------------------------------------------------------
struct Capacity {
  double cpu = 0;        // CPU FLOP/s per process
  double gpu_dev = 0;    // GPU FLOP/s per device
  double gpu_share = 0;  // this rank's share
  double bw_dev = 0;     // GEMV FLOP/s bound per device (DRAM_BW / 4)
  double bw_share = 0;
  double mem_share = 0;  // device bytes this rank may plan to use
  int sm = 1, sharers = 1, slot_limit = 1;
};
static Capacity g_cap;

struct GpuCost {
  double demand = 0;      // FLOP/s
  double unit_flops = 0;  // per epoch / per chunk
  int chunk = 1;          // inference samples per unit
  double mem = 0;         // device bytes
  bool gemv = false;
};
static std::vector<GpuCost> g_cost[2];

static double ceil_div(double a, double b) { return std::ceil(a / b); }

static GpuCost train_cost(const TrainShape& s) {
  GpuCost c;
  const double T = cfg.gpu_tile;
  double tiles = (ceil_div(s.B, T) * ceil_div(s.N, T) + ceil_div(s.K, T) * ceil_div(s.N, T) +
                  ceil_div(s.B, T) * ceil_div(s.K, T)) / 3.0;
  double occ = std::min(1.0, tiles / (double)(g_cap.sm * cfg.ctas_per_sm));
  c.unit_flops = train_epoch_flops(s);
  const double launches = 9.0;  // 3 GEMM + AXPY + dY/loss + 3 reduce + copy
  double launch_bound = c.unit_flops / (launches * cfg.launch_us * 1e-6);
  c.demand = std::min(g_cap.gpu_dev * occ, launch_bound);
  c.mem = 8.0 * (2.0 * s.B * s.K + 3.0 * s.B * s.N + 2.0 * s.K * s.N + 70000);
  return c;
}

static GpuCost infer_cost(const InferShape& s) {
  GpuCost c;
  c.gemv = true;
  const double sf = infer_sample_flops(s);
  double rows = 0, nw = 0, nb = 0;
  int maxd = 0;
  for (int l = 0; l < s.L; ++l) {
    rows += s.d[l + 1];
    nw += (double)s.d[l + 1] * s.d[l];
    nb += s.d[l + 1];
  }
  for (int l = 0; l <= s.L; ++l) maxd = std::max(maxd, s.d[l]);
  rows /= s.L;
  const double launches_per_layer = 3.0, launches_fixed = 4.0;
  auto demand_for = [&](int chunk) {
    double occ = std::min(1.0, rows * chunk / (double)(g_cap.sm * 2048));
    double peak = std::min(g_cap.gpu_dev, g_cap.bw_dev > 0 ? g_cap.bw_dev : g_cap.gpu_dev);
    double launch_bound = chunk * sf / ((launches_per_layer * s.L + launches_fixed) *
                                        cfg.launch_us * 1e-6);
    return std::min(peak * occ, launch_bound);
  };
  // chunk size: as many samples as fit ~unit_ms at the resulting demand
  int chunk = std::min(gpu::max_chunk(), s.batch);
  for (int it = 0; it < 3; ++it) {
    double d = demand_for(chunk);
    int want = (int)std::ceil(d * cfg.unit_ms * 1e-3 / std::max(sf, 1.0));
    chunk = std::max(1, std::min({want, gpu::max_chunk(), s.batch}));
  }
  c.chunk = chunk;
  c.demand = demand_for(chunk);
  c.unit_flops = chunk * sf;
  c.mem = 8.0 * (nw + nb + (double)s.batch * s.d[0] + 2.0 * gpu::max_chunk() * maxd + 70000);
  return c;
}

static void precompute_gpu_costs() {
  g_cost[TRAIN].resize(cfg.nshapes);
  g_cost[INFER].resize(cfg.nshapes);
  for (int i = 0; i < cfg.nshapes; ++i) {
    g_cost[TRAIN][i] = train_cost(g_tshapes[i]);
    g_cost[INFER][i] = infer_cost(g_ishapes[i]);
  }
}

// Calibrate the CPU FLOP rate of this process: single-threaded DGEMM.
static double calibrate_cpu_flops() {
  const int n = 384;
  std::vector<double> A((size_t)n * n), B((size_t)n * n), C((size_t)n * n);
  fill_uniform(A.data(), A.size(), 1, 1.0);
  fill_uniform(B.data(), B.size(), 2, 1.0);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A.data(), n, B.data(), n,
              0.0, C.data(), n);  // warm-up
  int reps = 0;
  double t0 = MPI_Wtime(), t = 0;
  do {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A.data(), n, B.data(),
                n, 0.0, C.data(), n);
    ++reps;
    t = MPI_Wtime() - t0;
  } while (t < 0.25);
  g_sink += C[0];
  return 2.0 * n * n * n * reps / t;
}

// ----------------------------------------------------------------------------
// GPU jobs. A job is one task resident on one stream slot. It is advanced by
// job_service() without blocking: retire finished units (events), then
// enqueue up to gpu_depth units ahead. Interrupts take effect at unit
// boundaries: pending units finish, then the task state (W / checksum) is
// pulled back for the checkpoint — exactly the state the CPU kernels keep.
// ----------------------------------------------------------------------------
[[noreturn]] static void gpu_fatal(const char* where) {
  std::fprintf(stderr, "r%d: fatal GPU error in %s: %s\n", g_rank, where, gpu::last_error());
  std::fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, 3);
  std::abort();
}

struct GpuJob {
  Task task;
  int slot = -1;
  GpuCost cost;
  bool from_drain = false, spilled = false;
  int total = 0;     // epochs or batch size
  int next = 0;      // first epoch / sample not yet enqueued
  struct Unit { int ring, start, count; } u[gpu::RING];
  int head = 0, n = 0, ring_next = 0;
  int stop_req = CONTINUE;  // pending interrupt
  double last_loss = 0;
};

// Upload / generate the task's data on the job's slot. OK, OOM or ERR.
static int job_begin(GpuJob& j) {
  Task& t = j.task;
  j.cost = g_cost[t.type][t.shape];
  j.next = t.progress;
  j.head = j.n = j.ring_next = 0;
  j.stop_req = CONTINUE;
  if (t.type == TRAIN) {
    const TrainShape& s = g_tshapes[t.shape];
    j.total = cfg.epochs;
    return gpu::train_begin(j.slot, s.B, s.K, s.N, mix3(cfg.seed, t.id, 1),
                            mix3(cfg.seed, t.id, 2), mix3(cfg.seed, t.id, 3),
                            1.0 / std::sqrt((double)s.K), t.W.empty() ? nullptr : t.W.data());
  }
  const InferShape& s = g_ishapes[t.shape];
  j.total = s.batch;
  uint64_t sw[MAXL], sb[MAXL];
  double scw[MAXL];
  for (int l = 0; l < s.L; ++l) {
    sw[l] = mix3(cfg.seed, t.id, 100 + l);
    sb[l] = mix3(cfg.seed, t.id, 200 + l);
    scw[l] = std::sqrt(3.0 / s.d[l]);
  }
  return gpu::infer_begin(j.slot, s.batch, s.L, s.d, sw, scw, sb, 0.1, mix3(cfg.seed, t.id, 99),
                          t.acc);
}

static void job_commit(GpuJob& j, const GpuJob::Unit& u) {
  Task& t = j.task;
  double f;
  if (t.type == TRAIN) {
    double loss = gpu::ring_value(j.slot, u.ring);
    if (u.start == 0) t.acc = loss;
    j.last_loss = loss;
    t.progress = u.start + 1;
    f = j.cost.unit_flops;
    st.exec_train += f;
    st.epochs_run += 1;
  } else {
    t.progress = u.start + u.count;
    f = u.count * infer_sample_flops(g_ishapes[t.shape]);
    st.exec_infer += f;
    st.samples_run += u.count;
  }
  t.flops_run += f;
  st.exec_gpu += f;
}

// Advance a job. Returns CONTINUE while it is running; otherwise its final
// status (task state already pulled back to the host). `progressed` is set
// if any unit retired or was enqueued. stop_at >= 0: stop there (self test).
static Status job_service(GpuJob& j, bool* progressed, int stop_at = -1) {
  Task& t = j.task;
  while (j.n > 0) {  // retire completed units (in order)
    const GpuJob::Unit& u = j.u[j.head];
    int q = gpu::query(j.slot, u.ring);
    if (q < 0) gpu_fatal("event query");
    if (q == 0) break;
    job_commit(j, u);
    j.head = (j.head + 1) % gpu::RING;
    j.n--;
    *progressed = true;
  }
  if (j.stop_req == EXIT_NOSAVE) return EXIT_NOSAVE;  // abandon, in-flight work dropped
  const int end = stop_at >= 0 ? std::min(j.total, stop_at) : j.total;
  if (j.stop_req != CONTINUE || j.next >= end) {
    if (j.n > 0) return CONTINUE;  // let the in-flight units finish
    if (t.type == TRAIN) {
      const TrainShape& s = g_tshapes[t.shape];
      t.W.resize((size_t)s.K * s.N);
      if (gpu::train_download_W(j.slot, t.W.data()) != gpu::OK) gpu_fatal("download W");
    } else if (gpu::infer_get_acc(j.slot, &t.acc) != gpu::OK) {
      gpu_fatal("download checksum");
    }
    *progressed = true;
    // completion wins over an interrupt that arrived while the last units ran
    if (t.progress >= j.total) {
      if (t.type == TRAIN && t.acc > 0) st.loss_ratio_sum += j.last_loss / t.acc;
      return COMPLETED;
    }
    if (j.stop_req != CONTINUE) return (Status)j.stop_req;
    return PREEMPTED;  // stop_at reached
  }
  const int depth = std::max(1, std::min(cfg.gpu_depth, gpu::RING));
  while (j.n < depth && j.next < end) {  // keep the stream fed
    int ring = j.ring_next;
    j.ring_next = (j.ring_next + 1) % gpu::RING;
    int count = t.type == TRAIN ? 1 : std::min(j.cost.chunk, end - j.next);
    int rc = t.type == TRAIN ? gpu::train_enqueue_epoch(j.slot, ring)
                             : gpu::infer_enqueue_chunk(j.slot, j.next, count, ring);
    if (rc != gpu::OK) gpu_fatal("enqueue");
    j.u[(j.head + j.n) % gpu::RING] = {ring, j.next, count};
    j.n++;
    j.next += count;
    *progressed = true;
  }
  return CONTINUE;
}

// Run one task synchronously on GPU slot 0 (self test).
static Status run_gpu_sync(Task& t, int stop_at, bool* ok) {
  GpuJob j;
  j.task = std::move(t);
  j.slot = 0;
  *ok = job_begin(j) == gpu::OK;
  if (!*ok) {
    t = std::move(j.task);
    return GPU_UNAVAILABLE;
  }
  Status s;
  bool prog = false;
  while ((s = job_service(j, &prog, stop_at)) == CONTINUE) std::this_thread::yield();
  t = std::move(j.task);
  return s;
}

// Dispatch one task synchronously to the CPU or GPU implementation (self
// test). *on_gpu reports where it ran.
static Status run_task(Task& t, bool want_gpu, bool* on_gpu, int stop_at = -1) {
  if (want_gpu) {
    bool ok = false;
    Status s = run_gpu_sync(t, stop_at, &ok);
    if (ok) {
      *on_gpu = true;
      return s;
    }
    st.gpu_fallbacks++;
    LOG("GPU unavailable for %s task %lld (%s) - running it on the CPU", tname(t.type),
        (long long)t.id, gpu::last_error());
  }
  *on_gpu = false;
  return t.type == TRAIN ? run_training(t, stop_at) : run_inference(t, stop_at);
}

// ----------------------------------------------------------------------------
// Checkpoint files: one per rank, mirrors the rank's pending checkpoint queue
// ----------------------------------------------------------------------------
static const char CK_MAGIC[8] = {'S', 'I', 'M', 'C', 'K', 'P', 'T', '1'};
static std::deque<Task> pending;

static std::string ckpt_path(int rank) {
  return cfg.ckpt_dir + "/rank_" + std::to_string(rank) + ".ckpt";
}

static bool write_tasks(const std::string& path, const std::deque<Task>& q, double* bytes) {
  std::string tmp = path + ".tmp";
  FILE* f = std::fopen(tmp.c_str(), "wb");
  if (!f) {
    std::fprintf(stderr, "r%d: cannot write %s: %s\n", g_rank, tmp.c_str(), std::strerror(errno));
    return false;
  }
  size_t nb = 0;
  auto W = [&](const void* p, size_t n) { nb += std::fwrite(p, 1, n, f); };
  int32_t cnt = (int32_t)q.size();
  uint64_t seed = cfg.seed;
  int32_t ep = cfg.epochs;
  W(CK_MAGIC, 8);
  W(&seed, 8);
  W(&ep, 4);
  W(&cnt, 4);
  for (const Task& t : q) {
    int32_t hdr[4] = {t.type, t.origin, t.shape, t.progress};
    W(hdr, sizeof hdr);
    W(&t.id, 8);
    W(&t.acc, 8);
    int64_t nw = (int64_t)t.W.size();
    W(&nw, 8);
    if (nw) W(t.W.data(), nw * sizeof(double));
  }
  std::fclose(f);
  if (std::rename(tmp.c_str(), path.c_str()) != 0) return false;
  if (bytes) *bytes += nb;
  return true;
}

static bool read_tasks(const std::string& path, std::deque<Task>& q) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  bool ok = true;
  auto R = [&](void* p, size_t n) { ok = ok && std::fread(p, 1, n, f) == n; };
  char mg[8];
  uint64_t seed = 0;
  int32_t ep = 0, cnt = 0;
  R(mg, 8);
  R(&seed, 8);
  R(&ep, 4);
  R(&cnt, 4);
  if (!ok || std::memcmp(mg, CK_MAGIC, 8) != 0) {
    std::fclose(f);
    std::fprintf(stderr, "r%d: %s is not a checkpoint file, ignored\n", g_rank, path.c_str());
    return false;
  }
  if (seed != cfg.seed || ep != cfg.epochs)
    std::fprintf(stderr, "r%d: warning: %s was written with seed=%llu epochs=%d\n", g_rank,
                 path.c_str(), (unsigned long long)seed, ep);
  for (int i = 0; i < cnt && ok; ++i) {
    Task t;
    int32_t hdr[4];
    int64_t nw = 0;
    R(hdr, sizeof hdr);
    R(&t.id, 8);
    R(&t.acc, 8);
    R(&nw, 8);
    if (!ok || nw < 0 || nw > (int64_t)DIM_CAP * DIM_CAP) { ok = false; break; }
    t.type = hdr[0]; t.origin = hdr[1]; t.shape = hdr[2]; t.progress = hdr[3];
    t.W.resize(nw);
    if (nw) R(t.W.data(), nw * sizeof(double));
    if (!ok) break;
    // the record must fit the current shape table (seed / dims / nshapes may differ)
    bool fits = (t.type == TRAIN || t.type == INFER) && t.shape >= 0 && t.shape < cfg.nshapes;
    if (fits && t.type == TRAIN) {
      const TrainShape& s = g_tshapes[t.shape];
      fits = (nw == 0 || nw == (int64_t)s.K * s.N) && t.progress >= 0 && t.progress < cfg.epochs;
    } else if (fits) {
      fits = nw == 0 && t.progress >= 0 && t.progress < g_ishapes[t.shape].batch;
    }
    if (!fits) {
      std::fprintf(stderr, "r%d: %s: %s task %lld does not match the current shapes, dropped\n",
                   g_rank, path.c_str(), tname(t.type), (long long)t.id);
      continue;
    }
    q.push_back(std::move(t));
  }
  std::fclose(f);
  return ok;
}

// On resume: replace the queue with what the checkpoint file holds, keeping
// the in-memory per-run FLOP counters (not part of the file format) so work
// executed before the suspension is still credited when the task completes.
static void reload_pending() {
  std::deque<Task> fromfile;
  if (!read_tasks(ckpt_path(g_rank), fromfile)) return;
  for (Task& t : fromfile)
    for (const Task& m : pending)
      if (m.id == t.id && m.type == t.type) {
        t.flops_run = m.flops_run;
        break;
      }
  pending.swap(fromfile);
}

static void save_pending() {
  double t0 = MPI_Wtime();
  if (write_tasks(ckpt_path(g_rank), pending, &st.ckpt_bytes)) st.ckpt_writes++;
  st.t_ckpt += MPI_Wtime() - t0;
}

// ----------------------------------------------------------------------------
// Scheduling policy
// ----------------------------------------------------------------------------
static std::mt19937_64 g_rng;
static int force_infer = 0;  // next N decisions are forced to inference
static int since_sync = 0;

static void sync_global_mix() {
  g_train_seen = rma_read(W_GTRAIN);
  g_inf_seen = rma_read(W_GINF);
  st.syncs++;
}

// Biased coin: base probability p, corrected towards the target mix using the
// global (all ranks, synced every cfg.sync_every decisions) and local counters,
// so both the machine-wide and per-rank train/infer mix track p.
static int choose_type() {
  if (force_infer > 0) {
    --force_infer;
    return INFER;
  }
  const double p = cur_p;
  if (p <= 0.0) return TRAIN;
  if (p >= 1.0) return INFER;
  double gt = (double)(g_inf_seen + g_train_seen);
  double fg = gt > 0 ? g_inf_seen / gt : p;
  double lt = st.infer_new + st.train_new;
  double fl = lt > 0 ? st.infer_new / lt : p;
  double pe = std::min(1.0, std::max(0.0, p + 0.5 * (p - fg) + 0.5 * (p - fl)));
  return std::uniform_real_distribution<double>(0.0, 1.0)(g_rng) < pe ? INFER : TRAIN;
}

static int pick_shape(int type, int64_t id) {
  return (int)(mix3(cfg.seed, (uint64_t)id, 7 + type) % (uint64_t)cfg.nshapes);
}

// ----------------------------------------------------------------------------
// Self test: interrupted+checkpointed+resumed == uninterrupted (bitwise)
// ----------------------------------------------------------------------------
// Largest relative difference between the results of two runs of the same task
static double rel_diff(const Task& a, const Task& b) {
  double d = std::fabs(a.acc - b.acc) / std::max(std::fabs(a.acc), 1e-300);
  double wmax = 0, dmax = 0;
  if (a.W.size() != b.W.size()) return INFINITY;
  for (size_t i = 0; i < a.W.size(); ++i) {
    wmax = std::max(wmax, std::fabs(a.W[i]));
    dmax = std::max(dmax, std::fabs(a.W[i] - b.W[i]));
  }
  if (!a.W.empty()) d = std::max(d, dmax / std::max(wmax, 1e-300));
  return d;
}

// The cheapest task ids of a type (keeps the self test fast with large dims)
static std::vector<int64_t> cheapest_ids(int type, int count) {
  std::vector<std::pair<double, int64_t>> c;
  std::vector<bool> seen(cfg.nshapes, false);
  for (int64_t id = 0; id < 20000; ++id) {
    int sh = pick_shape(type, id);
    if (seen[sh]) continue;
    seen[sh] = true;
    c.push_back({g_task_flops[type][sh], id});
  }
  std::sort(c.begin(), c.end());
  std::vector<int64_t> ids;
  for (int i = 0; i < count && i < (int)c.size(); ++i) ids.push_back(c[i].second);
  return ids;
}

// Run `a` to completion on dev_a; run `b` (same task) to the midpoint on dev_b,
// checkpoint to disk, reload, finish on dev_c. Returns the relative difference.
static double split_run(int type, int64_t id, bool dev_a, bool dev_b, bool dev_c, bool* ok) {
  std::string path = cfg.ckpt_dir + "/selftest.ckpt";
  Task a;
  a.type = type; a.id = id; a.shape = pick_shape(type, id);
  Task b = a;
  bool g;
  Status sa = run_task(a, dev_a, &g);
  bool ga = g;
  int half = type == TRAIN ? cfg.epochs / 2 : g_ishapes[b.shape].batch / 2;
  Status sb = run_task(b, dev_b, &g, half);
  bool gb = g;
  std::deque<Task> q{b}, q2;
  write_tasks(path, q, nullptr);
  read_tasks(path, q2);
  std::remove(path.c_str());
  if (q2.empty()) { *ok = false; return INFINITY; }
  Task c = q2.front();
  Status sc = run_task(c, dev_c, &g);
  *ok = sa == COMPLETED && sb == PREEMPTED && sc == COMPLETED && c.progress == a.progress &&
        ga == dev_a && gb == dev_b && g == dev_c;
  return rel_diff(a, c);
}

static bool selftest() {
  Stats saved = st;
  selftest_mode = true;
  bool all_ok = true;
  auto dn = [](bool g) { return g ? "gpu" : "cpu"; };
  struct Case { bool a, b, c; double tol; const char* what; };
  std::vector<Case> cases = {{false, false, false, 0.0, "cpu run vs cpu ckpt+resume"}};
  if (g_gpu_ready) {
    cases.push_back({true, true, true, 1e-12, "gpu run vs gpu ckpt+resume"});
    cases.push_back({false, true, true, 1e-9, "cpu run vs gpu run (+resume)"});
    cases.push_back({false, true, false, 1e-9, "cpu run vs gpu ckpt -> cpu resume"});
  }
  for (const Case& cs : cases) {
    for (int type : {TRAIN, INFER}) {
      for (int64_t id : cheapest_ids(type, 2)) {
        bool ok = false;
        double d = split_run(type, id, cs.a, cs.b, cs.c, &ok);
        bool pass = ok && d <= cs.tol;
        char res[64];
        if (cs.tol == 0.0)
          std::snprintf(res, sizeof res, "%s", pass ? "PASS (bitwise identical)" : "FAIL");
        else
          std::snprintf(res, sizeof res, "%s (rel diff %.2e)", pass ? "PASS" : "FAIL", d);
        std::printf("  %-34s %-5s task %-5lld shape %-3d [%s|%s->%s] : %s\n", cs.what,
                    tname(type), (long long)id, pick_shape(type, id), dn(cs.a), dn(cs.b),
                    dn(cs.c), res);
        all_ok = all_ok && pass;
      }
    }
  }
  // Several tasks resident on separate streams at once must each give the
  // same result as running alone on the CPU (no cross-stream interference).
  if (g_gpu_ready) {
    std::vector<Task> ref;
    for (int type : {TRAIN, INFER})
      for (int64_t id : cheapest_ids(type, 3)) {
        Task t;
        t.type = type; t.id = id; t.shape = pick_shape(type, id);
        ref.push_back(t);
      }
    int nj = std::min((int)ref.size(), cfg.gpu_streams);
    std::vector<GpuJob> jobs(nj);
    bool began = true;
    for (int i = 0; i < nj; ++i) {
      jobs[i].task = ref[i];
      jobs[i].slot = i;
      began = began && job_begin(jobs[i]) == gpu::OK;
    }
    for (int i = 0; i < nj; ++i) {  // CPU references
      bool g;
      run_task(ref[i], false, &g);
    }
    std::vector<Status> res(nj, CONTINUE);
    for (int left = nj; began && left > 0;) {
      for (int i = 0; i < nj; ++i) {
        if (res[i] != CONTINUE) continue;
        bool prog = false;
        res[i] = job_service(jobs[i], &prog);
        if (res[i] != CONTINUE) --left;
      }
      std::this_thread::yield();
    }
    double worst = 0;
    bool ok = began;
    for (int i = 0; i < nj && ok; ++i) {
      ok = res[i] == COMPLETED;
      worst = std::max(worst, rel_diff(ref[i], jobs[i].task));
    }
    bool pass = ok && worst <= 1e-9;
    std::printf("  %-34s %d tasks on %d streams at once vs cpu   : %s (max rel diff %.2e)\n",
                "concurrent gpu jobs", nj, nj, pass ? "PASS" : "FAIL", worst);
    all_ok = all_ok && pass;
  }
  selftest_mode = false;
  st = saved;
  return all_ok;
}

// ----------------------------------------------------------------------------
// Main loop
// ----------------------------------------------------------------------------
static void run_loop() {
  bool drain = !pending.empty();  // leftovers from a previous run go first
  bool was_susp = false;
  double t_loop0 = MPI_Wtime();

  while (true) {
    st.loops++;
    double now = MPI_Wtime();
    sample_timeline(now);
    poll_control();
    prio_event = false;  // no training task in flight right now

    if (cur_exit >= 0) {
      if (cur_exit == 1) {
        save_pending();
        LOG("exit e:1 while idle/between tasks, %zu checkpoint(s) in %s", pending.size(),
            ckpt_path(g_rank).c_str());
      }
      break;
    }

    if (susp_me) {
      if (!was_susp) {
        was_susp = true;
        st.suspends++;
        save_pending();
        LOG("SUSPENDED: %zu checkpoint(s) saved to %s", pending.size(),
            ckpt_path(g_rank).c_str());
      }
      st.idle_loops++;
      double t0 = MPI_Wtime();
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
      st.t_idle += MPI_Wtime() - t0;
      continue;
    }
    if (was_susp) {
      was_susp = false;
      st.resumes++;
      reload_pending();  // run exactly what was saved
      drain = !pending.empty();
      LOG("RESUMED: %zu checkpoint(s) to drain before new tasks", pending.size());
    }

    Task task;
    bool resumed = false;
    double ts = MPI_Wtime();
    if (drain && !pending.empty()) {
      task = std::move(pending.front());
      pending.pop_front();
      resumed = true;
    } else {
      drain = false;
      int type = choose_type();
      auto it = std::find_if(pending.begin(), pending.end(),
                             [&](const Task& t) { return t.type == type; });
      if (it != pending.end()) {
        task = std::move(*it);
        pending.erase(it);
        resumed = true;
      } else {
        task.type = type;
        task.id = nxtval();
        task.shape = pick_shape(type, task.id);
        const int64_t one = 1;
        MPI_Accumulate(&one, 1, MPI_INT64_T, 0,
                       type == TRAIN ? W_GTRAIN : W_GINF, 1, MPI_INT64_T, MPI_SUM, g_win);
        MPI_Win_flush(0, g_win);
        (type == TRAIN ? st.train_new : st.infer_new) += 1;
        if (++since_sync >= cfg.sync_every) {
          since_sync = 0;
          sync_global_mix();
        }
      }
    }
    st.t_sched += MPI_Wtime() - ts;
    if (resumed) {
      (task.type == TRAIN ? st.train_resumed : st.infer_resumed) += 1;
      save_pending();  // file always mirrors the queue
      if (cfg.verbose > 1)
        LOG("resume %s task %lld (%s) at %d", tname(task.type), (long long)task.id,
            oname(task.origin), task.progress);
    }

    st.task_runs++;
    bool on_gpu = false;
    Status s = run_task(task, route_to_gpu(task.type, task.shape), &on_gpu);

    switch (s) {
      case COMPLETED:
        if (task.type == TRAIN) {
          st.train_done++;
          st.flops_train += task.flops_run;
          if (on_gpu) st.train_done_gpu++, st.flops_train_gpu += task.flops_run;
        } else {
          st.infer_done++;
          st.flops_infer += task.flops_run;
          st.checksum += task.acc;
          if (on_gpu) st.infer_done_gpu++, st.flops_infer_gpu += task.flops_run;
        }
        break;
      case PREEMPTED:
        task.origin = ORIG_PREEMPT;
        st.preempts++;
        LOG("priority -> %.3f: training task %lld (%s) checkpointed at epoch %d/%d, switching "
            "to inference", cur_p, (long long)task.id, on_gpu ? "gpu" : "cpu", task.progress,
            cfg.epochs);
        pending.push_back(std::move(task));
        save_pending();
        force_infer = 1;
        break;
      case SUSPENDED:
        task.origin = ORIG_SUSPEND;
        LOG("suspend: %s task %lld (%s) checkpointed at %d", tname(task.type),
            (long long)task.id, on_gpu ? "gpu" : "cpu", task.progress);
        pending.push_front(std::move(task));  // in-flight work drains first on resume
        save_pending();
        drain = true;
        break;
      case EXIT_SAVE:
        task.origin = ORIG_EXIT;
        LOG("exit e:1: %s task %lld (%s) checkpointed at %d", tname(task.type),
            (long long)task.id, on_gpu ? "gpu" : "cpu", task.progress);
        pending.push_front(std::move(task));
        save_pending();
        break;
      case EXIT_NOSAVE:
        st.flops_discarded += task.flops_run;
        LOG("exit e:0: %s task %lld (%s) abandoned at %d (no checkpoint)", tname(task.type),
            (long long)task.id, on_gpu ? "gpu" : "cpu", task.progress);
        break;
      default:
        break;
    }
    if (s == EXIT_SAVE || s == EXIT_NOSAVE) break;
  }
  st.wall = MPI_Wtime() - t_loop0;
  st.t_busy = st.t_train + st.t_infer;
  for (const Task& t : pending) st.flops_pending += t.flops_run;
  st.pending_left = (double)pending.size();
}

// ----------------------------------------------------------------------------
// Concurrent engine (exec_mode gpu / cpu,gpu)
//
// The main thread is an orchestrator: it polls control, retires/enqueues GPU
// units on up to slot_limit streams, and starts tasks while the GPU has spare
// modelled capacity (sum of demands <= capacity share) and, in cpu,gpu mode,
// while the CPU worker is idle. Control semantics are those of run_loop():
//   * priority rises -> every running training task (all GPU jobs and the CPU
//     worker) checkpoints, and the next N decisions are inference
//   * s:R -> all running tasks checkpoint, rank idles; r:R -> drains the
//     checkpoints (no new task until they have all completed)
//   * e:1 -> all running tasks checkpoint, exit; e:0 -> abandon, exit
// ----------------------------------------------------------------------------
static void cpu_worker_main() {
  tl_in_worker = true;
  std::unique_lock<std::mutex> lk(g_cpu.m);
  for (;;) {
    g_cpu.cv.wait(lk, [] { return g_cpu.has || g_cpu.quit; });
    if (g_cpu.quit) break;
    lk.unlock();
    Task& t = g_cpu.task;  // owned by the worker while `has` is set
    Status s = t.type == TRAIN ? run_training(t) : run_inference(t);
    worker_publish();
    lk.lock();
    g_cpu.result = s;
    g_cpu.has = false;
    g_cpu.done.store(true, std::memory_order_release);
  }
  g_cpu.final_stats = st;
}

struct Staged {  // a task taken (NXTVAL or checkpoint queue) but not yet placed
  Task task;
  bool from_drain = false;
  double t_staged = 0;
};

// FLOPs a task still has to execute
static double remaining_flops(const Task& t) {
  if (t.type == TRAIN)
    return (cfg.epochs - t.progress) * train_epoch_flops(g_tshapes[t.shape]);
  const InferShape& s = g_ishapes[t.shape];
  return (s.batch - t.progress) * infer_sample_flops(s);
}

static void run_loop_concurrent() {
  const bool cpu_slot = (cfg.exec_mode & EXEC_CPU) != 0;  // CPU takes regular tasks
  std::vector<GpuJob> jobs;
  std::vector<int> free_slots;
  for (int i = g_cap.slot_limit - 1; i >= 0; --i) free_slots.push_back(i);
  double sum_dem = 0, sum_bw = 0, sum_mem = 0;
  bool gpu_mem_full = false;  // an allocation failed; wait until a job finishes
  std::deque<Staged> staged;
  int cpu_type = TRAIN;
  bool cpu_from_drain = false;
  bool drain = !pending.empty();
  bool was_susp = false, dirty = false;
  int fallback_logs = 0;

  g_cpu.th = std::thread(cpu_worker_main);
  const double t_loop0 = MPI_Wtime();
  double t_last = t_loop0;

  auto note_resumed = [&](const Task& t) {
    (t.type == TRAIN ? st.train_resumed : st.infer_resumed) += 1;
    dirty = true;  // the checkpoint file mirrors the queue
    if (cfg.verbose > 1)
      LOG("resume %s task %lld (%s) at %d", tname(t.type), (long long)t.id, oname(t.origin),
          t.progress);
  };
  auto cpu_idle = [&] { return !g_cpu.running && (cfg.pack || jobs.empty()); };
  auto gpu_fits = [&](const GpuCost& c) {
    if (free_slots.empty()) return false;
    if (!cfg.pack) return jobs.empty() && !g_cpu.running;
    if (jobs.empty()) return true;  // a task bigger than the share still has to run
    if (gpu_mem_full) return false;
    const double eps = 1e-9;
    if (sum_dem + c.demand > g_cap.gpu_share * (1 + eps)) return false;
    if (c.gemv && g_cap.bw_share > 0 && sum_bw + c.demand > g_cap.bw_share * (1 + eps))
      return false;
    return sum_mem + c.mem <= g_cap.mem_share;
  };
  // smallest demand of any task that may run on the GPU: "room" means at
  // least that much capacity is still free
  double min_dem = INFINITY;
  for (int ty = 0; ty < 2; ++ty)
    for (int i = 0; i < cfg.nshapes; ++i)
      if (route_to_gpu(ty, i) || (cfg.spill && cpu_slot))
        min_dem = std::min(min_dem, g_cost[ty][i].demand);
  auto gpu_room = [&] {
    if (free_slots.empty()) return false;
    if (!cfg.pack) return jobs.empty() && !g_cpu.running;
    return jobs.empty() ||
           (!gpu_mem_full && sum_dem + min_dem <= g_cap.gpu_share * (1 + 1e-9));
  };
  auto start_cpu = [&](Staged& x) {
    cpu_type = x.task.type;
    cpu_from_drain = x.from_drain;
    {
      std::lock_guard<std::mutex> lk(g_cpu.m);
      g_cpu.task = std::move(x.task);
      g_cpu.intr.store(CONTINUE);
      g_cpu.done.store(false);
      g_cpu.has = true;
    }
    g_cpu.cv.notify_one();
    g_cpu.running = true;
    st.task_runs++;
    st.loops++;
  };
  // returns false (task left in x) if the device memory is exhausted
  auto start_gpu = [&](Staged& x, bool spilled) {
    GpuJob j;
    j.task = std::move(x.task);
    j.slot = free_slots.back();
    j.from_drain = x.from_drain;
    j.spilled = spilled;
    int rc = job_begin(j);
    if (rc == gpu::OOM) {  // free the pools of idle slots and retry once
      for (int s : free_slots)
        if (s != j.slot) gpu::release_slot(s);
      rc = job_begin(j);
    }
    if (rc == gpu::ERR) gpu_fatal("job begin");
    if (rc == gpu::OOM) {
      x.task = std::move(j.task);
      gpu_mem_full = true;
      return false;
    }
    free_slots.pop_back();
    sum_dem += j.cost.demand;
    if (j.cost.gemv) sum_bw += j.cost.demand;
    sum_mem += j.cost.mem;
    st.gpu_jobs++;
    st.task_runs++;
    st.loops++;
    if (spilled) st.spilled++;
    jobs.push_back(std::move(j));
    st.max_conc = std::max(st.max_conc, (double)jobs.size());
    return true;
  };
  // GPU admission policy for the staged tasks of one fill pass:
  //   * GPU-routed tasks have priority on the GPU: no spill while one waits
  //   * backfilling (a later, smaller task passing a blocked one) is allowed
  //     until the oldest blocked GPU task has waited max_wait_ms; then only it
  //     may start on the GPU (no starvation of large tasks)
  bool gpu_waiting = false;  // some GPU-routed task is staged
  const Staged* aged = nullptr;  // oldest over-age GPU-routed staged task
  // Earliest-finish-time check for a GPU-routed task blocked behind GPU work:
  // would it finish sooner on the (idle) CPU? Rates are the measured ones once
  // there is enough history, the modelled ones before.
  double gpu_ahead = 0;  // FLOPs queued ahead on the GPU (running + earlier staged)
  auto cpu_rate = [&] {
    double ex = g_cpu.pub_exec_train.load() + g_cpu.pub_exec_infer.load();
    return st.t_cpu_occ > 0.5 && ex > 0 ? ex / st.t_cpu_occ : g_cap.cpu;
  };
  auto gpu_rate = [&] {
    return st.t_gpu > 0.5 && st.exec_gpu > 0 ? std::min(st.exec_gpu / st.t_gpu, g_cap.gpu_share)
                                             : g_cap.gpu_share;
  };
  auto cpu_finishes_sooner = [&](const Task& t) {
    const double f = remaining_flops(t);
    const double gr = gpu_rate();
    const double t_gpu = gpu_ahead / gr + f / std::max(1.0, std::min(g_cost[t.type][t.shape].demand, gr));
    return f / cpu_rate() < t_gpu;
  };
  auto try_place = [&](Staged& x) {
    const Task& t = x.task;
    const GpuCost& c = g_cost[t.type][t.shape];
    if (route_to_gpu(t.type, t.shape)) {
      if (cpu_slot && cfg.rebalance && cfg.pack && cpu_idle() && !jobs.empty() &&
          cpu_finishes_sooner(t)) {
        st.rebalanced++;
        start_cpu(x);
        return true;
      }
      if (aged && aged != &x) return false;
      if (!gpu_fits(c)) {
        gpu_ahead += remaining_flops(t);  // later staged tasks queue behind this one
        return false;
      }
      if (start_gpu(x, false)) return true;
      if (jobs.empty() && !g_cpu.running) {  // nothing to wait for: use the CPU
        st.gpu_fallbacks++;
        if (fallback_logs++ < 3)
          LOG("GPU unavailable for %s task %lld (%s) - running it on the CPU", tname(t.type),
              (long long)t.id, gpu::last_error());
        start_cpu(x);
        return true;
      }
      return false;
    }
    if (cpu_slot && cpu_idle()) {
      start_cpu(x);
      return true;
    }
    if (cfg.spill && cfg.pack && !gpu_waiting && gpu_fits(c) && start_gpu(x, true)) return true;
    return false;
  };
  auto handle_result = [&](Task&& task, Status s, bool on_gpu) {
    const char* dev = on_gpu ? "gpu" : "cpu";
    switch (s) {
      case COMPLETED:
        if (task.type == TRAIN) {
          st.train_done++;
          st.flops_train += task.flops_run;
          if (on_gpu) st.train_done_gpu++, st.flops_train_gpu += task.flops_run;
        } else {
          st.infer_done++;
          st.flops_infer += task.flops_run;
          st.checksum += task.acc;
          if (on_gpu) st.infer_done_gpu++, st.flops_infer_gpu += task.flops_run;
        }
        break;
      case PREEMPTED:
        task.origin = ORIG_PREEMPT;
        st.preempts++;
        force_infer++;
        LOG("priority -> %.3f: training task %lld (%s) checkpointed at epoch %d/%d, switching "
            "to inference", cur_p, (long long)task.id, dev, task.progress, cfg.epochs);
        pending.push_back(std::move(task));
        dirty = true;
        break;
      case SUSPENDED:
        task.origin = ORIG_SUSPEND;
        LOG("suspend: %s task %lld (%s) checkpointed at %d", tname(task.type),
            (long long)task.id, dev, task.progress);
        pending.push_front(std::move(task));  // in-flight work drains first on resume
        dirty = true;
        break;
      case EXIT_SAVE:
        task.origin = ORIG_EXIT;
        LOG("exit e:1: %s task %lld (%s) checkpointed at %d", tname(task.type),
            (long long)task.id, dev, task.progress);
        pending.push_front(std::move(task));
        dirty = true;
        break;
      case EXIT_NOSAVE:
        st.flops_discarded += task.flops_run;
        LOG("exit e:0: %s task %lld (%s) abandoned at %d (no checkpoint)", tname(task.type),
            (long long)task.id, dev, task.progress);
        break;
      default:
        break;
    }
  };
  auto unstage_all = [&] {  // staged tasks become (progress 0) checkpoints
    while (!staged.empty()) {
      pending.push_front(std::move(staged.back().task));
      staged.pop_back();
      dirty = true;
    }
  };

  while (true) {
    // ---- time accounting for the interval since the previous iteration ----
    double now = MPI_Wtime();
    double dt = now - t_last;
    t_last = now;
    st.sched_iters++;
    if (!jobs.empty()) {
      st.t_gpu += dt;
      st.conc_int += jobs.size() * dt;
      st.dem_int += std::min(sum_dem, g_cap.gpu_share) * dt;
    }
    if (g_cpu.running) st.t_cpu_occ += dt;
    if (!jobs.empty() || g_cpu.running) st.t_busy += dt;
    else if (susp_me) st.t_idle += dt;
    sample_timeline(now);
    bool prog = false;
    if (now - last_poll >= poll_s) poll_control();

    // ---- 1. propagate control state to running tasks ----
    const int intr = cur_exit == 0 ? EXIT_NOSAVE
                     : cur_exit == 1 ? EXIT_SAVE
                     : susp_me       ? SUSPENDED
                                     : CONTINUE;
    if (intr == CONTINUE && prio_event) {
      int n = 0;
      for (auto& j : jobs)
        if (j.task.type == TRAIN && j.stop_req == CONTINUE) j.stop_req = PREEMPTED, ++n;
      if (g_cpu.running && cpu_type == TRAIN && g_cpu.intr.load() == CONTINUE)
        g_cpu.intr.store(PREEMPTED), ++n;
      // not-yet-started training tasks step aside as well
      for (auto it = staged.begin(); it != staged.end();) {
        if (it->task.type == TRAIN && !it->from_drain) {
          pending.push_back(std::move(it->task));
          it = staged.erase(it);
          dirty = true;
        } else {
          ++it;
        }
      }
      if (n)
        LOG("priority -> %.3f: checkpointing %d running training task(s) before taking "
            "inference", cur_p, n);
    } else if (intr != CONTINUE) {
      for (auto& j : jobs)
        if (j.stop_req == CONTINUE || intr == EXIT_NOSAVE) j.stop_req = intr;
      if (g_cpu.running && (g_cpu.intr.load() == CONTINUE || intr == EXIT_NOSAVE))
        g_cpu.intr.store(intr, std::memory_order_release);
    }
    prio_event = false;

    // ---- 2. advance GPU jobs ----
    for (size_t i = 0; i < jobs.size();) {
      Status s = job_service(jobs[i], &prog);
      if (s == CONTINUE) {
        ++i;
        continue;
      }
      GpuJob j = std::move(jobs[i]);
      jobs[i] = std::move(jobs.back());
      jobs.pop_back();
      free_slots.push_back(j.slot);
      sum_dem -= j.cost.demand;
      if (j.cost.gemv) sum_bw -= j.cost.demand;
      sum_mem -= j.cost.mem;
      if (jobs.empty()) sum_dem = sum_bw = sum_mem = 0;  // no drift
      gpu_mem_full = false;
      handle_result(std::move(j.task), s, true);
      prog = true;
    }
    // ---- 3. CPU worker ----
    if (g_cpu.running && g_cpu.done.load(std::memory_order_acquire)) {
      Task t;
      Status s;
      {
        std::lock_guard<std::mutex> lk(g_cpu.m);
        t = std::move(g_cpu.task);
        s = g_cpu.result;
        g_cpu.done.store(false);
      }
      g_cpu.running = false;
      handle_result(std::move(t), s, false);
      prog = true;
    }
    if (dirty) {
      save_pending();
      dirty = false;
    }

    // ---- 4. exit ----
    if (cur_exit >= 0) {
      if (jobs.empty() && !g_cpu.running) {
        if (cur_exit == 1) {
          unstage_all();
          save_pending();
          LOG("exit e:1: %zu checkpoint(s) in %s", pending.size(), ckpt_path(g_rank).c_str());
        }
        break;
      }
      if (!prog) std::this_thread::sleep_for(std::chrono::microseconds((int)cfg.gpu_poll_us));
      continue;
    }

    // ---- 5. suspension ----
    if (susp_me) {
      if (!jobs.empty() || g_cpu.running) {  // running tasks are still checkpointing
        if (!prog) std::this_thread::sleep_for(std::chrono::microseconds((int)cfg.gpu_poll_us));
        continue;
      }
      if (!was_susp) {
        was_susp = true;
        st.suspends++;
        unstage_all();
        save_pending();
        dirty = false;
        LOG("SUSPENDED: %zu checkpoint(s) saved to %s", pending.size(),
            ckpt_path(g_rank).c_str());
      }
      st.idle_loops++;
      st.loops++;
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
      continue;
    }
    if (was_susp) {
      was_susp = false;
      st.resumes++;
      reload_pending();  // run exactly what was saved
      drain = !pending.empty();
      LOG("RESUMED: %zu checkpoint(s) to drain before new tasks", pending.size());
    }

    // ---- 6. fill free capacity ----
    double ts = MPI_Wtime();
    for (int guard = 0; guard < 4 * gpu::MAX_SLOTS; ++guard) {
      gpu_waiting = false;
      aged = nullptr;
      gpu_ahead = 0;
      for (const GpuJob& j : jobs) gpu_ahead += remaining_flops(j.task);
      for (const Staged& x : staged)
        if (route_to_gpu(x.task.type, x.task.shape)) {
          gpu_waiting = true;
          if (!aged && now - x.t_staged > cfg.max_wait_ms * 1e-3) aged = &x;
        }
      for (auto it = staged.begin(); it != staged.end();) {
        if (try_place(*it)) {
          if (aged == &*it) aged = nullptr;
          it = staged.erase(it);
          prog = true;
        } else {
          ++it;
        }
      }
      // Every staged task left here is blocked on its device. Look ahead only
      // while some resource is free and nothing staged can use it: up to 4
      // tasks to find one that fits the GPU's remaining capacity, up to 8 when
      // the CPU is idle and everything staged is waiting for the GPU.
      const bool want_cpu = cpu_slot && cpu_idle();
      const bool want_gpu = gpu_room() && !aged;
      if (!want_cpu && !want_gpu) break;
      if (staged.size() >= (want_cpu ? 8u : 4u)) break;
      Staged x;
      if (drain) {
        if (pending.empty()) {
          bool outstanding = (g_cpu.running && cpu_from_drain);
          for (auto& j : jobs) outstanding = outstanding || j.from_drain;
          for (auto& s : staged) outstanding = outstanding || s.from_drain;
          if (outstanding) break;  // checkpoints are consumed only when they complete
          drain = false;
          continue;
        }
        x.task = std::move(pending.front());
        pending.pop_front();
        x.from_drain = true;
        note_resumed(x.task);
      } else {
        int type = choose_type();
        auto it = std::find_if(pending.begin(), pending.end(),
                               [&](const Task& t) { return t.type == type; });
        if (it != pending.end()) {
          x.task = std::move(*it);
          pending.erase(it);
          note_resumed(x.task);
        } else {
          x.task.type = type;
          x.task.id = nxtval();
          x.task.shape = pick_shape(type, x.task.id);
          const int64_t one = 1;
          MPI_Accumulate(&one, 1, MPI_INT64_T, 0, type == TRAIN ? W_GTRAIN : W_GINF, 1,
                         MPI_INT64_T, MPI_SUM, g_win);
          MPI_Win_flush(0, g_win);
          (type == TRAIN ? st.train_new : st.infer_new) += 1;
          if (++since_sync >= cfg.sync_every) {
            since_sync = 0;
            sync_global_mix();
          }
        }
      }
      x.t_staged = now;
      staged.push_back(std::move(x));
    }
    st.t_sched += MPI_Wtime() - ts;
    if (dirty) {
      save_pending();
      dirty = false;
    }

    if (!prog) std::this_thread::sleep_for(std::chrono::microseconds((int)cfg.gpu_poll_us));
  }

  st.wall = MPI_Wtime() - t_loop0;
  for (const Task& t : pending) st.flops_pending += t.flops_run;
  st.pending_left = (double)pending.size();
  {  // stop the CPU worker and merge its counters
    std::lock_guard<std::mutex> lk(g_cpu.m);
    g_cpu.quit = true;
  }
  g_cpu.cv.notify_one();
  g_cpu.th.join();
  const int nf = Stats::count();
  std::vector<double> a(nf), b(nf);
  st.to(a.data());
  g_cpu.final_stats.to(b.data());
  for (int i = 0; i < nf; ++i) a[i] += b[i];
  st.from(a.data());
}

// ----------------------------------------------------------------------------
// Report
// ----------------------------------------------------------------------------
static void report(const std::vector<double>& all, const std::vector<int>& tcounts,
                   const std::vector<double>& tl, int64_t nxt, int64_t gtr, int64_t gin) {
  const int nf = Stats::count();
  std::vector<Stats> S(g_np);
  for (int r = 0; r < g_np; ++r) S[r].from(&all[(size_t)r * nf]);
  Stats T;
  {
    std::vector<double> acc(nf, 0.0), v(nf);
    for (auto& s : S) {
      s.to(v.data());
      for (int i = 0; i < nf; ++i) acc[i] += v[i];
    }
    T.from(acc.data());
  }
  double wall = 0;
  for (auto& s : S) wall = std::max(wall, s.wall);
  auto G = [](double f) { return f * 1e-9; };

  std::printf("\n================================================================================"
              "==================\n");
  std::printf(" MPI TRAIN/INFER SIMULATOR — FINAL REPORT   (np=%d, wall=%.2fs, epochs/task=%d, "
              "final p=%.3f)\n", g_np, wall, cfg.epochs, cur_p);
  std::printf("=================================================================================="
              "================\n");
  std::printf(" rank |  loops   idle | train  infer | train GFLOP  infer GFLOP | busy%%  idle%%  "
              "ovh%% | GF/s busy | preempt resumed ckpt-wr\n");
  std::printf("------+---------------+--------------+--------------------------+--------------------"
              "+-----------+------------------------\n");
  for (int r = 0; r < g_np; ++r) {
    const Stats& s = S[r];
    double busy = s.t_busy;  // time with any task running (CPU or GPU)
    double w = std::max(s.wall, 1e-12);
    std::printf(" %4d | %6.0f %6.0f | %5.0f  %5.0f | %11.4f  %11.4f | %5.1f  %5.1f %5.1f | %9.3f "
                "| %7.0f %7.0f %7.0f\n",
                r, s.loops, s.idle_loops, s.train_done, s.infer_done, G(s.flops_train),
                G(s.flops_infer), 100 * busy / w, 100 * s.t_idle / w,
                100 * (w - busy - s.t_idle) / w, G(s.exec_train + s.exec_infer) / std::max(busy, 1e-12),
                s.preempts, s.train_resumed + s.infer_resumed, s.ckpt_writes);
  }
  std::printf("------+---------------+--------------+--------------------------+--------------------"
              "+-----------+------------------------\n");
  double tb = T.t_busy;
  std::printf("  all | %6.0f %6.0f | %5.0f  %5.0f | %11.4f  %11.4f | %5.1f  %5.1f %5.1f | %9.3f "
              "| %7.0f %7.0f %7.0f\n",
              T.loops, T.idle_loops, T.train_done, T.infer_done, G(T.flops_train),
              G(T.flops_infer), 100 * tb / (wall * g_np), 100 * T.t_idle / (wall * g_np),
              100 * (T.wall - tb - T.t_idle) / (wall * g_np),
              G(T.exec_train + T.exec_infer) / std::max(tb, 1e-12), T.preempts,
              T.train_resumed + T.infer_resumed, T.ckpt_writes);

  std::printf("\n Flop counts are BLAS kernel flops of COMPLETED tasks, per rank (training and "
              "inference kept separate).\n");
  std::printf("   training : %.4f GFLOP total   (%.0f tasks, %.0f epochs executed, mean "
              "loss[end]/loss[0] = %.4f)\n", G(T.flops_train), T.train_done, T.epochs_run,
              T.train_done > 0 ? T.loss_ratio_sum / T.train_done : 0.0);
  std::printf("   inference: %.4f GFLOP total   (%.0f tasks, %.0f samples executed, output "
              "checksum %.6e)\n", G(T.flops_infer), T.infer_done, T.samples_run, T.checksum);
  std::printf("   executed (incl. partial work)  : train %.4f + infer %.4f GFLOP\n",
              G(T.exec_train), G(T.exec_infer));
  std::printf("   partial work held in checkpoints: %.4f GFLOP in %.0f checkpoint(s)\n",
              G(T.flops_pending), T.pending_left);
  std::printf("   discarded by e:0                : %.4f GFLOP\n", G(T.flops_discarded));
  std::printf("   aggregate throughput            : %.3f GFLOP/s useful (completed) over wall, "
              "%.3f GFLOP/s per rank while busy\n",
              G(T.flops_train + T.flops_infer) / std::max(wall, 1e-12),
              G(T.exec_train + T.exec_infer) / std::max(tb, 1e-12));

  std::printf("\n Scheduling\n");
  std::printf("   NXTVAL final value = %lld  (new tasks handed out; %lld train / %lld infer -> "
              "infer fraction %.3f)\n", (long long)nxt, (long long)gtr, (long long)gin,
              (gtr + gin) ? (double)gin / (gtr + gin) : 0.0);
  std::printf("   total simulation loops = %.0f (%.0f task runs, %.0f idle/suspended loops)\n",
              T.loops, T.task_runs, T.idle_loops);
  std::printf("   control polls = %.0f, control changes seen = %.0f (per rank avg), global-mix "
              "syncs = %.0f\n", T.ctl_polls, T.ctl_changes / g_np, T.syncs);
  std::printf("   preemptions (p rose toward inference) = %.0f, suspends = %.0f, resumes = %.0f, "
              "checkpoint bytes written = %.0f\n", T.preempts, T.suspends, T.resumes, T.ckpt_bytes);
  std::printf("   time split (all ranks): CPU train %.1f%%  CPU infer %.1f%%  GPU active %.1f%%  "
              "suspended %.1f%%  control %.1f%%  sched %.1f%%  ckpt-io %.2f%%\n",
              100 * T.t_train / T.wall, 100 * T.t_infer / T.wall, 100 * T.t_gpu / T.wall,
              100 * T.t_idle / T.wall, 100 * T.t_ctl / T.wall, 100 * T.t_sched / T.wall,
              100 * T.t_ckpt / T.wall);
  if (g_concurrent)
    std::printf("   (concurrent engine: CPU and GPU time overlap; busy%% = time with any task "
                "running; %.0f orchestrator iterations)\n", T.sched_iters);

  // ---- CPU vs GPU split ----
  static const char* mode_name[] = {"", "cpu", "gpu", "cpu,gpu"};
  std::printf("\n Device split  (exec_mode=%s", mode_name[cfg.exec_mode]);
  if (cfg.exec_mode == EXEC_BOTH)
    std::printf(", tasks > %.3f TFLOP -> GPU", cfg.gpu_threshold * 1e-12);
  std::printf(")\n");
  std::printf(" rank | dev | train cpu/gpu | infer cpu/gpu |   train GFLOP cpu / gpu   |   infer "
              "GFLOP cpu / gpu   | GF/s cpu  GF/s gpu | gpu%%time fallbk\n");
  std::printf("------+-----+---------------+---------------+---------------------------+-----------"
              "----------------+--------------------+----------------\n");
  auto dev_row = [&](const char* lbl, const Stats& s, double wall_s) {
    double tcpu = s.t_train + s.t_infer;  // CPU kernel time
    double ecpu = s.exec_train + s.exec_infer - s.exec_gpu;
    char dev[8];
    if (s.gpu_dev >= 0 && g_gpu_ready) std::snprintf(dev, sizeof dev, "g%-2.0f", s.gpu_dev);
    else std::snprintf(dev, sizeof dev, " - ");
    std::printf(" %4s | %s | %5.0f / %-5.0f | %5.0f / %-5.0f | %11.3f / %-11.3f | %11.3f / %-11.3f "
                "| %8.3f %9.3f | %6.1f %7.0f\n",
                lbl, dev, s.train_done - s.train_done_gpu, s.train_done_gpu,
                s.infer_done - s.infer_done_gpu, s.infer_done_gpu,
                G(s.flops_train - s.flops_train_gpu), G(s.flops_train_gpu),
                G(s.flops_infer - s.flops_infer_gpu), G(s.flops_infer_gpu),
                tcpu > 0 ? G(ecpu) / tcpu : 0.0, s.t_gpu > 0 ? G(s.exec_gpu) / s.t_gpu : 0.0,
                100 * s.t_gpu / std::max(wall_s, 1e-12), s.gpu_fallbacks);
  };
  for (int r = 0; r < g_np; ++r) {
    char lbl[16];
    std::snprintf(lbl, sizeof lbl, "%d", r);
    dev_row(lbl, S[r], S[r].wall);
  }
  {
    Stats TT = T;
    TT.gpu_dev = -1;
    dev_row("all", TT, T.wall);
  }
  std::printf("   GF/s = executed flops / time that device was running work (per rank; 'all' "
              "= ratio of sums).\n");

  // ---- GPU capacity and packing ----
  if (g_gpu_ready) {
    std::printf("\n GPU capacity & packing  (capacity = %s; demand per task from its shape; "
                "packing %s, spill %s, rebalance %s)\n",
                cfg.gpu_gflops > 0 ? "--gpu-gflops" : "gpu_mult x CPU GFLOP/s per process",
                cfg.pack ? "on" : "OFF", cfg.spill ? "on" : "off", cfg.rebalance ? "on" : "off");
    std::printf(" rank | share | CPU GF/s | GPU cap GF/s | strm | jobs  spill rebal | avg conc max | "
                "model util%% | GPU GF/s achieved (%% of cap) | CPU occ%% (kernel%%)\n");
    std::printf("------+-------+----------+--------------+------+-------------------+--------------+----"
                "---------+------------------------------+----------\n");
    auto pack_row = [&](const char* lbl, const Stats& s, double cap, double w, double maxc,
                        double sharers) {
      double ach = s.exec_gpu / std::max(w, 1e-12);
      std::printf(" %4s | 1/%-3.0f | %8.2f | %12.1f | %4.0f | %5.0f %5.0f %5.0f | %8.2f %3.0f | %11.1f | "
                  "%14.2f (%9.1f%%) | %6.1f (%5.1f)\n",
                  lbl, sharers, G(s.cap_cpu), G(cap), s.slot_limit, s.gpu_jobs, s.spilled, s.rebalanced,
                  s.t_gpu > 0 ? s.conc_int / s.t_gpu : 0.0, maxc,
                  100 * s.dem_int / std::max(cap * w, 1e-12), G(ach),
                  100 * ach / std::max(cap, 1e-12), 100 * s.t_cpu_occ / std::max(w, 1e-12),
                  100 * (s.t_train + s.t_infer) / std::max(w, 1e-12));
    };
    double maxc = 0;
    for (int r = 0; r < g_np; ++r) {
      char lbl[16];
      std::snprintf(lbl, sizeof lbl, "%d", r);
      pack_row(lbl, S[r], S[r].cap_gpu, S[r].wall, S[r].max_conc, S[r].sharers);
      maxc = std::max(maxc, S[r].max_conc);
    }
    {  // whole machine: summed capacity and flops over the common wall time
      double cap = T.cap_gpu;
      double ach = T.exec_gpu / std::max(wall, 1e-12);
      std::printf("  all |   -   | %8.2f | %12.1f | %4.0f | %5.0f %5.0f %5.0f | %8.2f %3.0f | %11.1f | "
                  "%14.2f (%9.1f%%) | %6.1f (%5.1f)\n",
                  G(T.cap_cpu / g_np), G(cap), T.slot_limit, T.gpu_jobs, T.spilled, T.rebalanced,
                  T.t_gpu > 0 ? T.conc_int / T.t_gpu : 0.0, maxc,
                  100 * T.dem_int / std::max(cap * wall, 1e-12), G(ach),
                  100 * ach / std::max(cap, 1e-12), 100 * T.t_cpu_occ / std::max(wall * g_np, 1e-12),
                  100 * (T.t_train + T.t_infer) / std::max(wall * g_np, 1e-12));
    }
    std::printf("   model util%% = time-integrated sum of running jobs' modelled demand / capacity "
                "share; achieved = GPU flops / wall;\n   CPU occ%% = time a task was assigned to "
                "the CPU worker, kernel%% = time inside the CPU BLAS loops.\n");
    if (std::strstr(gpu::backend(), "emulation"))
      std::printf("   NOTE: emulated device - 'achieved' is what the host thread(s) executing the "
                  "emulated GPU deliver, not a real GPU.\n");
  }

  // ---- utilization over time: busy% per rank in 10 equal windows ----
  const int NB = 10;
  std::printf("\n Utilization over time (busy%% per window of %.2fs; S = suspended most of the "
              "window)\n   rank ", wall / NB);
  for (int b = 0; b < NB; ++b) std::printf("  w%-3d", b);
  std::printf("  | GFLOP/s per window\n");
  size_t off = 0;
  for (int r = 0; r < g_np; ++r) {
    std::vector<Sample> v(tcounts[r]);
    for (int i = 0; i < tcounts[r]; ++i) {
      const double* p = &tl[(off + i) * SAMPLE_DOUBLES];
      v[i] = {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]};
    }
    off += tcounts[r];
    auto at = [&](double t) {  // linear interpolation of cumulative counters
      Sample z{t, 0, 0, 0, 0, 0, 0, 0};
      if (v.empty() || t <= v.front().t) {
        if (!v.empty()) z = v.front();
        return z;
      }
      for (size_t i = 1; i < v.size(); ++i)
        if (v[i].t >= t) {
          double a = (t - v[i - 1].t) / std::max(v[i].t - v[i - 1].t, 1e-12);
          auto L = [&](double x, double y) { return x + a * (y - x); };
          return Sample{t, L(v[i - 1].exec_train, v[i].exec_train),
                        L(v[i - 1].exec_infer, v[i].exec_infer), L(v[i - 1].busy, v[i].busy),
                        L(v[i - 1].idle, v[i].idle), v[i].susp, v[i].p,
                        L(v[i - 1].exec_gpu, v[i].exec_gpu)};
        }
      return v.back();
    };
    std::printf("   %4d ", r);
    std::string rates;
    double t0 = v.empty() ? 0 : v.front().t;
    double span = (v.empty() ? 0 : v.back().t) - t0;
    for (int b = 0; b < NB; ++b) {
      double a0 = t0 + span * b / NB, a1 = t0 + span * (b + 1) / NB;
      Sample s0 = at(a0), s1 = at(a1);
      double dt = std::max(a1 - a0, 1e-12);
      double busy = (s1.busy - s0.busy) / dt, idle = (s1.idle - s0.idle) / dt;
      if (idle > 0.5)
        std::printf("    S ");
      else
        std::printf(" %4.0f ", 100 * busy);
      char buf[32];
      std::snprintf(buf, sizeof buf, " %.2f",
                    G(s1.exec_train + s1.exec_infer - s0.exec_train - s0.exec_infer) / dt);
      rates += buf;
    }
    std::printf(" |%s\n", rates.c_str());
  }
  std::printf("\n Timeline samples written to %s (rank,t_s,cum_train_flops,cum_infer_flops,"
              "cum_busy_s,cum_idle_s,suspended,p,cum_gpu_flops)\n", cfg.timeline_file.c_str());
  std::printf("=================================================================================="
              "================\n");
  std::fflush(stdout);

  FILE* f = std::fopen(cfg.timeline_file.c_str(), "w");
  if (f) {
    std::fprintf(f, "rank,t_s,cum_train_flops,cum_infer_flops,cum_busy_s,cum_idle_s,suspended,p,"
                 "cum_gpu_flops\n");
    size_t o = 0;
    for (int r = 0; r < g_np; ++r)
      for (int i = 0; i < tcounts[r]; ++i, ++o) {
        const double* p = &tl[o * SAMPLE_DOUBLES];
        std::fprintf(f, "%d,%.4f,%.0f,%.0f,%.5f,%.5f,%.0f,%.4f,%.0f\n", r, p[0], p[1], p[2], p[3],
                     p[4], p[5], p[6], p[7]);
      }
    std::fclose(f);
  }
}

// ----------------------------------------------------------------------------
// Argument parsing
// ----------------------------------------------------------------------------
static void usage() {
  std::printf(
      "usage: mpirun -np N ./mpisim [p:0.5] [s:R] ... [options]\n"
      "  control tokens : p:<0..1>  s:<rank|all>  r:<rank|all>  e:0  e:1\n"
      "                   live: append to the control file (default sim_control.txt) or type on\n"
      "                   rank 0's stdin, e.g.  echo p:0.9 >> sim_control.txt\n"
      "  exec_mode:cpu | exec_mode:gpu | exec_mode:cpu,gpu      (cpu)\n"
      "                   (also --exec-mode MODE, or 'exec_mode: cpu,gpu')\n"
      "                   cpu,gpu: a task runs on the GPU iff its total FLOPs > threshold\n"
      "  --gpu-threshold F FLOPs per task above which cpu,gpu uses the GPU   (1e12)\n"
      " capacity model / packing (exec_mode gpu, cpu,gpu):\n"
      "  --cpu-gflops X    CPU GFLOP/s per process             (0 = calibrate)\n"
      "  --gpu-mult X      GPU GFLOP/s = X * CPU GFLOP/s       (100)\n"
      "  --gpu-gflops X    absolute GPU GFLOP/s per device     (overrides --gpu-mult)\n"
      "  --gpu-streams N   max concurrent streams per device   (16, shared by its ranks)\n"
      "  --no-pack         one task at a time per rank (baseline for comparison)\n"
      "  --no-spill        cpu,gpu: CPU-routed tasks never use spare GPU capacity\n"
      "  --no-rebalance    cpu,gpu: idle CPU never takes a waiting GPU-routed task\n"
      "  --gpu-depth N     units in flight per GPU job          (2, max 4)\n"
      "  --gpu-tile N      DGEMM tile edge in occupancy model   (64)\n"
      "  --ctas-per-sm N   concurrent GEMM tiles per SM         (2)\n"
      "  --launch-us X     kernel launch cost in demand model   (5)\n"
      "  --unit-ms X       target duration of an inference GPU unit (20)\n"
      "  --max-wait-ms X   blocked GPU task age that stops backfilling (200)\n"
      "  --epochs N        training epochs per task            (40)\n"
      "  --nshapes N       shape combinations per workload     (100)\n"
      "  --min-dim N       smallest random dimension           (256)\n"
      "  --max-dim N       largest random dimension, <=8192    (4096)\n"
      "  --max-layers N    inference layers in [2,N], <=16     (8)\n"
      "  --seed S          shape/data seed\n"
      "  --ctl FILE        control file                        (sim_control.txt)\n"
      "  --ckpt-dir DIR    checkpoint directory                (checkpoints)\n"
      "  --timeline FILE   per-rank timeline CSV               (sim_timeline.csv)\n"
      "  --fresh           discard checkpoints left by a previous run\n"
      "  --no-stdin        do not read control tokens from stdin\n"
      "  --sync-every N    decisions between global-mix syncs  (8)\n"
      "  --poll-ms X       control poll interval in kernels    (0.5)\n"
      "  --timeline-ms X   timeline sample interval            (250)\n"
      "  --max-seconds T   inject e:1 after T seconds          (0 = never)\n"
      "  --selftest        verify checkpoint/resume (and CPU<->GPU agreement), then run\n"
      "  --quiet | --verbose\n");
}

static bool parse_args(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        if (g_rank == 0) std::fprintf(stderr, "missing value for %s\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a.size() >= 3 && a[1] == ':' && std::strchr("psre", a[0])) cfg.init_tokens.push_back(a);
    else if (a.rfind("exec_mode", 0) == 0 || a == "--exec-mode") {
      // exec_mode:cpu,gpu | exec_mode=gpu | "exec_mode: cpu,gpu" | --exec-mode cpu
      std::string v;
      if (a == "--exec-mode") v = next("--exec-mode");
      else if (a.size() > 10 && (a[9] == ':' || a[9] == '=')) v = a.substr(10);
      else if (a == "exec_mode" || a == "exec_mode:" || a == "exec_mode=") v = next("exec_mode");
      else {
        if (g_rank == 0) std::fprintf(stderr, "bad argument '%s'\n", a.c_str());
        return false;
      }
      int m = 0;
      std::stringstream ss(v);
      std::string part;
      while (std::getline(ss, part, ',')) {
        part.erase(std::remove(part.begin(), part.end(), ' '), part.end());
        if (part == "cpu") m |= EXEC_CPU;
        else if (part == "gpu") m |= EXEC_GPU;
        else if (!part.empty()) {
          if (g_rank == 0)
            std::fprintf(stderr, "bad exec_mode '%s' (cpu, gpu or cpu,gpu)\n", v.c_str());
          return false;
        }
      }
      if (!m) {
        if (g_rank == 0) std::fprintf(stderr, "empty exec_mode\n");
        return false;
      }
      cfg.exec_mode = m;
    }
    else if (a == "--gpu-threshold") cfg.gpu_threshold = std::stod(next("--gpu-threshold"));
    else if (a == "--cpu-gflops") cfg.cpu_gflops = std::stod(next("--cpu-gflops"));
    else if (a == "--gpu-mult") cfg.gpu_mult = std::stod(next("--gpu-mult"));
    else if (a == "--gpu-gflops") cfg.gpu_gflops = std::stod(next("--gpu-gflops"));
    else if (a == "--gpu-streams")
      cfg.gpu_streams = std::max(1, std::min(gpu::MAX_SLOTS, std::stoi(next("--gpu-streams"))));
    else if (a == "--no-pack") cfg.pack = false;
    else if (a == "--no-spill") cfg.spill = false;
    else if (a == "--no-rebalance") cfg.rebalance = false;
    else if (a == "--gpu-depth")
      cfg.gpu_depth = std::max(1, std::min(gpu::RING, std::stoi(next("--gpu-depth"))));
    else if (a == "--gpu-tile") cfg.gpu_tile = std::max(1, std::stoi(next("--gpu-tile")));
    else if (a == "--ctas-per-sm") cfg.ctas_per_sm = std::max(1, std::stoi(next("--ctas-per-sm")));
    else if (a == "--launch-us") cfg.launch_us = std::max(0.0, std::stod(next("--launch-us")));
    else if (a == "--unit-ms") cfg.unit_ms = std::max(0.01, std::stod(next("--unit-ms")));
    else if (a == "--max-wait-ms") cfg.max_wait_ms = std::max(0.0, std::stod(next("--max-wait-ms")));
    else if (a == "--epochs") cfg.epochs = std::stoi(next("--epochs"));
    else if (a == "--nshapes") cfg.nshapes = std::stoi(next("--nshapes"));
    else if (a == "--min-dim") cfg.min_dim = std::stoi(next("--min-dim"));
    else if (a == "--max-dim") cfg.max_dim = std::stoi(next("--max-dim"));
    else if (a == "--max-layers") cfg.max_layers = std::stoi(next("--max-layers"));
    else if (a == "--seed") cfg.seed = std::stoull(next("--seed"));
    else if (a == "--ctl") cfg.ctl_file = next("--ctl");
    else if (a == "--ckpt-dir") cfg.ckpt_dir = next("--ckpt-dir");
    else if (a == "--timeline") cfg.timeline_file = next("--timeline");
    else if (a == "--fresh") cfg.fresh = true;
    else if (a == "--no-stdin") cfg.use_stdin = false;
    else if (a == "--sync-every") cfg.sync_every = std::max(1, std::stoi(next("--sync-every")));
    else if (a == "--poll-ms") cfg.poll_ms = std::stod(next("--poll-ms"));
    else if (a == "--timeline-ms") cfg.timeline_ms = std::stod(next("--timeline-ms"));
    else if (a == "--max-seconds") cfg.max_seconds = std::stod(next("--max-seconds"));
    else if (a == "--selftest") cfg.selftest = true;
    else if (a == "--quiet") cfg.verbose = 0;
    else if (a == "--verbose") cfg.verbose = 2;
    else if (a == "-h" || a == "--help") {
      if (g_rank == 0) usage();
      return false;
    } else {
      if (g_rank == 0) std::fprintf(stderr, "unknown argument '%s' (try --help)\n", a.c_str());
      return false;
    }
  }
  cfg.max_dim = std::min(std::max(cfg.max_dim, 1), DIM_CAP);
  cfg.min_dim = std::min(std::max(cfg.min_dim, 1), cfg.max_dim);
  cfg.max_layers = std::min(std::max(cfg.max_layers, 2), MAXL);
  cfg.nshapes = std::max(cfg.nshapes, 1);
  cfg.epochs = std::max(cfg.epochs, 1);
  return true;
}

// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
  int provided = 0;
  // Only the main thread calls MPI; the monitor thread is MPI-free.
  // The concurrent engine (exec_mode gpu / cpu,gpu) runs CPU kernels on a
  // worker thread; only the main thread communicates, but the worker calls
  // MPI_Wtime, so ask for MPI_THREAD_MULTIPLE in that case.
  bool want_mt = false;
  for (int i = 1; i < argc; ++i)
    if (std::strstr(argv[i], "gpu") && std::strncmp(argv[i], "--gpu", 5) != 0) want_mt = true;
  MPI_Init_thread(&argc, &argv, want_mt ? MPI_THREAD_MULTIPLE : MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_np);
  g_t0 = MPI_Wtime();
  MPI_Bcast(&g_t0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (!parse_args(argc, argv)) {
    MPI_Finalize();
    return 0;
  }
  if (openblas_set_num_threads) openblas_set_num_threads(1);
  poll_s = cfg.poll_ms * 1e-3;
  g_rng.seed(cfg.seed ^ (0x5851f42d4c957f2dULL * (uint64_t)(g_rank + 1)));
  make_shapes();
  precompute_task_flops();

  // GPU setup: node-local rank picks device (local_rank % #devices)
  st.gpu_dev = -1;
  gpu::DeviceInfo ginfo{};
  if (cfg.exec_mode & EXEC_GPU) {
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &node);
    int local_rank = 0;
    MPI_Comm_rank(node, &local_rank);
    int rc = gpu::compiled() ? gpu::init(local_rank, cfg.gpu_streams, &ginfo) : gpu::ERR;
    int dev = rc == gpu::OK ? ginfo.device : -1;
    int ok = dev >= 0, all_ok = 0;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!ok) std::fprintf(stderr, "r%d: GPU initialisation failed: %s\n", g_rank, gpu::last_error());
    if (!all_ok) {
      if (g_rank == 0)
        std::fprintf(stderr, "exec_mode includes gpu but not every rank could use a GPU; "
                             "run with exec_mode:cpu or fix the GPU setup.\n");
      MPI_Comm_free(&node);
      gpu::shutdown();
      MPI_Finalize();
      return 1;
    }
    g_gpu_ready = true;
    g_concurrent = true;
    st.gpu_dev = dev;

    // ---- capacity model ----
    // ranks sharing this device split its capacity, streams and memory
    MPI_Comm devcomm;
    MPI_Comm_split(node, dev, local_rank, &devcomm);
    MPI_Comm_size(devcomm, &g_cap.sharers);
    MPI_Comm_free(&devcomm);
    MPI_Comm_free(&node);
    double cpu = cfg.cpu_gflops > 0 ? cfg.cpu_gflops * 1e9 : calibrate_cpu_flops();
    double cpu_sum = 0;
    MPI_Allreduce(&cpu, &cpu_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    g_cap.cpu = cpu_sum / g_np;  // mean CPU FLOP/s per process
    g_cap.gpu_dev = cfg.gpu_gflops > 0 ? cfg.gpu_gflops * 1e9 : cfg.gpu_mult * g_cap.cpu;
    g_cap.gpu_share = g_cap.gpu_dev / g_cap.sharers;
    g_cap.bw_dev = ginfo.mem_bw_gbs * 1e9 / 4.0;  // GEMV: 2 FLOP per 8-byte weight
    g_cap.bw_share = g_cap.bw_dev / g_cap.sharers;
    g_cap.sm = std::max(1, ginfo.sm_count);
    g_cap.slot_limit = cfg.pack ? std::max(1, cfg.gpu_streams / g_cap.sharers) : 1;
    size_t fb = 0, tb = 0;
    if (gpu::mem_info(&fb, &tb) != gpu::OK) fb = (size_t)(ginfo.mem_gb * 1073741824.0);
    g_cap.mem_share = 0.9 * (double)fb / g_cap.sharers;
    precompute_gpu_costs();
    st.cap_cpu = g_cap.cpu;
    st.cap_gpu = g_cap.gpu_share;
    st.slot_limit = g_cap.slot_limit;
    st.sharers = g_cap.sharers;
    LOG("GPU %d (%s, %.1f GB, %d SMs, %.0f GB/s) via %s, node-local rank %d, shared by %d "
        "rank(s)", dev, ginfo.name, ginfo.mem_gb, ginfo.sm_count, ginfo.mem_bw_gbs,
        gpu::backend(), local_rank, g_cap.sharers);
  }

  if (g_rank == 0) {
    mkdir(cfg.ckpt_dir.c_str(), 0755);
    FILE* f = std::fopen(cfg.ctl_file.c_str(), "w");  // start with an empty control file
    if (f) std::fclose(f);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // RMA window: counters + control block on rank 0
  const int nwin = W_SUSP0 + g_np;
  MPI_Win_allocate(g_rank == 0 ? nwin * sizeof(int64_t) : 0, sizeof(int64_t), MPI_INFO_NULL,
                   MPI_COMM_WORLD, &g_winbuf, &g_win);
  if (g_rank == 0) std::fill(g_winbuf, g_winbuf + nwin, 0);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Win_lock_all(MPI_MODE_NOCHECK, g_win);

  // initial control state from argv
  std::thread mon;
  if (g_rank == 0) {
    ctl.susp.assign(g_np, 0);
    ctl.p = 0.5;
    ctl.version = 1;
    for (auto& t : cfg.init_tokens) apply_token(t, true);
    publish_if_needed();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  poll_control();
  if (g_rank == 0) mon = std::thread(monitor_thread);

  if (g_rank == 0) {
    double tf = 0, inf = 0;
    for (auto& s : g_tshapes) tf += train_epoch_flops(s) * cfg.epochs;
    for (auto& s : g_ishapes) inf += infer_sample_flops(s) * s.batch;
    std::printf("mpisim: %d ranks (MPI thread level %d), %d train + %d infer shapes, dims in "
                "[%d,%d], %d epochs/task, p=%.3f\n", g_np, provided, cfg.nshapes, cfg.nshapes,
                cfg.min_dim, cfg.max_dim, cfg.epochs, cur_p);
    std::printf("        avg task size: train %.3f GFLOP, infer %.3f GFLOP; control file: %s%s\n",
                tf / cfg.nshapes * 1e-9, inf / cfg.nshapes * 1e-9, cfg.ctl_file.c_str(),
                cfg.use_stdin ? " (+ stdin)" : "");
    static const char* mode_name[] = {"", "cpu", "gpu", "cpu,gpu"};
    int ng[2] = {0, 0};
    for (int ty = 0; ty < 2; ++ty)
      for (int i = 0; i < cfg.nshapes; ++i) ng[ty] += route_to_gpu(ty, i);
    std::printf("        exec_mode=%s", mode_name[cfg.exec_mode]);
    if (cfg.exec_mode == EXEC_BOTH)
      std::printf(" (GPU iff task > %.3f TFLOP)", cfg.gpu_threshold * 1e-12);
    std::printf(": GPU-routed shapes  train %d/%d, infer %d/%d\n", ng[0], cfg.nshapes, ng[1],
                cfg.nshapes);
    if (g_gpu_ready) {
      std::printf("        capacity: CPU %.2f GFLOP/s per process (%s); GPU %.1f GFLOP/s per "
                  "device (%s); per rank %.1f GFLOP/s, %d stream(s); GEMV DRAM bound %.1f "
                  "GFLOP/s\n",
                  g_cap.cpu * 1e-9, cfg.cpu_gflops > 0 ? "given" : "calibrated",
                  g_cap.gpu_dev * 1e-9,
                  cfg.gpu_gflops > 0 ? "given" : (std::to_string((int)cfg.gpu_mult) + " x CPU").c_str(),
                  g_cap.gpu_share * 1e-9, g_cap.slot_limit, g_cap.bw_dev * 1e-9);
      for (int ty = 0; ty < 2; ++ty) {  // what the demand model says about the shapes
        std::vector<double> d;
        for (int i = 0; i < cfg.nshapes; ++i) d.push_back(g_cost[ty][i].demand);
        std::sort(d.begin(), d.end());
        double med = d[d.size() / 2];
        double fit = g_cap.gpu_share / std::max(med, 1.0);
        bool bw_bound = ty == INFER && g_cap.bw_share > 0 && g_cap.bw_share < g_cap.gpu_share;
        if (bw_bound) fit = g_cap.bw_share / std::max(med, 1.0);
        std::printf("        %s demand per task (GFLOP/s): min %.1f  median %.1f  max %.1f  -> "
                    "~%.1f median tasks fit one rank's %s share\n",
                    ty == TRAIN ? "train" : "infer", d.front() * 1e-9, med * 1e-9,
                    d.back() * 1e-9, fit, bw_bound ? "DRAM-bandwidth" : "FLOP");
      }
      std::printf("        packing %s (max %d concurrent GPU jobs per rank)%s\n",
                  cfg.pack ? "on" : "OFF", g_cap.slot_limit,
                  cfg.exec_mode == EXEC_BOTH
                      ? (cfg.spill ? "; CPU worker runs CPU-routed tasks concurrently; spill on"
                                   : "; CPU worker runs CPU-routed tasks concurrently; spill off")
                      : "");
    }
    std::fflush(stdout);
  }

  // checkpoints left by a previous run (e:1 / suspension)
  if (cfg.fresh) {
    std::remove(ckpt_path(g_rank).c_str());
  } else if (read_tasks(ckpt_path(g_rank), pending) && !pending.empty()) {
    for (auto& t : pending) t.origin = ORIG_PREVRUN, t.flops_run = 0;
    LOG("loaded %zu checkpoint(s) from %s, draining them first", pending.size(),
        ckpt_path(g_rank).c_str());
  }

  if (cfg.selftest && g_rank == 0) {
    std::printf("checkpoint/resume self test:\n");
    bool ok = selftest();
    std::printf("self test %s\n", ok ? "PASSED" : "FAILED");
    std::fflush(stdout);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  sync_global_mix();
  sample_timeline(MPI_Wtime(), true);

  if (g_concurrent) run_loop_concurrent();
  else run_loop();

  sample_timeline(MPI_Wtime(), true);
  if (g_rank == 0) {
    monitor_stop = true;
    mon.join();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  int64_t nxt = 0, gtr = 0, gin = 0;
  if (g_rank == 0) {
    nxt = rma_read(W_NXT);
    gtr = rma_read(W_GTRAIN);
    gin = rma_read(W_GINF);
  }
  MPI_Win_unlock_all(g_win);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Win_free(&g_win);

  // gather stats + timelines
  const int nf = Stats::count();
  std::vector<double> mine(nf), all(g_rank == 0 ? (size_t)nf * g_np : 0);
  st.to(mine.data());
  MPI_Gather(mine.data(), nf, MPI_DOUBLE, all.data(), nf, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  int cnt = (int)g_timeline.size();
  std::vector<int> counts(g_np), displs(g_np);
  MPI_Gather(&cnt, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<double> flat((size_t)cnt * SAMPLE_DOUBLES);
  for (int i = 0; i < cnt; ++i) {
    const Sample& s = g_timeline[i];
    double v[SAMPLE_DOUBLES] = {s.t,    s.exec_train, s.exec_infer, s.busy,
                                s.idle, s.susp,       s.p,          s.exec_gpu};
    std::copy(v, v + SAMPLE_DOUBLES, &flat[(size_t)i * SAMPLE_DOUBLES]);
  }
  std::vector<int> rc(g_np), rd(g_np);
  size_t total = 0;
  if (g_rank == 0)
    for (int r = 0; r < g_np; ++r) {
      rc[r] = counts[r] * SAMPLE_DOUBLES;
      rd[r] = (int)total;
      total += rc[r];
    }
  std::vector<double> tl(total);
  MPI_Gatherv(flat.data(), cnt * SAMPLE_DOUBLES, MPI_DOUBLE, tl.data(), rc.data(), rd.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (g_rank == 0) report(all, counts, tl, nxt, gtr, gin);
  if (g_sink == 12345.678) std::printf(" ");  // keep dX computation observable
  gpu::shutdown();
  MPI_Finalize();
  return 0;
}
