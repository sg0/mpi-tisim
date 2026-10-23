# mpisim — MPI training/inference workload simulator

C++17 + MPI-3 simulator. Every rank runs an infinite self-scheduling loop that
pulls tasks from a global counter (NXTVAL) and runs either a **training** task
(DGEMM/DAXPY) or an **inference** task (DGEMV), until told to exit. Each task
runs on the CPU (OpenBLAS) or the GPU (cuBLAS), chosen by `exec_mode`.

| File | Role |
|---|---|
| `sim.cpp` | MPI harness, scheduling, control, checkpoints, CPU kernels, report |
| `gpu_kernels.cu` / `.h` | GPU kernels: cuBLAS + small CUDA kernels, plain C++ API |
| `gpu_stub.cpp` | stand-in for CPU-only builds |
| `cuda_emul.h` | host emulation of the CUDA/cuBLAS subset, for testing without a GPU |

## Build & run

```bash
make                                  # CPU only (exec_mode:gpu exits with an error)
make GPU=1                            # + CUDA/cuBLAS  (CUDA_HOME, CUDA_ARCH="-arch=sm_90", ...)
make EMU=1                            # + GPU path emulated on the host (testing only)
make GPU=1 OPENBLAS_DIR=/opt/openblas # custom OpenBLAS prefix

mpirun -np 4 ./mpisim p:0.5                          # exec_mode:cpu (default)
mpirun -np 4 ./mpisim exec_mode:gpu                  # every task on the GPU
mpirun -np 4 ./mpisim exec_mode:cpu,gpu --selftest   # route by task FLOPs, verify first
```

`exec_mode` is a startup argument. All of these forms work: `exec_mode:cpu,gpu`,
`exec_mode: cpu,gpu`, `exec_mode=gpu` and `--exec-mode cpu`. Live control
(below) works exactly as before.

## CPU / GPU execution

* **`cpu`** runs the original OpenBLAS kernels, unchanged.
* **`gpu`** runs every task through `gpu_kernels.cu`.
* **`cpu,gpu`** decides per task. Each shape's total task FLOPs are
  precalculated (`epochs·(6BKN+2KN)` for training, `batch·Σ(2·dₗ₊₁·dₗ+dₗ₊₁)`
  for inference). A task above **1 TFLOP** (`--gpu-threshold`) runs on the
  GPU, the rest on the CPU. The startup banner prints how many of the 100
  training and 100 inference shapes are GPU-routed.

GPU implementation (`gpu_kernels.cu`):

* **Training:** `cublasDgemm` (forward), `cublasDgemm` ×2 (dW, dX) and
  `cublasDaxpy` (update). A CUDA kernel computes `dY` and loss partial sums.
* **Inference:** per input vector, per layer: bias copy, then
  `cublasDgemv(beta=1)`, then a ReLU kernel.
* **Layout:** host data stays row-major. Each cuBLAS call is written as the
  transposed column-major problem, and the mapping is commented at each call.
* **Data:** generated on the device with the same counter-based RNG as the
  CPU, so the values are bitwise identical.
* **Determinism:** reductions use fixed-shape strided partial sums (no
  atomics), so they are deterministic.
* **Checkpoints:** they hold the same state as on the CPU (epoch + W, or
  sample index + checksum). A GPU checkpoint can resume on the CPU and vice
  versa.
* **Device selection:** node-local rank % device count. Several ranks can
  share one GPU.
* **Device memory:** each rank keeps a grow-only buffer pool, up to ~1 GB at
  4096-dims. If allocation fails, that task runs on the CPU and the report
  counts it under `fallbk`.

Interrupt points are the same on both devices (between epochs / samples): the
GPU synchronizes once per epoch or sample, so priority preemption, `s:`/`r:`
and `e:0`/`e:1` behave as before.

`--selftest` runs these checks:

* CPU interrupt → checkpoint file → resume is bitwise identical to an
  uninterrupted run.

With a GPU active it also runs:

* the same check on the GPU;
* a CPU run vs a GPU run;
* a GPU checkpoint resumed on the CPU.

Live control from another terminal (same working directory):

```bash
./simctl.sh p:0.9    # favour inference -> running training tasks checkpoint now
./simctl.sh s:1      # suspend rank 1   -> checkpoints/rank_1.ckpt, stops taking tasks
./simctl.sh r:1      # resume rank 1    -> drains its checkpoints, then new tasks
./simctl.sh e:1      # exit, saving checkpoints
./simctl.sh e:0      # exit immediately, no checkpoints
```

Tokens can also be typed on `mpirun`'s stdin (forwarded to rank 0), and
`s:all` / `r:all` work. `--help` lists all options.

**About "checking argc":** a process's `argc/argv` are fixed when it starts,
so they can't change during the run. The `p:/s:/r:/e:` tokens on the command
line set the initial state, and a monitor thread on rank 0 keeps reading the
*same* tokens from the control file (`sim_control.txt`, recreated empty on
start; append with `>>`) and from stdin. That thread never calls MPI
(`MPI_THREAD_FUNNELED`). Rank 0's main thread publishes the state into the RMA
window, and every rank reads it with atomic RMA ops.

## Workloads (100 random shapes each, dimensions in [256, 4096] by default)

Use `--min-dim` and `--max-dim` to change the range (cap 8192), and
`--nshapes` to change the number of shapes. At the defaults:

* Average task size is ~2.3 TFLOP for training and ~95 GFLOP for inference.
* About 63% of training shapes cross the 1 TFLOP threshold, and essentially
  no inference shape does. A GEMV-only inference task tops out near 1.1 TFLOP.
* Lower `--gpu-threshold` (e.g. `1e11`) to send large inference batches to
  the GPU as well.
* CPU-side training tasks at these sizes take minutes. Each CPU epoch can take
  tens of seconds, so control commands take effect at the next epoch
  boundary.

| | Per unit of work | Flops counted |
|---|---|---|
| **Training** (B,K,N), `--epochs` (40) epochs | `Y=XW` (fwd DGEMM) → `dW=XᵀdY` (DGEMM) → `dX=dYWᵀ` (DGEMM) → `W-=lr·dW` (DAXPY) | `6BKN + 2KN` per epoch |
| **Inference** (batch, L∈[2,8], d₀…d_L) | per input vector, per layer: `y=b; y+=W·a` (DGEMV) → ReLU | `Σ(2·dₗ₊₁·dₗ + dₗ₊₁)` per sample |

The training task is a real linear-regression SGD: the loss drops (the report
shows mean final/initial loss ≈ 0.4). Inference accumulates an output checksum.
All task data is regenerated deterministically from `(seed, task id)`, so a
checkpoint only stores mutable state: the epoch number and W for training, the
sample index and checksum for inference. `--selftest` confirms that a
checkpointed and resumed task matches an uninterrupted one bit for bit.

## Scheduling

* **RMA window on rank 0** (`MPI_Win_allocate`, `MPI_Win_lock_all`), all `int64`:
  `[NXTVAL, global train starts, global infer starts, ctl version, priority, exit, suspend[np]]`.
* **NXTVAL** = `MPI_Fetch_and_op(+1, MPI_SUM)` + flush. Each new task gets the
  next global id; the id picks the shape and data. Ranks pull work when they
  are free, so faster or less loaded ranks take more tasks.
* **Equitable mix:** the task type is a biased coin
  `p_eff = p + ½(p − f_global) + ½(p − f_local)`. Here `f_local` is this rank's
  inference fraction, and `f_global` is synced from the window counters every
  `--sync-every` decisions. This keeps both the machine-wide mix and each
  rank's mix tracking `p`. `p=0` and `p=1` are exact.
* All control reads and writes use accumulate-family atomics (`MPI_NO_OP` /
  `MPI_REPLACE`). The version field is written last, so a rank that sees a new
  version also sees the matching fields.

## Control semantics

* **Priority rises (e.g. 0.5→0.9):** kernels poll every `--poll-ms` (0.5 ms),
  i.e. at epoch or sample boundaries. Every in-flight training task on every
  rank checkpoints and the rank's next task is forced to be inference. The
  preempted task is resumed later, when that rank next draws "training",
  instead of taking a new NXTVAL id.
* **`s:R`:** rank R checkpoints its in-flight task (training or inference)
  plus any queued checkpoints to `checkpoints/rank_R.ckpt`, then idles.
  Suspending rank 0 is fine: it keeps polling, so it keeps serving the window.
* **`r:R`:** the rank reloads its file and drains every checkpoint before it
  calls NXTVAL again.
* **`e:1`:** all ranks checkpoint in-flight work and exit. The next launch
  picks up `checkpoints/rank_*.ckpt` automatically and drains them first
  (`--fresh` ignores them).
* **`e:0`:** all ranks drop in-flight work and exit. The dropped flops are
  reported as "discarded".

Each rank's checkpoint file is rewritten atomically (tmp + rename) whenever
its queue changes, so the file always matches the queue.

## Report (additions for GPU)

A **Device split** table gives, per rank:

* train and infer tasks completed on the CPU and on the GPU;
* train and infer GFLOP on the CPU and on the GPU;
* GFLOP/s on each device;
* % of time on the GPU;
* number of CPU fallbacks.

`sim_timeline.csv` gains a `cum_gpu_flops` column.

## Report

The run prints one row per rank, then totals:

* simulation loops (including idle/suspended loops);
* completed train and infer tasks;
* **cumulative training GFLOP and inference GFLOP of completed tasks**;
* busy / suspended / overhead %, and GFLOP/s while busy;
* preemptions, resumes, checkpoint writes.

It then gives totals and global figures:

* executed flops, flops still held in checkpoints, flops discarded by `e:0`;
* NXTVAL final value and the achieved train/infer mix;
* the time split across train, infer, suspended, control, scheduling and
  checkpoint I/O.

**Utilization over time** is a per-rank busy% and GFLOP/s across 10 time
windows, with `S` marking a suspended window. The full time series is written
to `sim_timeline.csv` for plotting.

Notes: OpenBLAS is pinned to 1 thread per rank. When there are more ranks than
cores, busy% is wall time spent inside kernels, so lower GFLOP/s there reflects
time-sharing rather than idleness. Passive-target progress relies on rank 0
entering MPI between kernels, which it does every ≤0.5 ms.
