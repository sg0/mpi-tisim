# mpi-tisim — MPI training/inference workload simulator

C++17 + MPI-3 simulator. Every rank runs an infinite self-scheduling loop that
pulls tasks from a global task counter (NXTVAL) and runs either a **training** task
(DGEMM+DAXPY) or an **inference** task (DGEMV), until told to exit. Each task
runs on the CPU (OpenBLAS) or the GPU (cuBLAS), chosen by `exec_mode`. With a
GPU in play, each rank packs several tasks onto the GPU on separate CUDA
streams, up to the GPU's FLOP capacity, while its CPU runs its own task
concurrently.

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
make EMU=1 OPENBLAS_DIR=<openblas built with USE_LOCKING=1>   # GPU path emulated on the host
make GPU=1 OPENBLAS_DIR=/opt/openblas # custom OpenBLAS prefix

mpirun -np 4 ./mpisim p:0.5                          # exec_mode:cpu (default)
mpirun -np 4 ./mpisim exec_mode:gpu                  # every task on the GPU
mpirun -np 4 ./mpisim exec_mode:cpu,gpu --selftest   # route by task FLOPs, verify first
mpirun -np 4 ./mpisim exec_mode:gpu --gpu-mult 250   # GPU = 250 x a CPU process
mpirun -np 4 ./mpisim exec_mode:gpu --no-pack        # baseline: one task at a time per rank
```

The `EMU=1` build calls BLAS from the emulated device thread and the CPU worker
at the same time, so it needs a thread-safe OpenBLAS. Real builds (`GPU=1`) only
call OpenBLAS from one thread per rank.

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
* **Inference:** work is done in chunks of samples (sized to about
  `--unit-ms`, 20 ms, at the task's modelled rate). Per layer: a bias
  broadcast, then **`cublasDgemvStridedBatched`** (one GEMV per sample with
  shared weights, `strideA = 0`), then a ReLU kernel. If the batched GEMV is
  unavailable (CUDA < 11.6 or rejected), it falls back to one `cublasDgemv`
  per sample.
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
* **Job slots:** each slot has its own stream, cuBLAS handle, buffers,
  pinned host scalars and events. cuBLAS runs in **device pointer mode** (the
  learning rate and α/β live on the device), so nothing blocks the host.
* **Device memory:** allocation is stream-ordered (`cudaMallocAsync` /
  `cudaFreeAsync`), so growing one slot's buffers never synchronizes the
  device and stalls the other streams. On out-of-memory, idle slots' pools
  are released and the allocation is retried. If it still fails, the task
  waits for running jobs to finish, or runs on the CPU if nothing is running
  (counted under `fallbk`).
* **Hardware queues:** `CUDA_DEVICE_MAX_CONNECTIONS` is raised to match the
  stream count (max 32) unless it is already set.

Interrupts take effect at unit boundaries, i.e. between epochs or chunks.
Each GPU job keeps `--gpu-depth` units (2) in flight. On an interrupt, those
units finish, then W or the checksum is pulled back, giving exactly the state
a CPU checkpoint holds.

## GPU capacity and packing (exec_mode `gpu`, `cpu,gpu`)

**Capacity.** GPU FLOP/s per device = `--gpu-mult` (100) × the CPU FLOP/s of
one process. The CPU rate is calibrated at startup with a single-threaded
DGEMM, averaged over ranks, or set with `--cpu-gflops`; `--gpu-gflops` sets an
absolute value instead. Ranks on the same node that map to the same device
split its capacity, streams (`--gpu-streams`, 16 per device), and free memory.

**Demand per task.** This is the FLOP/s a task can usefully use, computed
once per shape from its dimensions and operations:

| | occupancy | other bounds |
|---|---|---|
| training (DGEMM) | output tiles of its 3 GEMMs (`--gpu-tile` 64²) ÷ (SMs × `--ctas-per-sm` 2) | launch-bound: epoch FLOPs ÷ (9 launches × `--launch-us`) |
| inference (GEMV) | rows in flight per chunk ÷ (SMs × 2048 threads) | DRAM-bound: GEMV does 2 FLOP per 8-byte weight ⇒ ≤ bandwidth/4; launch-bound |

`demand = min(capacity × occupancy, bounds)`. For example, if the device
delivers 1000 GFLOP/s and a task needs 100, there are 900 GFLOP/s left for
more tasks.

**Packing.** A rank starts a new GPU job while all of these hold:

* jobs < its stream share;
* Σ demand + new ≤ its capacity share;
* Σ GEMV demand + new ≤ its bandwidth share;
* the memory estimate fits.

A task larger than the whole share still runs when the GPU is otherwise
empty. The scheduler looks ahead up to 4 tasks (8 when the CPU is idle) to
find work that fits.

Rules in `cpu,gpu` mode:

* **Home device:** set by the `--gpu-threshold` rule, as before.
* **CPU worker:** the CPU runs its task on a worker thread, concurrently
  with the GPU jobs.
* **Spill** (`--no-spill` turns it off): a CPU-routed task may use spare GPU
  capacity while the CPU is busy. It never does while a GPU-routed task is
  waiting.
* **Rebalance** (`--no-rebalance` turns it off): an idle CPU takes a waiting
  GPU-routed task if it would finish sooner there. The comparison uses
  measured rates: remaining FLOPs ÷ CPU rate vs (FLOPs queued ahead on the
  GPU + its own) ÷ GPU rate. With a GPU about 100× faster this almost never
  happens. It matters when the GPU is the bottleneck.
* **No starvation:** smaller tasks may backfill past a blocked large one,
  until that one has waited `--max-wait-ms` (200). After that, only it may
  start on the GPU.

What the model predicts at the default shapes (256–4096): one training task's
GEMMs already fill the device, so training tasks don't pack. Inference is
bandwidth-bound, at ~1 task per device. At 64–1024 dimensions, about 3
training and 2 inference tasks fit per GPU. The startup banner prints these
estimates for the current settings.

`--selftest` runs these checks:

* CPU interrupt → checkpoint file → resume is bitwise identical to an
  uninterrupted run.

With a GPU active it also runs:

* the same check on the GPU;
* a CPU run vs a GPU run;
* a GPU checkpoint resumed on the CPU;
* 6 tasks resident on 6 streams at once, each compared with its CPU result
  (no interference between streams).

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
(`MPI_THREAD_FUNNELED`; `MULTIPLE` is requested in GPU modes because the CPU
worker thread calls `MPI_Wtime`). Rank 0's main thread publishes the state into the RMA
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

The same semantics apply in every mode. With packing, "the running task"
means *all* running tasks of the rank: every GPU job and the CPU worker.

* **Priority rises (e.g. 0.5→0.9):** kernels poll every `--poll-ms` (0.5 ms),
  i.e. at epoch or sample boundaries. Every in-flight training task on every
  rank checkpoints, and that rank's next N decisions are forced to be
  inference (N = the number of tasks preempted). Training tasks taken but not
  yet started go back into the queue. The
  preempted task is resumed later, when that rank next draws "training",
  instead of taking a new NXTVAL id.
* **`s:R`:** rank R checkpoints its in-flight task (training or inference)
  plus any queued checkpoints to `checkpoints/rank_R.ckpt`, then idles.
  Suspending rank 0 is fine: it keeps polling, so it keeps serving the window.
* **`r:R`:** the rank reloads its file and drains every checkpoint before it
  calls NXTVAL again. With packing, checkpoints are drained concurrently, and
  no new task is taken until all of them have *completed*.
* **`e:1`:** all ranks checkpoint in-flight work and exit. The next launch
  picks up `checkpoints/rank_*.ckpt` automatically and drains them first
  (`--fresh` ignores them).
* **`e:0`:** all ranks drop in-flight work and exit. The dropped flops are
  reported as "discarded".

Each rank's checkpoint file is rewritten atomically (tmp + rename) whenever
its queue changes, so the file always matches the queue.

## Report (additions for GPU)

**Accounting.** In GPU modes, `busy%` is the time with any task running. CPU
and GPU time overlap, and the time split shows CPU kernel time and GPU-active
time separately. In a fresh run that ends with `e:1`, these always balance:

* tasks handed out (NXTVAL) = completed + checkpointed;
* executed FLOPs = completed + checkpointed.

A **Device split** table gives, per rank:

* train and infer tasks completed on the CPU and on the GPU;
* train and infer GFLOP on the CPU and on the GPU;
* GFLOP/s on each device;
* % of time on the GPU;
* number of CPU fallbacks.

A **GPU capacity & packing** table gives, per rank:

* device share and CPU GFLOP/s;
* GPU capacity share and stream limit;
* GPU jobs, spilled and rebalanced tasks;
* average and maximum concurrent jobs;
* **model util%** (time-integrated Σ demand ÷ capacity);
* **achieved GPU GFLOP/s** and its % of capacity, which shows whether the
  capacity model is realistic;
* **CPU occ%** (a task assigned to the CPU worker) and kernel%.

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
