# Builds
#   make                       CPU only (GPU entry points are stubs; exec_mode:gpu errors out)
#   make GPU=1                 CPU + CUDA/cuBLAS kernels      (needs nvcc, libcublas)
#   make EMU=1                 CPU + GPU code path emulated on the host (testing without a GPU)
#
# Variables
#   MPICXX=mpicxx  OPENBLAS_DIR=/opt/openblas  CUDA_HOME=/usr/local/cuda
#   CUDA_ARCH="-arch=native"   e.g. "-gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90"
MPICXX       ?= mpicxx
CXX          ?= g++
NVCC         ?= $(CUDA_HOME)/bin/nvcc
CUDA_HOME    ?= /usr/local/cuda
CUDA_ARCH    ?= -arch=native
OPENBLAS_DIR ?= /opt/homebrew/opt/openblas
CXXFLAGS     ?= -O3 -march=native -std=c++17 -Wall -Wextra -Wno-unused-parameter
GPU          ?= 0
EMU          ?= 0

ifneq ($(OPENBLAS_DIR),)
  BLAS_INC = -I$(OPENBLAS_DIR)/include
  BLAS_LIB = -L$(OPENBLAS_DIR)/lib -Wl,-rpath,$(OPENBLAS_DIR)/lib
endif

ifeq ($(GPU),1)
  MODE     = cuda
  GPU_OBJ  = gpu_kernels.o
  GPU_LIBS = -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcublas -lcudart
else ifeq ($(EMU),1)
  MODE     = emu
  GPU_OBJ  = gpu_emul.o
else
  MODE     = cpu
  GPU_OBJ  = gpu_stub.o
endif
STAMP = .build-$(MODE)

mpisim: $(STAMP) sim.o $(GPU_OBJ)
	$(MPICXX) $(CXXFLAGS) -o $@ sim.o $(GPU_OBJ) $(BLAS_LIB) $(GPU_LIBS) -lopenblas -lpthread
	@echo "built mpisim ($(MODE))"

# switching build flavour forces a clean rebuild
$(STAMP):
	rm -f .build-* *.o mpisim
	touch $@

sim.o: sim.cpp gpu_kernels.h $(STAMP)
	$(MPICXX) $(CXXFLAGS) $(BLAS_INC) -c sim.cpp -o $@

gpu_kernels.o: gpu_kernels.cu gpu_kernels.h $(STAMP)
	$(NVCC) -O3 -std=c++17 $(CUDA_ARCH) -c gpu_kernels.cu -o $@

gpu_emul.o: gpu_kernels.cu gpu_kernels.h cuda_emul.h $(STAMP)
	$(CXX) -O2 -std=c++17 -ffp-contract=off -DGPU_EMULATE $(BLAS_INC) -x c++ -c gpu_kernels.cu -o $@

gpu_stub.o: gpu_stub.cpp gpu_kernels.h $(STAMP)
	$(CXX) -O2 -std=c++17 -c gpu_stub.cpp -o $@

clean:
	rm -f mpisim *.o .build-* sim_timeline.csv *.txt
	rm -rf checkpoints

.PHONY: clean
