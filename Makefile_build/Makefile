# @(#)makefile


# custom define CUDA_ROOT as needed
CUDA_ROOT = /usr/local/cuda
CUDA_INC_DIR = $(CUDA_ROOT)/include
CUDA_LIB_FLAG = -L$(CUDA_ROOT)/lib64


NVCC = $(CUDA_ROOT)/bin/nvcc

CXXFLAGS  = -Wall -DNDEBUG -I../Buffers -I..
CXXFLAGS += -I/opt/local/include
CXXFLAGS += -I$(CUDA_INC_DIR) -I$(CUDA_ROOT)/samples/common/inc
CXXFLAGS += -I/usr/X11/include
CXXFLAGS += -std=c++11
CXXFLAGS      += -O2
#CXXFLAGS      += -g

LDLIBS  = -lfftw3f -lfftw3f_threads -ltiff -lX11
LDLIBS += -lboost_program_options -lboost_filesystem -lboost_regex -lboost_system
LDLIBS += -lcudart -lcufft
# LDLIBS += -lBuffer
LDFLAGS = $(CUDA_LIB_FLAG)

ifeq ($(shell uname),Darwin)
    LDFLAGS += -L/opt/local/lib -L/usr/X11/lib
endif

%.o: ../%.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@ 
%.o: ../Buffers/%.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@ 

%.o: ../%.cu
	$(NVCC) -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 --compiler-options="-Wall -O3 -I../Buffers" -c $^ 
#  -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -v


#  -gencode arch=compute_30,code=sm_30
# -I$(CUDA_INC_DIR)

all: cudaDeconv radialft otfviewer


cudaDeconv: linearDecon.o RL-Biggs-Andrews.o boostfs.o RLgpuImpl.o geometryTransform.o Buffer.o CPUBuffer.o GPUBuffer.o PinnedCPUBuffer.o
	$(NVCC) -v -o $@ $^ $(LDFLAGS) $(LDLIBS)

radialft: radialft-nonSIM.o
	$(CXX) -o $@ $^ -lboost_program_options -lX11 -ltiff -lfftw3f

otfviewer: OTF_TIFF_viewer.o
	$(CXX) -o $@ $^  -lX11 -ltiff -lpthread

clean:
	$(RM) *.o *.exe *~

depend:
	makedepend -- $(CFLAGS) -- $(SRC)


