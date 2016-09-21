# @(#)makefile


FFTWTHREAD   = -lfftw3f_threads -lpthread
FFTWLIBS     = -lfftw3f


LDLIBS      = $(FFTWTHREAD) $(FFTWLIBS)

CUDA_INC_DIR = /usr/local/cuda/include
CUDA_LIB_FLAG = -L/usr/local/cuda/lib64


NVCC = /usr/local/cuda/bin/nvcc

CXXFLAGS  = -Wall -DNDEBUG -I../CImg-1.5.8 -IBuffers
CXXFLAGS += -I/opt/local/include
CXXFLAGS += -I$(CUDA_INC_DIR) -I/usr/local/cuda/samples/common/inc
CXXFLAGS += -I/usr/X11/include
CXXFLAGS += -std=c++11
CXXFLAGS      += -O3
#CXXFLAGS      += -g

LDLIBS  = -lfftw3f -lfftw3f_threads -ltiff -lX11
LDLIBS += -lboost_program_options -lboost_filesystem -lboost_regex -lboost_system
LDLIBS += -lcudart -lcufft
# LDLIBS += -lBuffer
LDFLAGS = $(CUDA_LIB_FLAG)

ifeq ($(shell uname),Darwin)
    LDFLAGS += -L/opt/local/lib -L/usr/X11/lib
endif

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@ 

%.o: %.cu
	$(NVCC) -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 --compiler-options="-Wall -O3 -IBuffers" -c $^ 
# -v


#  -gencode arch=compute_30,code=sm_30
# -I$(CUDA_INC_DIR)

all: cudaDeconv radialft otfviewer

cudaDeconv: linearDecon.o RL-Biggs-Andrews.o boostfs.o RLgpuImpl.o geometryTransform.o Buffers/Buffer.o  Buffers/CPUBuffer.o  Buffers/GPUBuffer.o  Buffers/PinnedCPUBuffer.o
	$(NVCC) -v -o $@ $^ $(LDFLAGS) $(LDLIBS)

radialft: radialft-nonSIM.o
	$(CXX) -o $@ $^ -lboost_program_options -lX11 -ltiff -lfftw3f

otfviewer: OTF_TIFF_viewer.o
	$(CXX) -o $@ $^  -lX11 -ltiff -lpthread

clean:
	$(RM) *.o *.exe *~

depend:
	makedepend -- $(CFLAGS) -- $(SRC)


