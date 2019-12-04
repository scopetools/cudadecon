GTEST_DIR=~/gtest
INC_PATH+=-I $(GTEST_DIR)/include -I/usr/local/cuda/include -I./
LIB_PATH+=-L $(GTEST_DIR)/lib -L/usr/local/cuda/lib64
LIBS=
LIBS+=-lgtest_main -lgtest -pthread -lcuda -lcudart

#CXXFLAGS+=$(LIBS) $(INC_PATH) $(LIB_PATH) -std=c++0x -g -O0
CXXFLAGS+=$(LIBS) $(INC_PATH) $(LIB_PATH) -std=c++0x -O3
#CXX=/scr_3/gcc/gcc-4.6.3/bin/g++

BUFFER_OBJECT_FILES=Buffer.o CPUBuffer.o GPUBuffer.o PinnedCPUBuffer.o
all: $(BUFFER_OBJECT_FILES)

example: bufferExample

bufferExample: bufferExample.cpp $(BUFFER_OBJECT_FILES)

tests=test_CPUBuffer test_GPUBuffer

CPUBuffer.o: Buffer.o CPUBuffer.h CPUBuffer.cpp
GPUBuffer.o: Buffer.o GPUBuffer.h GPUBuffer.cpp
PinnedCPUBuffer.o: Buffer.o PinnedCPUBuffer.h PinnedCPUBuffer.cpp

test_CPUBuffer: test_CPUBuffer.cpp GPUBuffer.o CPUBuffer.o Buffer.o

test_GPUBuffer: test_GPUBuffer.cpp GPUBuffer.o CPUBuffer.o Buffer.o

.PHONY: check
check: $(tests)
	for t in $(tests) ; do \
		./$$t ; \
	done

.PHONY: clean
clean:
	rm -f $(tests) *.o

