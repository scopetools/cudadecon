#!/bin/bash

# the environmental variable CUDA_VERSION must be set, e.g.:
# export CUDA_VERSION=10.1 && conda build conda-recipe

rm -rf build
mkdir build
cd build

if [ `uname` == Linux ]; then
    export LDFLAGS="-L${PREFIX}/lib"
    export CC=$PREFIX/bin/x86_64-conda_cos6-linux-gnu-gcc
    export CXX=$PREFIX/bin/x86_64-conda_cos6-linux-gnu-g++
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda-${CUDA_VERSION}"
    CUDA_LIB_DIR="${CUDA_TOOLKIT_ROOT_DIR}"/lib64
fi 
if [ `uname` == Darwin ]; then
    export CC=gcc
    export CXX=g++
    CUDA_TOOLKIT_ROOT_DIR="/Developer/NVIDIA/CUDA-${CUDA_VERSION}"
    CUDA_LIB_DIR="${CUDA_TOOLKIT_ROOT_DIR}"/lib
fi

cmake ../src \
    -Wno-unused-function \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_INSTALL_RPATH:STRING="${PREFIX}/lib" \
    -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLKIT_ROOT_DIR}"


make -j2
make install

if [ `uname` == Darwin ]; then
    cp "${CUDA_LIB_DIR}"/libcufft.*.dylib "${PREFIX}"/lib/
fi

#if [ `uname` == Linux ]; then
#    cp "${CUDA_LIB_DIR}"/libcufft.*.so "${PREFIX}"/lib/
#fi

echo "####################### build.sh done"


