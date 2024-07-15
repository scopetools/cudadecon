#!/bin/bash

export PATH="$PATH:$BUILD_PREFIX/nvvm/bin/"

mkdir cmake_build
cd cmake_build
cmake ${CMAKE_ARGS} -DCMAKE_BUILD_TYPE=Release ../src
make
make install
