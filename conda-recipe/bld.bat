setlocal EnableDelayedExpansion

:: remove -GL from CXXFLAGS
set "CXXFLAGS=-MD"

mkdir build
cd build

set BUILD_CONFIG=Release
set CUDA_TOOLKIT_ROOT_DIR=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v%CUDA_VERSION%

cmake ../src -G "Ninja" ^
    -Wno-dev -Wno-unused-function ^
    -DCMAKE_BUILD_TYPE:STRING=Release ^
    -DCMAKE_INSTALL_PREFIX:PATH="%LIBRARY_PREFIX%" ^
    -DCMAKE_PREFIX_PATH:PATH="%LIBRARY_PREFIX%" ^
    -DCUDA_TOOLKIT_ROOT_DIR="%CUDA_TOOLKIT_ROOT_DIR%"
if errorlevel 1 exit 1

ninja
if errorlevel 1 exit 1

ninja install
if errorlevel 1 exit 1
