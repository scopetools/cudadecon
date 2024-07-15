mkdir cmake_build
cd cmake_build

:: Handle external CTK in CUDA 11 builds
if defined CUDA_PATH (
    SET CUDACXX=%CUDA_PATH%\bin\nvcc.exe
)

cmake -G Ninja ^
    -DCMAKE_INSTALL_PREFIX:PATH="%LIBRARY_PREFIX%" ^
    -DCMAKE_PREFIX_PATH:PATH="%LIBRARY_PREFIX%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    ../src
if errorlevel 1 exit 1

ninja
if errorlevel 1 exit 1

ninja install
if errorlevel 1 exit 1