# cudaDecon

[![DOI](https://zenodo.org/badge/172128164.svg)](https://zenodo.org/badge/latestdoi/172128164)
[![Conda](https://img.shields.io/conda/v/conda-forge/cudadecon)](https://github.com/conda-forge/cudadecon-feedstock)
![Conda Platform](https://img.shields.io/conda/pn/conda-forge/cudadecon)

GPU-accelerated 3D image deconvolution & affine transforms using CUDA.

Python bindings are also available at [pycudadecon](https://github.com/tlambert03/pycudadecon)

## Install

Precompiled binaries available for linux and windows at conda-forge
(see GPU driver requirements [below](#gpu-requirements))

```sh
# install just the executable binary and shared libraries from this repo
conda install -c conda-forge cudadecon

# install binary, libraries, and python bindings
conda install -c conda-forge pycudadecon
```

### GPU requirements

This software requires a CUDA-compatible NVIDIA GPU.

The libraries available on conda-forge have been compiled against different
versions of the CUDA toolkit.  The required CUDA libraries are bundled in the
conda distributions so you don't need to install the CUDA toolkit separately.

If desired, you may specify `cuda-version` as follows:

```sh
conda install -c conda-forge cudadecon cuda-version=<11 or 12>
```

You should also ensure that you have the
[minimum required driver version](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#cuda-11-and-later-defaults-to-minor-version-compatibility)
installed for the CUDA version you are using.

## Usage

```sh
# check that GPU is discovered
cudaDecon -Q

# Basic Usage
# 1. create an OTF from a PSF with "radialft"
radialft /path/to/psf.tif /path/to/otf_output.tif --nocleanup --fixorigin 10

# 2. run decon on a folder of tiffs:
# 'filename_pattern' is a string that must appear in the filename to be processed
cudaDecon $OPTIONS /folder/of/images filename_pattern /path/to/otf_output.tif

# see manual for all of the available arguments
cudaDecon --help
```

-----------------------

## Local build instructions

If you simply wish to use this package, it is best to install the precompiled binaries from conda as [described above](#install)

To build the source locally, you have two options:

### 1. Build using `run_docker_build`

With docker installed, use `.scripts/run_docker_build.sh` with one of the
configs available in `.ci_support`, for instance:

```shell
CONFIG=linux_64_cuda_compiler_version10.2 .scripts/run_docker_build.sh
```

### 2. using cmake directly

This package depends on boost, libtiff, fftw, and cuda.

Here we create a dedicated conda environment with all of the build dependencies
installed, and then use cmake directly.  This method is faster and creates an
immediately useable binary (i.e. it is better for iteration if you're changing
the source code), but requires that you set up build dependencies correctly.

1. install conda ([miniconda](https://docs.conda.io/en/latest/miniconda.html) or [miniforge](https://github.com/conda-forge/miniforge))
1. (*windows only*) install [build tools for
   VisualStudio](https://visualstudio.microsoft.com/downloads)
   2019.  For linux, all necessary build tools will be installed by conda.

1. create a new conda environment with all of the dependencies installed

    ```sh
    conda config --add channels conda-forge
    conda create -n build -y cmake libboost-devel libtiff fftw ninja cuda-nvcc libcufft-dev
    conda activate build  
    # you will need to reactivate the "build" environment each time you close the terminal
    ```

1. create a new `build` directory inside of the top level `cudaDecon` folder

    ```sh
    mkdir build  # inside the cudaDecon folder
    cd build
    ```

1. (*windows only*) Activate your build tools (adjust the path to your
   installation):

    ```cmd
    "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ```

1. Run `cmake` and compile with `ninja` on windows or `make` on linux.

    ```sh
    # windows
    cmake ../src -DCMAKE_BUILD_TYPE=Release -G "Ninja"
    ninja

    # linux
    cmake ../src -DCMAKE_BUILD_TYPE=Release
    make -j4
    ```

    *note that you can specify the CUDA version to use by using the
    `-DCUDA_TOOLKIT_ROOT_DIR` flag*

The binary will be written to `cudaDecon\build\<platform>-<compiler>-release`.
If you change the source code, you can just rerun `ninja` or `make` and the
binary will be updated.

### Testing

There is some test data included in test_data.  You can use it to test the binaries
created in the previous step.  For example, if you are on windows and followed the steps
above, your binary will be in `cudaDecon\build\windows-msvc-release\cudaDecon.exe`.

First build an otf:

```sh
.\build\windows-msvc-release\radialft.exe .\test_data\psf.tif .\test_data\otf.tif --nocleanup --fixorigin 10
```

Then run the deconvolution:

```sh
.\build\windows-msvc-release\cudaDecon.exe -z 0.3 -i 10 -D 31.5 .\test_data\ im_raw .\test_data\otf.tif
```

-----------------------

## Developer Notes

* GPU based resources have a `d_` prefix in their name such as : GPUBuffer &
  d_interpOTF

* transferConstants() is a function to send small data values from host to GPU
  device.

* The link between the function arguments of "transferConstants()" and the
  globals like : __constant__ unsigned const_nzotf; are found in RLgpuImpl.cu
  with calls like : cutilSafeCall(cudaMemcpyToSymbol(const_nzotf, &nzotf,
  sizeof(int)));

* This RL is based upon the built-in Matlab version : deconvlucy.m (see
  <http://ecco2.jpl.nasa.gov/opendap/hyrax/matlab/images/images/deconvlucy.m>)

* Cudadecon.exe `main()` function is in `src/linearDecon.cpp`

* If not enough memory is on the GPU, the program will use host PC's RAM.

* If you are processing on the GPU that drives the display, Windows will
terminate cudaDecon if an iteration takes too long.  Set the windows display
driver timeout to something larger (like 10 seconds instead of default 5
seconds) : see
<http://stackoverflow.com/questions/17186638/modifying-registry-to-increase-gpu-timeout-windows-7>
Running this command from an adminstrator command prompt should set the timeout
to 10 : `reg.exe ADD
"HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v
"TdrDelay" /t REG_DWORD /D "10" /f`

* Better yet, use a second GPU.  The GPU you wish to use for computation only
  should use the TCC driver (must be a Titan or Tesla or other GPU that supports
  TCC).  This card should be initialized after the display GPU, so put the
  compute card in a slot that is > display card.  The TCC driver is selected
  with NVIDIAsmi.exe -L from an administrator cmd window to show the GPUs, then
  NVIDIAsmi.exe -dm 1 -i 0 to set TCC on GPU 0.  Then use `set
  CUDA_VISIBLE_DEVICES` to pick the GPU the deconv code should execute on.

