# cudaDecon

[![DOI](https://zenodo.org/badge/172128164.svg)](https://zenodo.org/badge/latestdoi/172128164)

GPU-accelerated 3D image deconvolution & affine transforms using CUDA.

Python bindings are also available at [pycudadecon](https://github.com/tlambert03/pycudadecon)

### Install

Precompiled binaries available for linux and windows at conda-forge
(see GPU driver requirements [below](#gpu-requirements))

```sh
conda install -c conda-forge cudadecon

# or... to also install the python bindings
conda install -c conda-forge pycudadecon
```

### Usage

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

### GPU requirements

This software requires a CUDA-compatible NVIDIA GPU.
The libraries available on conda-forge have been compiled against different versions of the CUDA toolkit.  The required CUDA libraries are bundled in the conda distributions so you don't need to install the CUDA toolkit separately.  If desired, you can pick which version of CUDA you'd like based on your needs, but please note that different versions of the CUDA toolkit have different GPU driver requirements:

To specify a specific cudatoolkit version, install as follows (for instance, to use
`cudatoolkit=10.2`)

```sh
conda install -c conda-forge cudadecon cudatoolkit=10.2
```

| CUDA  | Linux driver | Win driver |
| ----- | ------------ | ---------- |
| 10.2  | ≥ 440.33     | ≥ 441.22   |
| 11.0  | ≥ 450.36.06  | ≥ 451.22   |
| 11.1  | ≥ 455.23     | ≥ 456.38   |
| 11.2  | ≥ 460.27.03  | ≥ 460.82   |


If you run into trouble, feel free to [open an issue](https://github.com/scopetools/cudaDecon/issues) and describe your setup.


----- 

## Notes

* Compatible GPUs are specified in this "C:\cudaDecon\CMakeLists.txt".  This also sets up all of the linking to dependent libraries.  If you end up adding other code libraries, or changing versions, etc you will want to edit this file.  Specifically where you see the lines like : "-gencode=arch=compute_75,code=sm_75"

* GPU based resources have a d_ prefix in their name such as : GPUBuffer & d_interpOTF

* transferConstants() is a function to send small data values from host to GPU device.

* The link between the function arguments of "transferConstants()" and the globals like : __constant__ unsigned const_nzotf; are found in RLgpuImpl.cu with calls like : cutilSafeCall(cudaMemcpyToSymbol(const_nzotf, &nzotf, sizeof(int)));

* This RL is based upon the built-in Matlab version : deconvlucy.m (see http://ecco2.jpl.nasa.gov/opendap/hyrax/matlab/images/images/deconvlucy.m)

* Cudadecon.exe `main()` function is in `src/linearDecon.cpp`

* If not enough memory is on the GPU, the program will use host PC's RAM.

* If you are processing on the GPU that drives the display, Windows will terminate cudaDecon if an iteration takes too long.  Set the windows display driver timeout to something larger (like 10 seconds instead of default 5 seconds) :
see http://stackoverflow.com/questions/17186638/modifying-registry-to-increase-gpu-timeout-windows-7
Running this command from an adminstrator command prompt should set the timeout to 10 :
`reg.exe ADD "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v "TdrDelay" /t REG_DWORD /D "10" /f`

* Better yet, use a second GPU.  The GPU you wish to use for computation only should use the TCC driver (must be a Titan or Tesla or other GPU that supports TCC).  This card should be initialized after the display GPU, so put the compute card in a slot that is > display card.  The TCC driver is selected with NVIDIAsmi.exe -L from an administrator cmd window to show the GPUs, then NVIDIAsmi.exe -dm 1 -i 0 to set TCC on GPU 0.  Then use `set CUDA_VISIBLE_DEVICES` to pick the GPU the deconv code should execute on.

---------------------

## Local build instructions

If you simply wish to use this package, it is best to just install the precompiled binaries from conda as [described above](#install-precompiled-binaries)

To build the source locally, you have two options:

### 1. Build using `run_docker_build`

With docker installed, use `.scripts/run_docker_build.sh` with one of the
configs available in `.ci_support`, for instance:

```
CONFIG=linux_64_cuda_compiler_version10.2 .scripts/run_docker_build.sh
```

### 2. using cmake directly in a conda environment

Here we create a dedicated conda environment with all of the build dependencies
installed, and then use cmake directly.  This method is faster and creates an
immediately useable binary (i.e. it is better for iteration if you're changing
the source code), but requires that you set up build dependencies correctly.
   

1. install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. install [cudatoolkit](https://developer.nvidia.com/cuda-10.1-download-archive-update2) (I haven't yet tried 10.2)
3. (*windows only*) install [build tools for VisualStudio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017) 2017.  For linux, all necessary build tools will be installed by conda.

4. create a new conda environment with all of the dependencies installed

    ```sh
    conda config --add channels conda-forge
    conda create -n build -y cmake boost-cpp libtiff fftw ninja
    conda activate build  
    # you will need to reactivate the "build" environment each time you close the terminal
    ```

5. create a new `build` directory inside of the top level `cudaDecon` folder

    ```sh
    mkdir build  # inside the cudaDecon folder
    cd build
    ```

6. (*windows only*) Activate your build tools:

    ```cmd
    "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ```

7. Run `cmake` and compile with `ninja` on windows or `make` on linux.

    ```sh
    # windows
    cmake ../src -DCMAKE_BUILD_TYPE=Release -G "Ninja"
    ninja

    # linux
    cmake ../src -DCMAKE_BUILD_TYPE=Release
    make -j4
    ```

    *note that you can specify the CUDA version to use by using the `-DCUDA_TOOLKIT_ROOT_DIR` flag* 

The binary will be written to `cudaDecon\build\<platform>-<compiler>-release`.
If you change the source code, you can just rerun `ninja` or `make` and the
binary will be updated.
