To get this. Please use Subversion (aka SVN.  If on windows, use TortoiseSVN) and perform a check out of :
https://subversion.int.janelia.org/betziglab/projects/
to :
C:\CudaDecon


DLL build instructions:

1. Prerequisites:
1.a. Visual Studio Community (make sure it's supported by CUDA SDK.  I'm using VS Community 2013). Run Windows Updates.
1.b. Install CMake Tools for Visual Studio.  This will give you color coded text when you make edits to CMakeLists.txt : http://cmaketools.codeplex.com/

Run Visual Studio
Select Tools->Visual Studio Command Prompt

At the VS command prompt, change to the Visual C++ installation directory. (The location depends on the system and the Visual Studio installation, but a typical location is C:\Program Files (x86)\Microsoft Visual Studio version\VC\.) 
Then configure this Command Prompt window for 64-bit command-line builds that target x64 platforms, at the command prompt, enter:

cd "\Program Files (x86)\Microsoft Visual Studio 12.0\VC"
vcvarsall amd64
cd \CudaDecon\build

1.b. Install CMAKE v2.6 and later

1.c. Unzip FFTW3 library into C:\fftw3 then created the x64 .lib files:
lib /machine:x64 /def:libfftw3-3.def
lib /machine:x64 /def:libfftw3l-3.def
lib /machine:x64 /def:libfftw3f-3.def

1.d.a Install zlib
Unzip zlib library into c:\zlib\zlib-1.2.11 or equivalent.
Open the solution file : C:\zlib\zlib-1.2.11\contrib\vstudio\vc12\zlibvc.sln in Visual Studio
At the top, change the pulldown menus to "Release" and "x64".
Build the solution. (F7 or from "Build" menu)

1.d. Unzip Libtiff library into C:\libtiff, then build:
cmake -G "Visual Studio 12 2013 Win64" -DZLIB_LIBRARY:STRING=C:\zlib\zlib-1.2.11\contrib\vstudio\vc12\x64\ZlibStatRelease\zlibstat.lib -DZLIB_INCLUDE_DIR:STRING=C:\zlib\zlib-1.2.11

This generates the cmake files and should identify that it found zlib.  Next run :

cmake --build . --config Release
ctest -V -C Release


1.e. Install CUDA SDK (I'm using 8.0)

1.f. Boost installed, and built with :
bootstrap
.\b2 address-model=64

1.g. Make a subdirectory under where the source code (or this README) is located; call it "cmake".  Then copy \\dm11\betziglab\shaol\FindFFTW3.cmake into this folder (FYI, you don't need to do this, since this is now in the SVN repo.)

2. Generate makefiles:
2.a. Make a subdirectory under where the source code (or this README) is located; let's call it "build"
2.b. From the VS command prompt window, cd into the "build" directory just created
2.c. At the prompt type 

cmake -D CMAKE_BUILD_TYPE=Release -G "NMake Makefiles" ..

Make sure there's no error message. To generate makefiles from scratch, the entire content of "build" folder has to be deleted first. 

3. Compile the libraries and executables:
nmake 

4. To generate the .sln files for Visual Studio (so that you have a nice IDE to view the source files), you can create a folder, call it "VS", then run this command within "VS":
cmake .. -G "Visual Studio 12 Wind64"

5. Copy "libfftw3f-3.dll" from C:\fftw3 into the directory with the CudaDecon.exe

********************* Notes ***************

* Compatible GPUs are specified in this "C:\CudaDecon\CMakeLists.txt".  This also sets up all of the linking to dependent libraries.  If you end up adding other code libraries, or changing versions, etc you will want to edit this file.

* GPU based resources have a d_ prefix in their name such as : GPUBuffer & d_interpOTF

* transferConstants() is a function to send small data values from host to GPU device.  
* The link between the function arguments of "transferConstants()" and the globals like : __constant__ unsigned const_nzotf; are found in RLgpuImpl.cu with calls like : cutilSafeCall(cudaMemcpyToSymbol(const_nzotf, &nzotf, sizeof(int)));
 
* RL is based upon the built-in Matlab version : deconvlucy.m (see http://ecco2.jpl.nasa.gov/opendap/hyrax/matlab/images/images/deconvlucy.m)

*  Cudadecon.exe 
Main function is in LinearDecon.cpp. 

* Set windows display driver timeout to something larger (like 10 seconds instead of default 5 seconds) :
reg.exe ADD "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v "TdrDelay" /t REG_DWORD /D "30" /f

* Better yet, use a second GPU.  The GPU you wish to use for computation only should use the TCC driver (must be a Titan 
	or Tesla or other GPU that supports TCC).  This card should be initialized after the display GPU, so put the compute 
	card in a slot that is > display card.  The TCC driver is selected with NVIDIAsmi.exe -L from an administrator cmd window
	to show the GPUs, then NVIDIAsmi.exe -dm 1 -i 0 to set TCC on GPU 0.  Then use "set CUDA_VISIBLE_DEVICES" to pick 
	the GPU the deconv code should execute on. 