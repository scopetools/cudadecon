To get this. Please use Subversion (aka SVN.  If on windows, use TortoiseSVN) and perform a check out of :
https://subversion.int.janelia.org/betziglab/projects/
to :
C:\CudaDecon


DLL build instructions:

1. Prerequisites:
1.a. Visual Studio Community (make sure it's supported by CUDA SDK.  I'm using VS Community 2017). Run Windows Updates.
1.b. Install CMake Tools for Visual Studio.  This will give you color coded text when you make edits to CMakeLists.txt : http://cmaketools.codeplex.com/

Run Visual Studio
Select Tools->Visual Studio Command Prompt

At the VS command prompt, change to the Visual C++ installation directory. (The location depends on the system and the Visual Studio installation, but a file search within the Visual Studio folder (like : C:\Program Files (x86)\Microsoft Visual Studio version\) will help you find it.) 
Then configure this Command Prompt window for 64-bit command-line builds that target x64 platforms, at the command prompt, enter:

"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64


1.b. Install CMAKE v2.6 and later

1.c. Intall FFTW3
* Download the 64-bit version dlls here : http://fftw.org/install/windows.html
* Unzip FFTW3 library into C:\fftw3 then created the x64 .lib files:

cd c:\fftw3
lib /machine:x64 /def:libfftw3-3.def
lib /machine:x64 /def:libfftw3l-3.def
lib /machine:x64 /def:libfftw3f-3.def

1.d.a Install zlib
* Download zlib source code from : https://www.zlib.net/
* Unzip zlib1211.zip source code into c:\zlib\zlib-1.2.11 or equivalent.
* Open the solution file : C:\zlib\zlib-1.2.11\contrib\vstudio\vc14\zlibvc.sln in Visual Studio
* At the top, change the pulldown menus to "Release" and "x64".
* Build the solution. (F7 or from "Build" menu)

1.d. Install Libtiff

* I needed to change libtiff a little bit to deal with custom tiff tags.  please SVN checkout into c:\libtiff from : 
https://subversion.int.janelia.org/betziglab/tool_codes/libtiff/trunk/libtiff

* in the revisions you will see the changes that I made.  If upgrading to new libtiff version, I would recommend : 1. checkout latest SVN version.  Unzip the latest libtiff download, and overwrite c:\libtiff.  Look at the svn entry for the changes I made (i.e. rev 60), and right click each of the files in rev 60 and "compare with working copy" to merge the additions I made in the left file with the right file.  Then build.


* I think this is old.  : 
// * then build:
//
//cd c:\libtiff
//cmake -G "Visual Studio 15 2017 Win64" -DZLIB_LIBRARY:STRING=C:\zlib\zlib-1.2.11\contrib\vstudio\vc14\x64\ZlibStatRelease\zlibstat.lib -DZLIB_INCLUDE_DIR:STRING=C:\zlib\zlib-1.2.11
// * This generates the cmake files and should identify that it found zlib.  Next run :
// 
// cmake --build . --config Release
// ctest -V -C Release

* This should do it instead.

cd c:\libtiff
nmake /f makefile.vc

* You should have new libtiff.lib file in c:\libtiff\libtiff


1.e. Install CUDA SDK (I'm using 10.0). Reboot.

1.f. Install Boost C++ Libraries.
* Download source code : https://www.boost.org/users/download/ into C:\boost folder, and build via:

cd C:\boost\boost_1_69_0
bootstrap
.\b2 address-model=64


2. Generate makefiles:
2.a. Make a subdirectory under where the source code (or this README) is located; let's call it "build"
2.b. From the VS command prompt window, cd into the "build" directory just created and build like this:

cd C:\CudaDecon\build
cmake -D CMAKE_BUILD_TYPE=Release -G "NMake Makefiles" ..


* Make sure there's no error message. To generate makefiles from scratch, the entire content of "build" folder has to be deleted first. 

3. Copy runtime .dlls :
* Copy "libfftw3f-3.dll" from C:\fftw3 into the directory with the cudaDeconv.exe
* Copy "cufft64_100.dll" and "cudart64_100.dll" into the directoy as well :

cp c:\fftw3\libfftw3f-3.dll c:\CudaDecon\build
cp "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin\cudart64_*.dll" c:\CudaDecon\build
cp "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin\cufft64_*.dll"  c:\CudaDecon\build



4. Compile the libraries and executables:

cd C:\CudaDecon\build
nmake 




5. To generate the .sln files for Visual Studio (so that you have a nice IDE to view the source files), you can create a folder, call it "VS", then run this command within the "VS" folder:

cd C:\CudaDecon\VS
cmake .. -G "Visual Studio 15 Wind64"

********************* Notes ***************

* Compatible GPUs are specified in this "C:\CudaDecon\CMakeLists.txt".  This also sets up all of the linking to dependent libraries.  If you end up adding other code libraries, or changing versions, etc you will want to edit this file.  Specifically where you see the lines like : "-gencode=arch=compute_75,code=sm_75"

* GPU based resources have a d_ prefix in their name such as : GPUBuffer & d_interpOTF

* transferConstants() is a function to send small data values from host to GPU device.  
* The link between the function arguments of "transferConstants()" and the globals like : __constant__ unsigned const_nzotf; are found in RLgpuImpl.cu with calls like : cutilSafeCall(cudaMemcpyToSymbol(const_nzotf, &nzotf, sizeof(int)));
 
* RL is based upon the built-in Matlab version : deconvlucy.m (see http://ecco2.jpl.nasa.gov/opendap/hyrax/matlab/images/images/deconvlucy.m)

*  Cudadecon.exe 
Main function is in LinearDecon.cpp. 


* If not enough memory is on the GPU, the program will use host PC's RAM.

* If you are processing on the GPU that drives the display, Windows will terminate cudaDecon if an iteration takes too long.  Set the windows display driver timeout to something larger (like 10 seconds instead of default 5 seconds) :
see http://stackoverflow.com/questions/17186638/modifying-registry-to-increase-gpu-timeout-windows-7
Running this command from an adminstrator command prompt should set the timeout to 10 :
reg.exe ADD "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v "TdrDelay" /t REG_DWORD /D "10" /f


* Better yet, use a second GPU.  The GPU you wish to use for computation only should use the TCC driver (must be a Titan 
	or Tesla or other GPU that supports TCC).  This card should be initialized after the display GPU, so put the compute 
	card in a slot that is > display card.  The TCC driver is selected with NVIDIAsmi.exe -L from an administrator cmd window
	to show the GPUs, then NVIDIAsmi.exe -dm 1 -i 0 to set TCC on GPU 0.  Then use "set CUDA_VISIBLE_DEVICES" to pick 
	the GPU the deconv code should execute on. 