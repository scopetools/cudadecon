#include "linearDecon.h"
#include <exception>
#include <ctime>

#ifdef _WIN32

// Disable silly warnings on some Microsoft VC++ compilers.
#pragma warning(disable : 4244) // Disregard loss of data from float to int.
#pragma warning(disable : 4267) // Disregard loss of data from size_t to unsigned int.
#pragma warning(disable : 4305) // Disregard loss of data from double to float.
#endif

std::string version_number = "0.4.1";
CImg<> next_file_image;

CImg<> ToSave;
CImg<unsigned short> U16ToSave;
CImg<> DeskewedToSave;

int load_next_thread(const char* my_path)
{
  next_file_image.assign(my_path);
  if (false) {
    float img_max = next_file_image(0, 0, 0);
    float img_min = next_file_image(0, 0, 0);
    cimg_forXYZ(next_file_image, x, y, z) {
      img_max = std::max(next_file_image(x, y, z), img_max);
      img_min = std::min(next_file_image(x, y, z), img_min);
    }
    std::cout << "next_file_image : " << next_file_image.width() << " x " << next_file_image.height() << " x " << next_file_image.depth() << ". " << std::endl;
    std::cout << "         next_file_image max, min : " << std::setw(8) << img_max << ", " << std::setw(8) << img_min << std::endl;
    std::cout << "Loaded from " << my_path << std::endl;
  }

  return 0;
}

unsigned compression = 0;

int save_in_thread(std::string inputFileName, const float *const voxel_size, float dz)
{
  std::string temp = "ImageJ=1.50i\n"
    "spacing=" + std::to_string(dz) + "\n"
    "unit=micron";
  const char *description = temp.c_str();
  ToSave.save_tiff(makeOutputFilePath(inputFileName).c_str(), compression, voxel_size, description);

  return 0;
}

int U16save_in_thread(std::string inputFileName, const float *const voxel_size, float dz)
{
  std::string temp = "ImageJ=1.50i\n"
    "spacing=" + std::to_string(dz) + "\n"
    "unit=micron";
  const char *description = temp.c_str();
  U16ToSave.save_tiff(makeOutputFilePath(inputFileName).c_str(), compression, voxel_size, description);

  return 0;
}

int DeSkewsave_in_thread(std::string inputFileName, const float *const voxel_size, const char *const description)
{
  DeskewedToSave.save_tiff(makeOutputFilePath(inputFileName, "Deskewed", "_deskewed").c_str(), compression, voxel_size, description);
  //raw_deskewed.save_tiff(makeOutputFilePath(*it,           "Deskewed", "_deskewed").c_str(), compression, voxel_size, description);
  return 0;
}


std::complex<float> otfinterpolate(std::complex<float> * otf, float kx, float ky, float kz, int nzotf, int nrotf)
// Use sub-pixel coordinates (kx,ky,kz) to linearly interpolate a rotationally-averaged 3D OTF ("otf").
// otf has 2 dimensions: fast dimension is kz with length "nzotf" while the slow dimension is kr.
{
  int irindex, izindex, indices[2][2];
  float krindex, kzindex;
  float ar, az;

  krindex = sqrt(kx*kx + ky*ky);
  kzindex = (kz<0 ? kz+nzotf : kz);

  if (krindex < nrotf-1 && kzindex < nzotf) {
    irindex = floor(krindex);
    izindex = floor(kzindex);

    ar = krindex - irindex;
    az = kzindex - izindex;  // az is always 0 for 2D case, and it'll just become a 1D interp

    if (izindex == nzotf-1) {
      indices[0][0] = irindex*nzotf+izindex;
      indices[0][1] = irindex*nzotf+0;
      indices[1][0] = (irindex+1)*nzotf+izindex;
      indices[1][1] = (irindex+1)*nzotf+0;
    }
    else {
      indices[0][0] = irindex*nzotf+izindex;
      indices[0][1] = irindex*nzotf+(izindex+1);
      indices[1][0] = (irindex+1)*nzotf+izindex;
      indices[1][1] = (irindex+1)*nzotf+(izindex+1);
    }

    return (1-ar)*(otf[indices[0][0]]*(1-az) + otf[indices[0][1]]*az) +
      ar*(otf[indices[1][0]]*(1-az) + otf[indices[1][1]]*az);
  }
  else
    return std::complex<float>(0, 0);
}

int wienerfilter(CImg<> & g, float dkx, float dky, float dkz,
                 CImg<> & otf, float dkr_otf, float dkz_otf,
                 float rcutoff, float wiener)
{
  /* 'g' is the raw data's FFT (half kx axis); it is also the result upon return */
  int i, j, k;
  float kz, ky, kx;
  float amp2, rho, kxscale, kyscale, kzscale, kr;
  std::complex<float> A_star_g, otf_val;
  float w;

  w = wiener*wiener;
  kxscale = dkx/dkr_otf;
  kyscale = dky/dkr_otf;
  kzscale = dkz/dkz_otf;

  int nx = g.width()/2; // '/2' because g is CImg<float> hijacked for complex storage
  int ny = g.height();
  int nz = g.depth();

  std::complex<float> result;

#pragma omp parallel for private(k, i, j, kz, ky, kx, kr, otf_val, amp2, A_star_g, rho, result)
  for (k=0; k<nz; k++) {
    kz = ( k>nz/2 ? k-nz : k );
    for (i=0; i<ny; i++) {
      ky = ( i > ny/2 ? i-ny : i );
      for (j=0; j<nx; j++) {
        kx = j;
        kr = sqrt(kx*kx*dkx*dkx + ky*ky*dky*dky);
        if (kr <=rcutoff) {
          otf_val = otfinterpolate((std::complex<float>*) otf.data(),
                                   kx*kxscale, ky*kyscale,
                                   kz*kzscale, otf.width()/2, otf.height());

          amp2 = otf_val.real() * otf_val.real() + otf_val.imag() * otf_val.imag();
          A_star_g = std::conj(otf_val) * std::complex<float>(g(2*j, i, k), g(2*j+1, i, k));

          /* apodization */
          rho = kr / rcutoff;
          result = A_star_g / (amp2+w) * (1-rho);
          g(2*j, i, k) = result.real();
          g(2*j+1, i, k) = result.imag();
        }
        else {
          g(2*j, i, k) = 0;
          g(2*j+1, i, k) = 0;
        }
      }
    }
  }
  return 0;
}

int main(int argc, char *argv[])
{
#ifdef _WIN32
  HANDLE  hConsole;
  hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

  SetConsoleTextAttribute(hConsole, 11); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
  printf("Created at Howard Hughes Medical Institute Janelia Research Campus. Copyright 2020. All rights reserved.\n");
  SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
#endif
  std::clock_t start_t;
  double duration;
  double iter_duration = 0;


  start_t = std::clock();

  int napodize, nZblend;
  float background;
  float NA=1.2;
  ImgParams imgParams, imgParamsOld;
  float dz_psf, dr_psf;
  float wiener;

  int myGPUdevice=0;
  int RL_iters=0;
  bool bSaveDeskewedRaw = false;
  bool bDontAdjustResolution = false;

  bool BlendTileOverlap = false;
  float deskewAngle=0.0;
  bool bSkewedDecon = false;
  float rotationAngle=0.0;
  bool bNoLimitRatio = false;  // Limit LR ratio in LRcore? use --nlr to turn on
  unsigned outputWidth;
  float padVal = 0.0;
  bool bTwoStepFFT = false; // Use 2-step FFTs (2D-R2C then 1D C2C) to save device mem
  bool bFFTwaInHost = false; // Use host memory for cuFFT work area to save device mem
  int extraShift=0;
  std::vector<int> final_CropTo_boundaries;
  bool bSaveUshort = false;
  std::vector<bool> bDoMaxIntProj;
  std::vector<bool> bDoRawMaxIntProj;
  std::vector< CImg<> > MIprojections;
  int Pad = 0;
  bool bFlatStartGuess = false;
  bool bDoRescale = false;
  bool UseOnlyHostMem = false;
  bool no_overwrite = false;
  bool lzw = false;
  int skip = 0;
  bool bDupRevStack = false;
  int tile_size=0;
  int tile_requested = 0;
  int tile_overlap_requested = 20;

  TIFFSetWarningHandler(NULL);

  std::string datafolder, filenamePattern, otffiles, LSfile;
  po::options_description progopts("cudaDeconv. Version: " + version_number + "\n");
  progopts.add_options()
    ("drdata", po::value<float>(&imgParams.dr)->default_value(.104), "Image x-y pixel size (um)")
    ("dzdata,z", po::value<float>(&imgParams.dz)->default_value(.25), "Image z step (um)")
    ("drpsf", po::value<float>(&dr_psf)->default_value(.104), "PSF x-y pixel size (um)")
    ("dzpsf,Z", po::value<float>(&dz_psf)->default_value(.1), "PSF z step (um)")
    ("wavelength,l", po::value<float>(&imgParams.wave)->default_value(.525), "Emission wavelength (um)")
    ("wiener,W", po::value<float>(&wiener)->default_value(-1.0), "Wiener constant (regularization factor); if this value is postive then do Wiener filter instead of R-L")
    ("background,b", po::value<float>(&background)->default_value(90.f), "User-supplied background")
    ("napodize,e", po::value<int>(&napodize)->default_value(15), "# of pixels to soften edge with")
    ("nzblend,E", po::value<int>(&nZblend)->default_value(0), "# of top and bottom sections to blend in to reduce axial ringing")
    ("dupRevStack,d", po::bool_switch(&bDupRevStack)->default_value(false), "Duplicate reversed stack prior to decon to reduce Z ringing")
    ("NA,n", po::value<float>(&NA)->default_value(1.2), "Numerical aperture")
    ("RL,i", po::value<int>(&RL_iters)->default_value(15), "Run Richardson-Lucy, and set how many iterations")
    ("deskew,D", po::value<float>(&deskewAngle)->default_value(0.0), "Deskew angle; if not 0.0 then perform deskewing before or after deconv")
    ("dcbds", po::bool_switch(&bSkewedDecon)->implicit_value(true),
     "If deskewing, do it after decon; require sample-scan PSF and non-RA OTF")
    ("nlr", po::bool_switch(&bNoLimitRatio)->implicit_value(true),
     "Choose to not use a ratio-limiting step in LRcore")
    ("FFTwaHost,H", po::bool_switch(&bFFTwaInHost)->implicit_value(true),
     "For large dataset, use this to save memory while not sacrificing speed much")
    // ("2stepFT,2", po::bool_switch(&bTwoStepFFT)->implicit_value(true),
     // "For large dataset, use this to save memory while not sacrificing speed much")
    ("padval", po::value<float>(&padVal)->default_value(0.0), "Value to pad image with when deskewing")
    ("width,w", po::value<unsigned>(&outputWidth)->default_value(0), "If deskewed, the output image's width")
    ("shift,x", po::value<int>(&extraShift)->default_value(0), "If deskewed, the output image's extra shift in X (positive->left")
    ("rotate,R", po::value<float>(&rotationAngle)->default_value(0.0), "Rotation angle; if not 0.0 then perform rotation around y axis after deconv")
    ("saveDeskewedRaw,S", po::bool_switch(&bSaveDeskewedRaw)->default_value(false), "Save deskewed raw data to files")
    ("crop,C", fixed_tokens_value< std::vector<int> >(&final_CropTo_boundaries, 6, 6), "Crop final image size to [x1:x2, y1:y2, z1:z2]; takes 6 integers separated by space: x1 x2 y1 y2 z1 z2; ")
    ("MIP,M", fixed_tokens_value< std::vector<bool> >(&bDoMaxIntProj, 3, 3), "Save max-intensity projection along x, y, or z axis; takes 3 binary numbers separated by space: 0 0 1")
    ("rMIP,m", fixed_tokens_value< std::vector<bool> >(&bDoRawMaxIntProj, 3, 3), "Save max-intensity projection of raw deskewed data along x, y, or z axis; takes 3 binary numbers separated by space: 0 0 1")
    ("uint16,u", po::bool_switch(&bSaveUshort)->implicit_value(true), "Save result in uint16 format; should be used only if no actual decon is performed")
    ("input-dir", po::value<std::string>(&datafolder)->required(), "Folder of input images")
    ("otf-file", po::value<std::string>(&otffiles)->required(), "OTF file")
    ("filename-pattern", po::value<std::string>(&filenamePattern)->required(), "File name pattern to find input images to process")
    ("DoNotAdjustResForFFT,a", po::bool_switch(&bDontAdjustResolution)->default_value(false), "Don't change data resolution size. Otherwise data is cropped to perform faster, more memory efficient FFT: size factorable into 2,3,5,7)")
    ("Pad", po::value<int>(&Pad)->default_value(0), "Pad the image data with mirrored values to avoid edge artifacts. Currently only enabled when rotate and deskew are zero.")
    ("LSC", po::value<std::string>(&LSfile), "Lightsheet correction file")
    ("FlatStart", po::bool_switch(&bFlatStartGuess)->default_value(false), "Start the RL from a guess that is a flat image filled with the median image value.  This may supress noise.")
    ("bleachCorrection,p", po::bool_switch(&bDoRescale)->default_value(false), "Apply bleach correction when running multiple images in a single batch")
    ("lzw", po::bool_switch(&lzw)->default_value(false), "Use LZW tiff compression")
    ("skip", po::value<int>(&skip)->default_value(0), "Skip the first 'skip' number of files.")
    ("no_overwrite", po::bool_switch(&no_overwrite)->default_value(false), "Don't reprocess files that are already deconvolved (i.e. exist in the GPUdecon folder).")
    ("tile", po::value<int>(&tile_requested)->default_value(0), "Tile size for tiled decon (in Y only) to attempt to fit into GPU. 0=no tiling. Best to use a power of 2 (i.e. 64, 128, etc).")
    ("TileOverlap", po::value<int>(&tile_overlap_requested)->default_value(30), "Overlap between Tiles.  You want this to be at least ~2x the PSF Y extent.")
    ("BlendTileOverlap", po::bool_switch(&BlendTileOverlap)->default_value(false), "Blend ~5 pixel in Overlap region between Tiles.")
    ("DevQuery,Q", "Show info and indices of available GPUs")
    ("help,h", "This help message.")
    ("version,v", "show version and quit")
    // ("GPUdevice", po::value<int>(&myGPUdevice)->default_value(0), "Index of GPU device to use (0=first device)")
    // ("UseOnlyHostMem", po::bool_switch(&UseOnlyHostMem)->default_value(false), "Just use Host Mapped Memory, and not GPU. For debugging only.")
    ;
  po::positional_options_description p;
  p.add("input-dir", 1);
  p.add("filename-pattern", 1);
  p.add("otf-file", 1);

  std::string commandline_string = __DATE__ ;
  commandline_string.append(" ");
  commandline_string.append(__TIME__);
  for (int i = 0; i < argc; i++) {
    commandline_string.append(" ");
    commandline_string.append(argv[i]);
  } // store commandline_string


    // Parse commandline option:
  po::variables_map varsmap;
  try {
    if (argc == 1)  { //if no arguments, show help.
      std::cout << progopts << "\n";
      return 0;
    }

    store(po::command_line_parser(argc, argv).
          options(progopts).positional(p).run(), varsmap);
    if (varsmap.count("help")) {
      std::cout << progopts << "\n";
      return 0;
    }

    if (varsmap.count("version")) {
      std::cout << version_number << "\n";
      return 0;
    }


    //****************************Query GPU devices***********************************
    if (varsmap.count("DevQuery")) {
      int deviceCount = 0;
      cudaGetDeviceCount(&deviceCount);
      // This function call returns 0 if there are no CUDA capable devices.
      if (deviceCount != 0)
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);

      int dev, driverVersion = 0, runtimeVersion = 0;

      for (dev = 0; dev < deviceCount; ++dev)
        {
          cudaSetDevice(dev);
          cudaDeviceProp mydeviceProp;
          cudaGetDeviceProperties(&mydeviceProp, dev);
          printf("\nDevice %d: \"%s\"\n", dev, mydeviceProp.name);

          cudaDriverGetVersion(&driverVersion);
          cudaRuntimeGetVersion(&runtimeVersion);
          printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
          printf("  CUDA Capability Major/Minor version number:    %d.%d\n", mydeviceProp.major, mydeviceProp.minor);
          printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                 (float)mydeviceProp.totalGlobalMem / 1048576.0f, (unsigned long long) mydeviceProp.totalGlobalMem);
        }
      return 0; // added return because I want query to simply query and quit
    }

    notify(varsmap);

    // if (varsmap.count("crop")) {
    //   if (final_CropTo_boundaries.size() != 6)
    //     throw std::runtime_error("Exactly 6 integers are required for the -C or --crop flag!");
    //   std::copy(final_CropTo_boundaries.begin(), final_CropTo_boundaries.end(), std::ostream_iterator<int>(std::cout, ", "));
    //   std::cout << std::endl;
    // }
    // std::cout << bDoMaxIntProj.size() << std::endl;
  }
  catch (std::exception &e) {
    std::cout << "\n!!Error occurred: " << e.what() << std::endl;
    return 0;
  }

  bool bAdjustResolution = !bDontAdjustResolution;
  imgParamsOld = imgParams; // Copy image Params

  //****************************Check to see if we have a proper GPU device***********************************
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id != cudaSuccess)
    {
      printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
      printf("Result = FAIL\n");
      exit(EXIT_FAILURE);
    }
  if (deviceCount == 0)
    printf("There are no available device(s) that support CUDA\n");


  cudaSetDeviceFlags(cudaDeviceMapHost);
  size_t GPUfree;
  size_t GPUtotal;
  cudaMemGetInfo(&GPUfree, &GPUtotal);
  cudaDeviceProp mydeviceProp;
  cudaGetDeviceProperties(&mydeviceProp, myGPUdevice);

#ifdef _WIN32
  SetConsoleTextAttribute(hConsole, 13); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
#endif

  std::cout << std::endl << "Built : " << __DATE__ << " " << __TIME__ << ".  GPU " << (GPUfree>>20) << " MB free / " << (GPUtotal>>20) << " MB total on " << mydeviceProp.name << std::endl;

  CImg<> raw_image, raw_imageFFT, complexOTF, raw_deskewed, LSImage, sub_image, file_image, stitch_image, blend_weight_image;
  CImg<float> AverageLS;
  float dkx_otf, dky_otf, dkz_otf;
  float dkx, dky, dkz, rdistcutoff;
  fftwf_plan rfftplan=NULL, rfftplan_inv=NULL;
  CPUBuffer rotMatrix;
  double deskewFactor=0;
  bool bCrop = false;
  unsigned new_ny, new_nz, new_nx;
  int deskewedXdim = 0;
  cufftHandle rfftplanGPU = NULL, rfftplanInvGPU = NULL;
  GPUBuffer workArea(myGPUdevice, bFFTwaInHost);
  GPUBuffer d_interpOTF(myGPUdevice, UseOnlyHostMem);
  GPUBuffer d_rawOTF(myGPUdevice, UseOnlyHostMem);
  // why were the above GPUBuffer()'s called with "1" as the first parameter ("size")??

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, myGPUdevice);

  unsigned nx, ny, nz = 0;

  int border_x = 0;
  int border_y = 0;
  int border_z = 0;

  float voxel_size [3];
  float voxel_size_decon [3];
  float imdz;
  const char *description;

  if (lzw) {
    compression = 1;
  }
  //****************************Main processing***********************************
  // Loop over all matching input TIFFs, :
  try {


    std::cout << "Looking for files to process... " ;
    // Gather all files in 'datafolder' and matching the file name pattern:
    //bool MIPsOnly = (RL_iters == 0 && bDoMaxIntProj.size() == 3);

    std::vector< std::string > all_matching_files = gatherMatchingFiles(datafolder, filenamePattern, no_overwrite);
    std::cout << "Found " << all_matching_files.size() << " file(s)." << std::endl ;

#ifdef _WIN32
    SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
#endif
    if (all_matching_files.size() == 0) {
#ifdef _WIN32
      SetConsoleTextAttribute(hConsole, 10); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
#endif
      std::cout<< "\nNo files need processing!" << std::endl;
#ifdef _WIN32
      SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
#endif
      return 0;
    }


    std::vector<std::string>::iterator next_it = all_matching_files.begin() + skip;//make a second incrementer that will be used to load the next raw image while we process.

    std::thread t1; // thread for loading files
    t1 = std::thread(load_next_thread, next_it->c_str());   //start loading the first file.

    std::thread tsave;  // thread for saving decon
    std::thread tDeskewsave; // thread for saving deskewed

    for (std::vector<std::string>::iterator it= all_matching_files.begin() + skip;
         it != all_matching_files.end(); it++) {

      int number_of_files_left = all_matching_files.end() - it;

#ifdef _WIN32
      SetConsoleTextAttribute(hConsole, 10); // colors are 9=blue 10=green and so on to 15=white
#endif
      std::cout << std::endl << "Loading raw_image: " << it - all_matching_files.begin() + 1 << " out of " << all_matching_files.size() << ".   ";
      if (it > all_matching_files.begin() + skip) // if this isn't the first iteration.
        {
          int seconds = number_of_files_left * iter_duration;
          int hours = ceil(seconds / (60 * 60));
          int minutes = ceil((seconds - (hours * 60 * 60)) / 60);
          std::cout << (int)iter_duration << " s/file.   " << number_of_files_left << " files left.  " << hours << " hours, " <<  minutes << " minutes remaining.";
        }

      std::cout <<  std::endl;
#ifdef _WIN32
      SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
#endif

      std::cout << std::endl << *it << std::endl;
      std::cout << "Waiting for separate thread to finish loading image... " ;
      t1.join();        // wait for loading thread to finish reading next_raw_image into memory.
      t1.~thread();     // destroy thread.
      std::cout << "Image loaded. Copying to 'file_image'... " ;
      file_image.assign(next_file_image, false); // Copy to file_image.  CImg<T>& assign(const CImg<t>& img, const bool is_shared)

      float img_max = file_image(0, 0);
      float img_min = file_image(0, 0);

#ifndef NDEBUG
      cimg_forXYZ(file_image, x, y, z) {
        img_max = std::max(file_image(x, y, z), img_max);
        img_min = std::min(file_image(x, y, z), img_min);
      }

      std::cout << "         raw img max, min : " << std::setw(8) << img_max << ", " << std::setw(8) << img_min << std::endl;
#endif



      std::cout << "Done." << std::endl;

      // start reading the next image
      next_it++; // increment next_it.  This will now have the next file to read.
      if (it < all_matching_files.end() - 1)  // If there are more files to process...
        t1 = std::thread(load_next_thread, next_it->c_str());       //start new thread and load the next file, while we process file_image

      //**************************** Tiling setup ***********************************
      int number_of_tiles = 1;
      int number_of_overlaps = 0;
      double tile_overlap = 0;


      if (tile_requested < 0) {
        tile_size = std::min(           128, file_image.height());
      }
      if (tile_requested > 0) {
        tile_size = std::min(tile_requested, file_image.height());
      }

      number_of_tiles = ceil((double)file_image.height() / ((double)tile_size - tile_overlap_requested)); // try for 20 pixel overlap
      number_of_overlaps = number_of_tiles - 1;
      tile_overlap = (((double)tile_size * number_of_tiles) - file_image.height()) / number_of_overlaps;
      tile_overlap = floor(tile_overlap); //


      if (tile_requested == 0) {
        number_of_tiles = 1;
        tile_size = file_image.height();
        tile_overlap = 0;
        number_of_overlaps = 0;
      }
      std::cout << std::endl << "# of tiles: " << number_of_tiles << ". # of overlaps: " << number_of_overlaps << ". Tile size: " << tile_size << " Y pix. Overlap: " << tile_overlap << " Y pix. " << std::endl;

      for (int tile_index = 0; tile_index < number_of_tiles; tile_index++) {
        int tile_y_offset = floor( (double)tile_index*(tile_size - tile_overlap) );
        //std::cout << std::endl << "tile_y_offset: " << tile_y_offset;

        if (number_of_tiles > 1) {

          //int y_end = std::min(file_image.height(), tile_size + tile_y_offset);
          int y_end = tile_size + tile_y_offset; // I think we can crop to outside the image, and the boundry conditions will fill it. This way each subimage is exactly the same size, which makes the rest of the code mighty happy.


          raw_image = file_image.get_crop(0,  /* X start */
                                          tile_y_offset,                  /* Y start */
                                          0,                              /* Z start */
                                          file_image.width() - 1,         /*  X end */
                                          y_end - 1,                      /*  Y end */
                                          file_image.depth() - 1,         /*  Z end */
                                          true); // get sub_image. raw_image = sub_image


          //for (int y = tile_y_offset; y < y_end; ++y) {

          //cimg_forXZ(file_image, x, z) {
          //raw_image(x, y_raw_index, z) = (unsigned long) file_image(x, y, z);
          //}
          //y_raw_index = y_raw_index + 1;
          //}
          //raw_image = file_image.get_crop(0,
          //  tile_y_offset,
          //  0,
          //  0,
          //  file_image.width() - 1,
          //  std::min(file_image.height(), tile_size + tile_y_offset) - 1,
          //  file_image.depth() - 1,
          //  0); // get sub_image. raw_image = sub_image
#ifdef _WIN32
          SetConsoleTextAttribute(hConsole, 10); // colors are 9=blue 10=green and so on to 15=white
#endif
          std::cout << std::endl << "Tile: " << tile_index + 1 << " out of " << number_of_tiles << ".   " << std::endl;
#ifdef _WIN32
          SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
#endif
          // raw_image.save(makeOutputFilePath(it->c_str(), "tile", "_tile").c_str()); //for debugging
        }
        else
          raw_image.assign(file_image);

        // If it's the first input file, initialize a bunch including:
        // 1. crop image to make dimensions nice factorizable numbers
        // 2. calculate deskew parameters, new X dimensions
        // 3. calculate rotation matrix
        // 4. create FFT plans
        // 5. transfer constants into GPU device constant memory
        // 6. make 3D OTF array in device memory

        bool bDifferent_sized_raw_img = (nx != raw_image.width() || ny != raw_image.height() || nz != raw_image.depth() ); // Check if raw.image has changed size from first iteration

        if (it != all_matching_files.begin() + skip && bDifferent_sized_raw_img) {
#ifdef _WIN32
          SetConsoleTextAttribute(hConsole, 14); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
#endif
          std::cout << std::endl << "File " << it - all_matching_files.begin() + 1 << " has a different size" << std::endl;
        }
        std::cout << "raw_image size             : " << raw_image.width() << " x " << raw_image.height() << " x " << raw_image.depth() << std::endl;
#ifdef _WIN32
        SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
#endif


        //******************* If first image OR if raw_img has changed size *****************
        if (nx == 0 || bDifferent_sized_raw_img ) {
          nx = raw_image.width();
          ny = raw_image.height();
          nz = raw_image.depth();
          imgParams = imgParamsOld;



          //****************************Adjust resolution if desired***********************************

          int step_size;

          if (fabs(deskewAngle) > 0.0 || fabs(rotationAngle) > 0.0)
            {
              Pad = 0;    // Currently padding is disabled if we are deskewing or rotating.
              std::cout << "Currently padding is disabled if we are deskewing or rotating." << std::endl;
            }

          if (Pad)
            step_size =  1;
          else
            step_size = -1;

          int startnx = nx;
          int startny = ny;
          int startnz = nz;

          if (Pad) { //pad by N number of border pixels
            startnx = startnx + Pad * 2;
            startny = startny + Pad * 2;
            startnz = startnz + Pad * 2;
            std::cout << "Min padding to use is " << Pad << " pixels." << std::endl;
          }

          if (RL_iters > 0 && (bAdjustResolution || Pad)) {
            new_ny = findOptimalDimension(startny, step_size);
            if (new_ny != startny) {
              printf("new ny=%d\n", new_ny);
              bCrop = true;
            }

            new_nz = findOptimalDimension(startnz, step_size);
            if (new_nz != startnz) {
              printf("new nz=%d\n", new_nz);
              bCrop = true;
            }

            // only if no deskewing is happening before decon do we want to change image width here
            new_nx = startnx;
            if (!(fabs(deskewAngle) > 0.0) || bSkewedDecon) {
              new_nx = findOptimalDimension(startnx, step_size);
              if (new_nx%2 && new_nx>1000) // Lin: for unknown reason, an odd X dimension
                // sometimes results in doubled cuFFT work size; to-do...
                new_nx = findOptimalDimension(new_nx-1, step_size);
              if (new_nx != startnx) {
                printf("new nx=%d\n", new_nx);
                bCrop = true;
              }
            }
          }
          else {
            new_nx = nx;
            new_ny = ny;
            new_nz = nz;
          }


          //****************************Load OTF to CPU RAM(assuming 3D rotationally averaged OTF)***********************************
          std::cout <<  "Loading OTF... ";
          complexOTF.assign(otffiles.c_str());
          unsigned nx_otf, ny_otf, nz_otf;

          if (bSkewedDecon && fabs(deskewAngle) > 0.0)
            dz_psf *= fabs(sin(deskewAngle * M_PI/180.));

          determine_OTF_dimensions(complexOTF, dr_psf, dz_psf,
                                   nx_otf, ny_otf, nz_otf,
                                   dkx_otf, dky_otf, dkz_otf);
          std::cout << "nx x ny x nz     : " << nx_otf << " x " << ny_otf << " x " << nz_otf << ". " << std::endl << std::endl ;
          // transfer raw OTF into device memory; texture TO-DO
          d_rawOTF.resize(nx_otf * ny_otf * nz_otf * sizeof(cuFloatComplex));
          cutilSafeCall(cudaMemcpy(d_rawOTF.getPtr(),complexOTF.data(),
                                   d_rawOTF.getSize(), cudaMemcpyDefault));

          //****************************Construct deskew matrix***********************************
          deskewedXdim = new_nx;
          if (fabs(deskewAngle) > 0.0) {
            float old_dz = imgParams.dz;
            if (deskewAngle <0) deskewAngle += 180.;
            deskewFactor = cos(deskewAngle * M_PI/180.) * imgParams.dz / imgParams.dr;
            if (outputWidth ==0)
              deskewedXdim += floor(new_nz * imgParams.dz *
                                    fabs(cos(deskewAngle * M_PI/180.)) / imgParams.dr);
            // TODO: sometimes deskewedXdim calc'ed this way is too large
            else
              deskewedXdim = outputWidth; // use user-provided output width if available

            // Adjust resolution to optimal FFT size if desired
            if (bAdjustResolution && !bSkewedDecon)
              deskewedXdim = findOptimalDimension(deskewedXdim);

            // update z step size:
            imgParams.dz *= fabs(sin(deskewAngle * M_PI/180.));
            if (fabs(deskewFactor) < 1) {
#ifdef _WIN32
              SetConsoleTextAttribute(hConsole, 14); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
#endif
              printf("Warning : deskewFactor is < 1.  Check that angle, dz, and dr sizes are correct.\n");
#ifdef _WIN32
              SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
#endif
            }
            printf("old dz = %f, dxy = %f, deskewFactor = %f, new nx = %d, new dz = %f\n", old_dz, imgParams.dr, deskewFactor, deskewedXdim, imgParams.dz);

          }



          //****************************Construct rotation matrix***********************************
          if (fabs(rotationAngle) > 0.0) {
            rotMatrix.resize(4*sizeof(float));
            rotationAngle *= M_PI/180;
            float stretch = imgParams.dr / imgParams.dz;
            float *p = (float *)rotMatrix.getPtr();
            p[0] = cos(rotationAngle) * stretch;
            p[1] = sin(rotationAngle) * stretch;
            p[2] = -sin(rotationAngle);
            p[3] = cos(rotationAngle);
          }



          if (wiener >0) {
            //****************Wiener filter instead of RL (not usually used.):*****************
            raw_imageFFT.assign(deskewedXdim+2, new_ny, new_nz);

            if (!fftwf_init_threads()) { /* one-time initialization required to use threads */
              printf("Error returned by fftwf_init_threads()\n");
            }

            fftwf_plan_with_nthreads(8);
            rfftplan = fftwf_plan_dft_r2c_3d(new_nz, new_ny, deskewedXdim,
                                             raw_image.data(),
                                             (fftwf_complex *) raw_imageFFT.data(),
                                             FFTW_ESTIMATE);
            rfftplan_inv = fftwf_plan_dft_c2r_3d(new_nz, new_ny, deskewedXdim,
                                                 (fftwf_complex *) raw_imageFFT.data(),
                                                 raw_image.data(),
                                                 FFTW_ESTIMATE);
          }
          else if (RL_iters>0) {

            //****************************RL decon***********************************

            //****************************Create reusable cuFFT plans************************
            //
            size_t workSize = 0;
            cufftResult cuFFTErr;

            if (cufftGetSize(rfftplanGPU, &workSize) == CUFFT_SUCCESS) { // if plan existed before, destroy it before creating a new plan.
              cufftDestroy(rfftplanGPU);
              std::cout << "Destroying rfftplanGPU." << std::endl;
            }
            if (cufftGetSize(rfftplanInvGPU, &workSize) == CUFFT_SUCCESS) { // if plan existed before, destroy it before creating a new plan.
              cufftDestroy(rfftplanInvGPU);
              std::cout << "Destroying rfftplanInvGPU." << std::endl;
            }

            size_t GPUfree_prev;
            cudaMemGetInfo(&GPUfree_prev, &GPUtotal);

            unsigned nz_final;
            if (bDupRevStack)  // to reduce severe Z ringing
              nz_final = new_nz*2;
            else
              nz_final = new_nz;

            unsigned xdim4FFT = deskewedXdim;
            if (bSkewedDecon) xdim4FFT = new_nx;

            if (bTwoStepFFT) {
              // Perform first a series of 2D R2C and then 1D C2C FFTs.
              // Based on comparison, this reduces workspace size by half, while
              // not affecting speed by much, if any.
              int fftN = nz_final;
              int strideC = new_ny * (xdim4FFT/2 + 1);
              cuFFTErr = cufftCreate(&rfftplanGPU);
              // cuFFTErr = cufftSetAutoAllocation(rfftplanGPU, 0); // turn off auto allocation
              cuFFTErr = cufftMakePlanMany(rfftplanGPU, 1, &fftN,
                                           &fftN, strideC, 1,
                                           &fftN, strideC, 1,
                                           CUFFT_C2C,  // specify C2C type
                                           strideC, &workSize);
              // As a bonus, this C2C plan can be shared between foward and inverse;
              // just need to specify direction when calling cufftExecC2C.
              // cuFFTErr = cufftSetWorkArea(rfftplanGPU, workArea.getPtr());
              if (cuFFTErr != CUFFT_SUCCESS) {
                cudaMemGetInfo(&GPUfree, &GPUtotal);
                std::cerr << "GPU " << (GPUfree>>20) << " MB free / " << (GPUtotal>>20)
                          << " MB total. " << std::endl;

                cufftEstimate3d(nz_final, new_ny, xdim4FFT, CUFFT_R2C, &workSize);
                std::cerr << "R2C FFT Plan desires " << (workSize>>20) << " MB. " << std::endl;
                throw std::runtime_error("cufftPlan3d() r2c failed.");
              }
              // In two-step FFT mode, rfftplanInvGPU is left unallocated and equal to NULL.
              // The only other plans needed are 2D R2C and C2R plans, which use tiny work
              // space and can be allocated and de-allocated inside RichardsonLucy_GPU()
            }

            else {
              // new way, share FFTplan workarea.
              // Possibly put this area on the Host RAM if it doesn't fit in GPU RAM.
              int autoAllocate   = 0; // don't allocate, we want to specify the workspace ourselves.
              size_t workSizeR2C = 0; // size of R2C plan to be filled in by cuFFT
              size_t workSizeC2R = 0; // size of C2R plan to be filled in by cuFFT
              workSize    = 0; // necessary size of plan to enable R2C or C2R (i.e. max of these)

              cuFFTErr = cufftCreate(&rfftplanGPU);                           // create object.
              cuFFTErr = cufftSetAutoAllocation(rfftplanGPU, autoAllocate);

              cuFFTErr = cufftMakePlan3d(rfftplanGPU, nz_final, new_ny, xdim4FFT, CUFFT_R2C, &workSizeR2C);   // make plan, retrieve needed workSize
              if (cuFFTErr != CUFFT_SUCCESS) {
                std::cerr << "cufftMakePlan3d() r2c failed. Error code: " << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;
                cudaMemGetInfo(&GPUfree, &GPUtotal);
                std::cerr << "GPU " << (GPUfree>>20) << " MB free / " << (GPUtotal>>20) << " MB total. " << std::endl;

                cuFFTErr = cufftEstimate3d(new_nz, new_ny, xdim4FFT, CUFFT_R2C, &workSizeR2C);
                if (cuFFTErr != CUFFT_SUCCESS)
                  std::cerr << "cufftEstimate3d() failed. " << new_nz << " x " << new_ny << " x " << xdim4FFT << " Error code: " << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;

                std::cerr << "R2C FFT Plan desires " << (workSizeR2C>>20) << " MB. " << std::endl;
                throw std::runtime_error("cufftMakePlan3d() r2c failed.");
              }

              cuFFTErr = cufftCreate(&rfftplanInvGPU);                        // create object.
              if (cuFFTErr != CUFFT_SUCCESS) {
                std::cerr << "cufftCreate(&rfftplanInvGPU) failed. Error code: " << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;
                throw std::runtime_error("cufftCreate() rfftplanInvGPU failed.");
              }

              cuFFTErr = cufftSetAutoAllocation(rfftplanInvGPU, autoAllocate);
              if (cuFFTErr != CUFFT_SUCCESS) {
                std::cerr << "cufftSetAutoAllocation(rfftplanInvGPU, autoAllocate) failed. Error code: " << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;
                throw std::runtime_error("cufftSetAutoAllocation(rfftplanInvGPU, autoAllocate) failed.");
              }

              cuFFTErr = cufftMakePlan3d(rfftplanInvGPU, nz_final/*new_nz??*/, new_ny, xdim4FFT, CUFFT_C2R, &workSizeC2R);// make plan, get workSize
              if (cuFFTErr != CUFFT_SUCCESS) {
                std::cerr <<"cufftMakePlan3d() c2r failed. " << nz_final << " x " << new_ny << " x " << xdim4FFT << " Error code: " << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;
                cudaMemGetInfo(&GPUfree, &GPUtotal);
                std::cerr << "GPU " << (GPUfree>>20) << " MB free / " << (GPUtotal>>20) << " MB total. " << std::endl;

                cuFFTErr = cufftEstimate3d(new_nz, new_ny, xdim4FFT, CUFFT_C2R, &workSizeC2R);
                if (cuFFTErr != CUFFT_SUCCESS)
                  std::cerr << "cufftEstimate3d() failed. " << new_nz << " x " << new_ny << " x " << xdim4FFT << " Error code: " << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;
                std::cerr << "C2R FFT Plan desires " << (workSizeC2R>>20) << " MB. " << std::endl;
                throw std::runtime_error("cufftMakePlan3d() c2r failed.");
              }

              // set workSize to max of C2R and R2C plans.
              workSize = std::max(workSizeC2R, workSizeR2C);

              if (workArea.getSize() != workSize) {  // do we need to (re)allocate workArea?
                workArea.resize(workSize);
                std::cout << "FFTplan workarea allctd.   (";
                cudaMemGetInfo(&GPUfree, &GPUtotal);
                std::cout << std::setw(4) << (workArea.getSize()>>20) << "MB)"
                          << std::setw(7) << (GPUfree>>20) << "MB free " << std::endl;
              }

              cuFFTErr = cufftSetWorkArea(rfftplanGPU, workArea.getPtr());
              if (cuFFTErr != CUFFT_SUCCESS) {
                std::cerr << "cufftSetWorkArea() r2c failed. Error code: "
                          << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;
                cudaMemGetInfo(&GPUfree, &GPUtotal);
                std::cerr << "GPU " << (GPUfree>>20) << " MB free / " << (GPUtotal>>20) << " MB total. " << std::endl;

                cufftEstimate3d(new_nz, new_ny, xdim4FFT, CUFFT_R2C, &workSizeR2C);
                std::cerr << "R2C FFT Plan desires " << (workSizeR2C>>20) << " MB. " << std::endl;
                throw std::runtime_error("cufftSetWorkArea() r2c failed.");
              }

              cuFFTErr = cufftSetWorkArea(rfftplanInvGPU, workArea.getPtr());
              if (cuFFTErr != CUFFT_SUCCESS) {
                std::cerr << "cufftSetWorkArea() c2r failed. Error code: " << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;
                cudaMemGetInfo(&GPUfree, &GPUtotal);
                std::cerr << "GPU " << (GPUfree>>20) << " MB free / " << (GPUtotal>>20) << " MB total. " << std::endl;

                cufftEstimate3d(new_nz, new_ny, xdim4FFT, CUFFT_C2R, &workSizeC2R);
                std::cerr << "C2R FFT Plan desires " << (workSizeC2R>>20) << " MB. " << std::endl;
                throw std::runtime_error("cufftSetWorkArea() c2r failed.");
              }

            } // end FFT plan allocation method (either normal way or "shared" plan way)

            std::cout << "FFT plans allocated.    ";
            cudaMemGetInfo(&GPUfree, &GPUtotal);
            std::cout << std::setw(8) << ((GPUfree_prev - GPUfree)>>20) << "MB"
                      << std::setw(8) << (GPUfree>>20) << "MB free"
                      << " nz=" << new_nz << ", ny=" << new_ny << ", nx=" << xdim4FFT
                      << std::endl;

          }  //else if (RL_iters>0)

          //*************Transfer a bunch of constants to device, including OTF array:**********

          unsigned xdim_during_decon = deskewedXdim;
          if (bSkewedDecon)
            xdim_during_decon = new_nx;
          dkx = 1.0/(imgParams.dr * xdim_during_decon);
          dky = 1.0/(imgParams.dr * new_ny);

          unsigned nz_const;   // why not keep using "nz_final"??
          if (bDupRevStack)
            nz_const = new_nz*2;
          else
            nz_const = new_nz;
          dkz = 1.0/(imgParams.dz * nz_const);

          rdistcutoff = 2*NA/(imgParams.wave); // lateral resolution limit in 1/um
          float eps = std::numeric_limits<float>::epsilon();

          // if (bSkewedDecon) {
          //   transferConstants(new_nx, new_ny, nz_const,
          //                     nx_otf, ny_otf, nz_otf,
          //                     dkx/dkx_otf, dky/dky_otf, dkz/dkz_otf,
          //                     bNoLimitRatio, eps);
          //   d_interpOTF.resize(nz_const * new_ny * (new_nx+2) * sizeof(float)); // allocate memory
          //   makeOTFarray(d_rawOTF, d_interpOTF, new_nx, new_ny, nz_const); // interpolate
          // }
          // else {
          transferConstants(xdim_during_decon, new_ny, nz_const,
                            nx_otf, ny_otf, nz_otf,
                            dkx/dkx_otf, dky/dky_otf, dkz/dkz_otf,
                            bNoLimitRatio, eps);

          d_interpOTF.resize(nz_const * new_ny * (xdim_during_decon/2+1)*2 * sizeof(float)); // allocate memory
          makeOTFarray(d_rawOTF, d_interpOTF, xdim_during_decon, new_ny, nz_const); // interpolate.
          // }
          d_rawOTF.resize(0);
          std::cout << "d_interpOTF allocated.  ";
          cudaMemGetInfo(&GPUfree, &GPUtotal);
          std::cout << std::setw(8) << (d_interpOTF.getSize()>>20) << "MB" << std::setw(8) << (GPUfree>>20) << "MB free" << std::endl;

          //****************************Prepare Light Sheet Correction if desired ***********************

          if (LSfile.size()) {

            const float my_min = 0.2;

            std::cout << std::endl << "Loading LS Correction      : ";
            LSImage.assign(LSfile.c_str());
            std::cout << LSImage.width() << " x " << LSImage.height() << " x " << LSImage.depth() << ". " << std::endl;
            AverageLS.resize(LSImage.width(), LSImage.height(), 1, 1, -1); //Set size of AverageLS
            AverageLS.fill(0); //fill with zeros

            cimg_forXYZ(LSImage, x, y, z) {
              AverageLS(x, y) = AverageLS(x, y) + LSImage(x, y, z); //sum image

              if (z == (LSImage.depth() - 1)) // if this is the last Z slice
                AverageLS(x, y) = AverageLS(x, y) / LSImage.depth(); //divide by number of slices
            }

            const int LSIx = AverageLS.width();
            const int LSIy = AverageLS.height();

            int LSIborderx = (LSIx - nx) / 2;
            int LSIbordery = (LSIy - ny) / 2;

            std::cout << "Excess LS border to crop   : " << LSIborderx << " x " << LSIbordery << std::endl;

            AverageLS.crop(LSIborderx, LSIbordery, LSIx - LSIborderx - 1, LSIy - LSIbordery - 1); //crop it

            cimg_forXY(AverageLS, x, y)
              AverageLS(x, y) = AverageLS(x, y) - background;


            img_max = AverageLS(0, 0);
            img_min = AverageLS(0, 0);
            cimg_forXY(AverageLS, x, y) {
              img_max = std::max(AverageLS(x, y), img_max);
              img_min = std::min(AverageLS(x, y), img_min);
            }
            std::cout << "               LS max, min : " << std::setw(8) << img_max << ", " << std::setw(8) << img_min << std::endl;

            const float normalization = img_max;

            img_max = -9999999;
            img_min = 9999999;
            cimg_forXY(AverageLS, x, y) {
              AverageLS(x, y) = AverageLS(x, y) / normalization;
              img_max = std::max(AverageLS(x, y), img_max);
              img_min = std::min(AverageLS(x, y), img_min);
            }
            std::cout << "       After normalization : " << std::setw(8) << img_max << ", " << std::setw(8) << img_min << std::endl;


            AverageLS.max(my_min); // set everything below 0.2 to 0.2.  This will prevent divide by zero.
          } // end if (LSfile.size())

        } // end if (nx == 0 || bDifferent_sized_raw_img )

        //******************** *Apply Light Sheet Correction if desired *******************
        if (LSfile.size()) {
          CImg<float> Temp(raw_image); // Copy raw_image to a floating point image

          cimg_forXYZ(Temp, x, y, z)
            Temp(x, y, z) = Temp(x, y, z) - background; // subtract background. min value =0.

          Temp.max((float)0); // set min value to 0.  have to cast zero to avoid compiler complaint.

          if (it == all_matching_files.begin()) {
            img_max = AverageLS(0, 0);
            img_min = AverageLS(0, 0);

            cimg_forXY(AverageLS, x, y) {
              img_max = std::max(AverageLS(x, y), img_max);
              img_min = std::min(AverageLS(x, y), img_min);
            }
            std::cout << "LSC size                   : " << AverageLS.width() << " x " << AverageLS.height() << " x " << AverageLS.depth() << std::endl;
            std::cout << "LSC max, min               : " << img_max << ", " << img_min << std::endl;
          }

          std::cout << "Dividing by LSC... ";

          Temp.div(AverageLS);
          std::cout << " Done." << std::endl;

          cimg_forXYZ(Temp, x, y, z)
            raw_image(x, y, z) = Temp(x, y, z) + background; //replace background and copy back to U16 raw_image.
        } // end if applying LS correction

        voxel_size[0] = imgParams.dr;
        voxel_size[1] = imgParams.dr;
        voxel_size[2] = imgParams.dz;
        imdz = imgParams.dz;
        if (rotMatrix.getSize()) {
          imdz = imgParams.dr;
        }
        voxel_size_decon[0] = imgParams.dr;
        voxel_size_decon[1] = imgParams.dr;
        voxel_size_decon[2] = imdz;
        std::string s = "ImageJ=1.50i\n"
          "spacing=" + std::to_string(imgParams.dz) + "\n"
          "unit=micron";
        description = s.c_str();

        //****************************Pad image.  Use Mirror image in padded border region***********************************

        if (Pad) {
          border_x = (new_nx - nx) / 2;   // get border size
          border_y = (new_ny - ny) / 2;
          border_z = (new_nz - nz) / 2;

          std::cout << std::endl <<  "Create padded img.  Border : " << border_x << " x " << border_y << " x " << border_z << ". " << std::endl;
          CImg<> raw_original(raw_image);       //copy from raw image
          std::cout << "Image with padding size    : " << new_nx << " x " << new_ny << " x " << new_nz << ". ";
          raw_image.resize(new_nx, new_ny, new_nz);     //resize with border

          int x_raw;
          int y_raw;
          int z_raw;

          int i_nx = (int)nx;
          int i_ny = (int)ny;
          int i_nz = (int)nz;

          // std::cout << "Line:" << __LINE__ << std::endl;
          std::cout << "Copy values... ";
          cimg_forXYZ(raw_image, x, y, z) // for every pixel in the new image, copy value from original image
            {
              x_raw = abs(x - border_x);
              y_raw = abs(y - border_y);
              z_raw = abs(z - border_z);

              if (x_raw >= i_nx)
                x_raw = i_nx - (x_raw - i_nx) - 1;

              if (y_raw >= i_ny)
                y_raw = i_ny - (y_raw - i_ny) - 1;

              if (z_raw >= i_nz)
                z_raw = i_nz - (z_raw - i_nz) - 1;

              //raw_image(x, y, z) = x_raw;
              raw_image(x, y, z) = raw_original(x_raw, y_raw, z_raw);
            }
          //***debug padded image.
          if (false) {
            std::cout << "Saving padded image... " << std::endl;
            makeNewDir("Padded");
            CImg<unsigned short> uint16Img(raw_image);
            uint16Img.save_tiff(makeOutputFilePath(*it, "Padded", "_padded").c_str(), 0, voxel_size, description);
          }
          std::cout << "Done." << std::endl;
        } // End Pad image creation.

        // moved here from boostfs.cpp in order to make it conditional on actually performing decon
        // this MAY cause some bugs ... but haven't seen any since adding a long time ago
        if (RL_iters || rotMatrix.getSize()) {
          if (it == all_matching_files.begin()) //make directory only on first call, and if we are saving Deskewed.
            makeNewDir("GPUdecon");
        }

        // initialize the raw_deskewed size everytime in case it is cropped on an earlier iteration
        if (bSaveDeskewedRaw && (fabs(deskewAngle) > 0.0) ) {
          raw_deskewed.assign(deskewedXdim, new_ny, new_nz);

          if (it == all_matching_files.begin() + skip) //make directory only on first call, and if we are saving Deskewed.
            makeNewDir("Deskewed");
        }

        // If deskew is to happen, it'll be performed inside RichardsonLucy_GPU() on GPU;
        // but here raw data's x dimension is still just "new_nx"
        if (bCrop)
          // raw_image.crop(0, 0, 0, 0, new_nx-1, new_ny-1, new_nz-1, 0);
          raw_image.crop((nx-new_nx)/2, (ny-new_ny)/2, (nz-new_nz)/2,
                         (nx+new_nx)/2-1, (ny+new_ny)/2-1, (nz+new_nz)/2-1);

        if (wiener >0) { // plain 1-step Wiener filtering
          raw_image -= background;
          fftwf_execute_dft_r2c(rfftplan, raw_image.data(), (fftwf_complex *) raw_imageFFT.data());

          wienerfilter(raw_imageFFT,
                       dkx, dky, dkz,
                       complexOTF,
                       dkx_otf, dkz_otf, // keep wiener filter stuff??
                       rdistcutoff, wiener);

          fftwf_execute_dft_c2r(rfftplan_inv, (fftwf_complex *) raw_imageFFT.data(), raw_image.data());
          raw_image /= raw_image.size();
        }
        else if (RL_iters || raw_deskewed.size() || rotMatrix.getSize()) {
#ifndef NDEBUG
          img_max = raw_image(0, 0, 0);
          img_min = raw_image(0, 0, 0);
          cimg_forXYZ(raw_image, x, y, z) {
            img_max = std::max(raw_image(x, y, z), img_max);
            img_min = std::min(raw_image(x, y, z), img_min);
          }
          std::cout << "RL_image : " << raw_image.width() << " x " << raw_image.height() << " x " << raw_image.depth() << ". " << std::endl;
          std::cout << "         RL_image max, min : " << std::setw(8) << img_max << ", " << std::setw(8) << img_min << std::endl;
#endif
          float my_median = 1;
          if (bFlatStartGuess)
            my_median = raw_image.median();

#ifdef USE_NVTX
          cudaProfilerStart();
#endif

          //************************************************************************************
          //****************************Run RL GPU**********************************************
          //************************************************************************************
          RichardsonLucy_GPU(raw_image, background, d_interpOTF, RL_iters, deskewFactor,
                             deskewedXdim, extraShift, napodize, nZblend, rotMatrix,
                             rfftplanGPU, rfftplanInvGPU, raw_deskewed, &deviceProp,
                             bFlatStartGuess, my_median, bDoRescale, bSkewedDecon,
                             padVal, bDupRevStack,
                             UseOnlyHostMem, myGPUdevice);
#ifdef USE_NVTX
          cudaProfilerStop();
#endif
        }
        else {
          std::cerr << "Nothing is performed\n";
        }

#ifndef NDEBUG
        img_max = raw_image(0, 0, 0);
        img_min = raw_image(0, 0, 0);
        cimg_forXYZ(raw_image, x, y, z) {
          img_max = std::max(raw_image(x, y, z), img_max);
          img_min = std::min(raw_image(x, y, z), img_min);
        }
        std::cout << "output_image : " << raw_image.width() << " x " << raw_image.height() << " x " << raw_image.depth() << ". " << std::endl;
        std::cout << "         output_image max, min : " << std::setw(8) << img_max << ", " << std::setw(8) << img_min << std::endl;
#endif

        //****************************Stitch tiles***********************************

        if (number_of_tiles > 1) { // are we tiling?

          if (tile_index == 0) {
            stitch_image.assign(raw_image.width(), file_image.height(), raw_image.depth()); // initialize destination stitch_image.
            std::cout << "         stitch_image : " << stitch_image.width() << " x " << stitch_image.height() << " x " << stitch_image.depth() << ". " << std::endl;
            stitch_image.fill((float)0.0);

            blend_weight_image.assign(raw_image.width(), file_image.height(), raw_image.depth()); // initialize destination blend_weight_image.
            blend_weight_image.fill(0);
          }
          // int no_zone    = floor((float)tile_overlap / 3) ;  // 1/3 no_zone, 1/3 blend zone, 1/3 no_zone
          int no_zone    = ceil((tile_overlap - 4) / 2.0) ;  //  no_zone, 4 pixel blend zone, yes_zone
          int blend_zone = tile_overlap - no_zone - no_zone;

          std::cout << "           tile_image : " << raw_image.width() << " x " << raw_image.height() << " x " << raw_image.depth() << ". tile_overlap: " << tile_overlap << ". blend_zone: " << blend_zone << ". tile_y_offset:" << tile_y_offset << ". " << std::endl;
          bool is_first_tile = tile_index == 0;
          bool is_last_tile = tile_index + 1 >= number_of_tiles;

          cimg_forXYZ(raw_image, x, y, z) {
            float blend = 1;

            if (is_first_tile && y <= tile_overlap)
              blend = 1;// front overlap region of 1st tile
            else if (is_last_tile && y >= raw_image.height() - tile_overlap )
              blend = 1;// back overlap region of last tile
            else {
              if (y <= no_zone) // front no_zone?
                blend = 0;

              else if(y > no_zone && y <= no_zone + blend_zone) // front blend region?
                blend = (double)(y - no_zone) / blend_zone;

              else if (y >= raw_image.height() - no_zone) // back no zone?
                blend = 0;

              else blend = 1; // middle zone or back blend zone
            }

            if (y + tile_y_offset < stitch_image.height()) { /* are we inside of the destination image?*/

              if (BlendTileOverlap) { // blend the overlap?
                if (blend > 0 && blend < 1 && blend_weight_image(x, y + tile_y_offset, z) > 0) { // need to blend?
                  stitch_image(x, y + tile_y_offset, z) = (raw_image(x, y, z) * blend) + (stitch_image(x, y + tile_y_offset, z) * (1.0 - blend)); // insert into image with blend in overlap region
                  //stitch_image(x, y + tile_y_offset, z) = blend * 1000;
                }

                else if (blend > 0 && blend_weight_image(x, y + tile_y_offset, z) == 0) // just put this pixel into the image
                  stitch_image(x, y + tile_y_offset, z) = raw_image(x, y, z);

                else if (blend == 1)
                  stitch_image(x, y + tile_y_offset, z) = raw_image(x, y, z); // just put this pixel into the image

                if (blend > 0)
                  blend_weight_image(x, y + tile_y_offset, z) = 1;
              }
              else // overlap is either 1 tile or the other.
                {
                  bool is_past_front_overlap = (y >= floor((float)tile_overlap / 2.0) || is_first_tile);
                  bool is_before_end_overlap = (y < floor(raw_image.height() - (float)tile_overlap / 2.0) || is_last_tile);

                  if (is_past_front_overlap && is_before_end_overlap)
                    stitch_image(x, y + tile_y_offset, z) = raw_image(x, y, z); // insert into image directly without blend in overlap region
                }
            }

          } // end loop on pixels
        } // end if (number_of_tiles > 1)
        else
          stitch_image.assign(raw_image, false); // no tiling, so just do image copy.

      } // end for (int tile_index = 0; ... ) tiling loop

      //****************************Crop***********************************

      //remove padding
      if (Pad)
        stitch_image.crop(
                          border_x, border_y,             // X, Y
                          border_z, 0,                    // Z, C
                          border_x + nx, border_y + ny,   // X, Y
                          border_z + nz, 0);              // Z, C

      if (! final_CropTo_boundaries.empty()) {
        stitch_image.crop(final_CropTo_boundaries[0], final_CropTo_boundaries[2],   // X, Y
                          final_CropTo_boundaries[4], 0,                            // Z, C
                          final_CropTo_boundaries[1], final_CropTo_boundaries[3],   // X, Y
                          final_CropTo_boundaries[5], 0);                           // Z, C
        if (raw_deskewed.size()) {
          // std::cout << std::endl << "The 'raw_deskewed' buffer uses " << raw_deskewed.size()*sizeof(float) << " bytes." << std::endl;
          raw_deskewed.crop(final_CropTo_boundaries[0], final_CropTo_boundaries[2],
                            final_CropTo_boundaries[4], 0,
                            final_CropTo_boundaries[1], final_CropTo_boundaries[3],
                            final_CropTo_boundaries[5], 0);
        }
      }

      if (tsave.joinable()) {
        tsave.join();           // wait for previous saving thread to finish.
        tsave.~thread();        // destroy thread.
      }

      if (tDeskewsave.joinable()) {
        tDeskewsave.join();     // wait for previous saving thread to finish.
        tDeskewsave.~thread();  // destroy thread.
      }


      //****************************Save Deskewed Raw***********************************
      if (bSaveDeskewedRaw && (fabs(deskewAngle) > 0.0)) {
        if (!bSaveUshort) {
          DeskewedToSave.assign(raw_deskewed);
          tDeskewsave = std::thread(DeSkewsave_in_thread, *it, voxel_size, description); //start saving "Deskewed To Save" file.
          //raw_deskewed.save_tiff(makeOutputFilePath(*it, "Deskewed", "_deskewed").c_str(), 0, voxel_size, description);
        }
        else {
          CImg<unsigned short> uint16Img(raw_deskewed);
          uint16Img.save_tiff(makeOutputFilePath(*it, "Deskewed", "_deskewed").c_str(), compression, voxel_size, description);
        }
      }


      //****************************Save Deskewed MIPs***********************************
      if (bDoRawMaxIntProj.size() == 3 && bSaveDeskewedRaw) {
        if(it == all_matching_files.begin() && (bDoRawMaxIntProj[0] || bDoRawMaxIntProj[1] || bDoRawMaxIntProj[2]))
          makeNewDir("Deskewed/MIPs");

        if (bDoRawMaxIntProj[0]) {
          CImg<> proj = MaxIntProj(raw_deskewed, 0);
          proj.save_tiff(makeOutputFilePath(*it, "Deskewed/MIPs", "_MIP_x").c_str(), compression, voxel_size, description);
        }
        if (bDoRawMaxIntProj[1]) {
          CImg<> proj = MaxIntProj(raw_deskewed, 1);
          proj.save_tiff(makeOutputFilePath(*it, "Deskewed/MIPs", "_MIP_y").c_str(), compression, voxel_size, description);
        }
        if (bDoRawMaxIntProj[2]) {
          CImg<> proj = MaxIntProj(raw_deskewed, 2);
          proj.save_tiff(makeOutputFilePath(*it, "Deskewed/MIPs", "_MIP_z").c_str(), compression, voxel_size, description);
        }

      }

      //****************************Save MIPs***********************************

      // I had to modify this a bit to save the behavior of --saveDeskewedRaw when NO RL is being performed...
      // otherwise, it was trying to create folders it shouldn't...
      // this might have unintended side effects (notably with weiner filtering only... and maybe others)
      if (bDoMaxIntProj.size() == 3 && RL_iters && (bDoMaxIntProj[0] || bDoMaxIntProj[1] || bDoMaxIntProj[2])) {
        if (it == all_matching_files.begin() + skip) {
          makeNewDir("GPUdecon/MIPs");
        }

        if (bDoMaxIntProj[0]) {
          CImg<> proj = MaxIntProj(stitch_image, 0);
          if (bSaveUshort) {
            CImg<unsigned short> uint16Img(proj);
            uint16Img.save_tiff(makeOutputFilePath(*it, "GPUdecon/MIPs", "_MIP_x").c_str(), compression, voxel_size, description);
          }
          else
            {
              proj.save_tiff(makeOutputFilePath(*it, "GPUdecon/MIPs", "_MIP_x").c_str(), compression, voxel_size, description);
            }
        }
        if (bDoMaxIntProj[1]) {
          CImg<> proj = MaxIntProj(stitch_image, 1);
          if (bSaveUshort) {
            CImg<unsigned short> uint16Img(proj);
            uint16Img.save_tiff(makeOutputFilePath(*it, "GPUdecon/MIPs", "_MIP_y").c_str(), compression, voxel_size, description);
          }
          else
            {
              proj.save_tiff(makeOutputFilePath(*it, "GPUdecon/MIPs", "_MIP_y").c_str(), compression, voxel_size, description);
            }
        }
        if (bDoMaxIntProj[2]) {
          CImg<> proj = MaxIntProj(stitch_image, 2);
          if (bSaveUshort) {
            CImg<unsigned short> uint16Img(proj);
            uint16Img.save_tiff(makeOutputFilePath(*it, "GPUdecon/MIPs", "_MIP_z").c_str(), compression, voxel_size, description);
          }
          else
            {
              proj.save_tiff(makeOutputFilePath(*it, "GPUdecon/MIPs", "_MIP_z").c_str(), compression, voxel_size, description);
            }
        }
      }

      //****************************Save Decon Image***********************************
      if (RL_iters || rotMatrix.getSize()) {
        // Stupid to redefine these here... but couldn't get the Z voxel size to work
        // correctly in ImageJ otherwise...

        if (!bSaveUshort) {

          ToSave.assign(stitch_image); //copy decon image (i.e. stitch_image) to new image space for saving.
          // ToSave.save_tiff(makeOutputFilePath(*it).c_str(), compression, voxel_size, description);

          tsave = std::thread(save_in_thread, *it, voxel_size_decon, imdz); //start saving "To Save" file.
        }
        else {
          U16ToSave = stitch_image;
          tsave = std::thread(U16save_in_thread, *it, voxel_size_decon, imdz); //start saving "To Save" file.
        }
      }

      /// please leave this here for LLSpy
      printf(">>>file_finished\n");

      iter_duration = (std::clock() - start_t) / (double)CLOCKS_PER_SEC / (it - (all_matching_files.begin() + skip) + 1 );
    } // end for (std::vector<std::string>::iterator it= all_matching_files.begin() + skip;

    if (tsave.joinable()) {
      tsave.join();               // wait for previous saving thread to finish.
      tsave.~thread();            // destroy thread.
    } //Make sure we have finished saving.

    if (tDeskewsave.joinable()) {
      tDeskewsave.join();         // wait for previous saving thread to finish.
      tDeskewsave.~thread();      // destroy thread.
    } //Make sure we have finished saving.

    duration = (std::clock() - start_t) / (double)CLOCKS_PER_SEC;

#ifdef _WIN32
    SetConsoleTextAttribute(hConsole, 10); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
#endif
    std::cout << "*** Finished! Elapsed " << duration << " seconds.  ";
    if (skip != 0)
      std::cout << "Skipped " << skip << "images.  ";
    std::cout << "Processed " << all_matching_files.size() - skip << " images.  " << duration / all_matching_files.size() << " seconds per image. ***" << std::endl;
#ifdef _WIN32
    SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
#endif

  } // end try {} block
  catch (std::exception &e) {
    std::cerr << "\n!!Error occurred: " << e.what() << std::endl;
    return 0;
  }
  return 0;
}


CImg<> MaxIntProj(CImg<> &input, int axis)
{
  CImg <> out;

  if (axis==0) {
    CImg<> maxvals(input.height(), input.depth());
    maxvals = -1e10;
#pragma omp parallel for
    cimg_forYZ(input, y, z) for (int x=0; x<input.width(); x++) {
      if (input(x, y, z) > maxvals(y, z))
        maxvals(y, z) = input(x, y, z);
    }
    return maxvals;
  }
  else if (axis==1) {
    CImg<> maxvals(input.width(), input.depth());
    maxvals = -1e10;
#pragma omp parallel for
    cimg_forXZ(input, x, z) for (int y=0; y<input.height(); y++) {
      if (input(x, y, z) > maxvals(x, z))
        maxvals(x, z) = input(x, y, z);
    }
    return maxvals;
  }

  else if (axis==2) {
    CImg<> maxvals(input.width(), input.height());
    maxvals = -1e10;
#pragma omp parallel for
    cimg_forXY(input, x, y) for (int z=0; z<input.depth(); z++) {
      if (input(x, y, z) > maxvals(x, y))
        maxvals(x, y) = input(x, y, z);
    }
    return maxvals;
  }
  else {
    throw std::runtime_error("unknown axis number in MaxIntProj()");
  }
  return out;
}
=======
#include "linearDecon.h"
#include <exception>
#include <ctime>

#ifdef _WIN32

// Disable silly warnings on some Microsoft VC++ compilers.
#pragma warning(disable : 4244) // Disregard loss of data from float to int.
#pragma warning(disable : 4267) // Disregard loss of data from size_t to unsigned int.
#pragma warning(disable : 4305) // Disregard loss of data from double to float.
#endif

std::string version_number = "0.4.1";
CImg<> next_file_image;

CImg<> ToSave;
CImg<unsigned short> U16ToSave;
CImg<> DeskewedToSave;

int load_next_thread(const char* my_path)
{
    next_file_image.assign(my_path);
    if (false)
    {
        float img_max = next_file_image(0, 0, 0);
        float img_min = next_file_image(0, 0, 0);
        cimg_forXYZ(next_file_image, x, y, z) {
            img_max = std::max(next_file_image(x, y, z), img_max);
            img_min = std::min(next_file_image(x, y, z), img_min);
        }
        std::cout << "next_file_image : " << next_file_image.width() << " x " << next_file_image.height() << " x " << next_file_image.depth() << ". " << std::endl;
        std::cout << "         next_file_image max, min : " << std::setw(8) << img_max << ", " << std::setw(8) << img_min << std::endl;
        std::cout << "Loaded from " << my_path << std::endl;
    }

    return 0;
}

unsigned compression = 0;


std::string make_Image_Description(float nz, float dz)
{
	// Need to impersonate ImageJ's Image Description Tiff tag for many programs to work (Arivis, ImageJ, etc)
	std::string temp = "ImageJ=1.52o\n"
		"spacing=" + std::to_string(dz) + "\n"
		"unit=micron" "\n"
		"images=" + std::to_string(nz) + "\n"
		"slices=" + std::to_string(nz) + "\n"; // "slices" will designate this as a volume and not a time series, "frames"
	// std::cout << "my image description=" << temp;  // debug it:
	return temp;
}


int save_in_thread(std::string inputFileName, const float *const voxel_size, float dz)
{
	std::string tiff_descript = make_Image_Description(ToSave.depth(), dz);
    ToSave.save_tiff(makeOutputFilePath(inputFileName).c_str(), compression, voxel_size, tiff_descript.c_str());

    return 0;
}

int U16save_in_thread(std::string inputFileName, const float *const voxel_size, float dz)
{   
	std::string tiff_descript = make_Image_Description(U16ToSave.depth(), dz);
    U16ToSave.save_tiff(makeOutputFilePath(inputFileName).c_str(), compression, voxel_size, tiff_descript.c_str());

    return 0;
}

int DeSkewsave_in_thread(std::string inputFileName, const float *const voxel_size, const char *const description)
{
	DeskewedToSave.save_tiff(makeOutputFilePath(inputFileName, "Deskewed", "_deskewed").c_str(), compression, voxel_size, description);
    
    return 0;
}


std::complex<float> otfinterpolate(std::complex<float> * otf, float kx, float ky, float kz, int nzotf, int nrotf)
// Use sub-pixel coordinates (kx,ky,kz) to linearly interpolate a rotationally-averaged 3D OTF ("otf").
// otf has 2 dimensions: fast dimension is kz with length "nzotf" while the slow dimension is kr.
{
    int irindex, izindex, indices[2][2];
    float krindex, kzindex;
    float ar, az;

    krindex = sqrt(kx*kx + ky*ky);
    kzindex = (kz<0 ? kz+nzotf : kz);

    if (krindex < nrotf-1 && kzindex < nzotf) {
        irindex = floor(krindex);
        izindex = floor(kzindex);

        ar = krindex - irindex;
        az = kzindex - izindex;  // az is always 0 for 2D case, and it'll just become a 1D interp

        if (izindex == nzotf-1) {
            indices[0][0] = irindex*nzotf+izindex;
            indices[0][1] = irindex*nzotf+0;
            indices[1][0] = (irindex+1)*nzotf+izindex;
            indices[1][1] = (irindex+1)*nzotf+0;
        }
        else {
            indices[0][0] = irindex*nzotf+izindex;
            indices[0][1] = irindex*nzotf+(izindex+1);
            indices[1][0] = (irindex+1)*nzotf+izindex;
            indices[1][1] = (irindex+1)*nzotf+(izindex+1);
        }

        return (1-ar)*(otf[indices[0][0]]*(1-az) + otf[indices[0][1]]*az) +
               ar*(otf[indices[1][0]]*(1-az) + otf[indices[1][1]]*az);
    }
    else
        return std::complex<float>(0, 0);
}

int wienerfilter(CImg<> & g, float dkx, float dky, float dkz,
                 CImg<> & otf, float dkr_otf, float dkz_otf,
                 float rcutoff, float wiener)
{
    /* 'g' is the raw data's FFT (half kx axis); it is also the result upon return */
    int i, j, k;
    float kz, ky, kx;
    float amp2, rho, kxscale, kyscale, kzscale, kr;
    std::complex<float> A_star_g, otf_val;
    float w;

    w = wiener*wiener;
    kxscale = dkx/dkr_otf;
    kyscale = dky/dkr_otf;
    kzscale = dkz/dkz_otf;

    int nx = g.width()/2; // '/2' because g is CImg<float> hijacked for complex storage
    int ny = g.height();
    int nz = g.depth();

    std::complex<float> result;

    #pragma omp parallel for private(k, i, j, kz, ky, kx, kr, otf_val, amp2, A_star_g, rho, result)
    for (k=0; k<nz; k++) {
        kz = ( k>nz/2 ? k-nz : k );
        for (i=0; i<ny; i++) {
            ky = ( i > ny/2 ? i-ny : i );
            for (j=0; j<nx; j++) {
                kx = j;
                kr = sqrt(kx*kx*dkx*dkx + ky*ky*dky*dky);
                if (kr <=rcutoff) {
                    otf_val = otfinterpolate((std::complex<float>*) otf.data(),
                                             kx*kxscale, ky*kyscale,
                                             kz*kzscale, otf.width()/2, otf.height());

                    amp2 = otf_val.real() * otf_val.real() + otf_val.imag() * otf_val.imag();
                    A_star_g = std::conj(otf_val) * std::complex<float>(g(2*j, i, k), g(2*j+1, i, k));

                    /* apodization */
                    rho = kr / rcutoff;
                    result = A_star_g / (amp2+w) * (1-rho);
                    g(2*j, i, k) = result.real();
                    g(2*j+1, i, k) = result.imag();
                }
                else {
                    g(2*j, i, k) = 0;
                    g(2*j+1, i, k) = 0;
                }
            }
        }
    }
    return 0;
}

int main(int argc, char *argv[])
{
    #ifdef _WIN32
    HANDLE  hConsole;
    hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

    SetConsoleTextAttribute(hConsole, 11); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
    printf("Created at Howard Hughes Medical Institute Janelia Research Campus. Copyright 2020. All rights reserved.\n");
    SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
    #endif
    std::clock_t start_t;
    double duration;
    double iter_duration = 0;


    start_t = std::clock();

    int napodize, nZblend;
    float background;
    float NA=1.2;
    ImgParams imgParams, imgParamsOld;
    float dz_psf, dr_psf;
    float wiener;

    int myGPUdevice=0;
    int RL_iters=0;
    bool bSaveDeskewedRaw = false;
    bool bDontAdjustResolution = false;

    bool BlendTileOverlap = false;
    float deskewAngle=0.0;
    float rotationAngle=0.0;
    unsigned outputWidth;
    float padVal = 0.0;

    int extraShift=0;
    std::vector<int> final_CropTo_boundaries;
    bool bSaveUshort = false;
    std::vector<bool> bDoMaxIntProj;
    std::vector<bool> bDoRawMaxIntProj;
    std::vector< CImg<> > MIprojections;
    int Pad = 0;
    bool bFlatStartGuess = false;
    bool bDoRescale = false;
    bool UseOnlyHostMem = false;
    bool no_overwrite = false;
    bool lzw = false;
    int skip = 0;
    bool bDupRevStack = false;
    int tile_size=0;
    int tile_requested = 0;
    int tile_overlap_requested = 20;

    TIFFSetWarningHandler(NULL);

    std::string datafolder, filenamePattern, otffiles, LSfile;
    po::options_description progopts("cudaDeconv. Version: " + version_number + "\n");
    progopts.add_options()
    ("drdata", po::value<float>(&imgParams.dr)->default_value(.104), "Image x-y pixel size (um)")
    ("dzdata,z", po::value<float>(&imgParams.dz)->default_value(.25), "Image z step (um)")
    ("drpsf", po::value<float>(&dr_psf)->default_value(.104), "PSF x-y pixel size (um)")
    ("dzpsf,Z", po::value<float>(&dz_psf)->default_value(.1), "PSF z step (um)")
    ("wavelength,l", po::value<float>(&imgParams.wave)->default_value(.525), "Emission wavelength (um)")
    ("wiener,W", po::value<float>(&wiener)->default_value(-1.0), "Wiener constant (regularization factor); if this value is postive then do Wiener filter instead of R-L")
    ("background,b", po::value<float>(&background)->default_value(90.f), "User-supplied background")
    ("napodize,e", po::value<int>(&napodize)->default_value(15), "# of pixels to soften edge with")
    ("nzblend,E", po::value<int>(&nZblend)->default_value(0), "# of top and bottom sections to blend in to reduce axial ringing")
    ("dupRevStack,d", po::bool_switch(&bDupRevStack)->default_value(false), "Duplicate reversed stack prior to decon to reduce Z ringing")
    ("NA,n", po::value<float>(&NA)->default_value(1.2), "Numerical aperture")
    ("RL,i", po::value<int>(&RL_iters)->default_value(15), "Run Richardson-Lucy, and set how many iterations")
    ("deskew,D", po::value<float>(&deskewAngle)->default_value(0.0), "Deskew angle; if not 0.0 then perform deskewing before deconv")
    ("padval", po::value<float>(&padVal)->default_value(0.0), "Value to pad image with when deskewing")
    ("width,w", po::value<unsigned>(&outputWidth)->default_value(0), "If deskewed, the output image's width")
    ("shift,x", po::value<int>(&extraShift)->default_value(0), "If deskewed, the output image's extra shift in X (positive->left")
    ("rotate,R", po::value<float>(&rotationAngle)->default_value(0.0), "Rotation angle; if not 0.0 then perform rotation around y axis after deconv")
    ("saveDeskewedRaw,S", po::bool_switch(&bSaveDeskewedRaw)->default_value(false), "Save deskewed raw data to files")
    ("crop,C", fixed_tokens_value< std::vector<int> >(&final_CropTo_boundaries, 6, 6), "Crop final image size to [x1:x2, y1:y2, z1:z2]; takes 6 integers separated by space: x1 x2 y1 y2 z1 z2; ")
    ("MIP,M", fixed_tokens_value< std::vector<bool> >(&bDoMaxIntProj, 3, 3), "Save max-intensity projection along x, y, or z axis; takes 3 binary numbers separated by space: 0 0 1")
    ("rMIP,m", fixed_tokens_value< std::vector<bool> >(&bDoRawMaxIntProj, 3, 3), "Save max-intensity projection of raw deskewed data along x, y, or z axis; takes 3 binary numbers separated by space: 0 0 1")
    ("uint16,u", po::bool_switch(&bSaveUshort)->implicit_value(true), "Save result in uint16 format; should be used only if no actual decon is performed")
    ("input-dir", po::value<std::string>(&datafolder)->required(), "Folder of input images")
    ("otf-file", po::value<std::string>(&otffiles)->required(), "OTF file")
    ("filename-pattern", po::value<std::string>(&filenamePattern)->required(), "File name pattern to find input images to process")
    ("DoNotAdjustResForFFT,a", po::bool_switch(&bDontAdjustResolution)->default_value(false), "Don't change data resolution size. Otherwise data is cropped to perform faster, more memory efficient FFT: size factorable into 2,3,5,7)")
    ("Pad", po::value<int>(&Pad)->default_value(0), "Pad the image data with mirrored values to avoid edge artifacts. Currently only enabled when rotate and deskew are zero.")
    ("LSC", po::value<std::string>(&LSfile), "Lightsheet correction file")
    ("FlatStart", po::bool_switch(&bFlatStartGuess)->default_value(false), "Start the RL from a guess that is a flat image filled with the median image value.  This may supress noise.")
    ("bleachCorrection,p", po::bool_switch(&bDoRescale)->default_value(false), "Apply bleach correction when running multiple images in a single batch")
    ("lzw", po::bool_switch(&lzw)->default_value(false), "Use LZW tiff compression")
    ("skip", po::value<int>(&skip)->default_value(0), "Skip the first 'skip' number of files.")
    ("no_overwrite", po::bool_switch(&no_overwrite)->default_value(false), "Don't reprocess files that are already deconvolved (i.e. exist in the GPUdecon folder).")
    ("tile", po::value<int>(&tile_requested)->default_value(0), "Tile size for tiled decon (in Y only) to attempt to fit into GPU. 0=no tiling. Best to use a power of 2 (i.e. 64, 128, etc).")
    ("TileOverlap", po::value<int>(&tile_overlap_requested)->default_value(30), "Overlap between Tiles.  You want this to be at least ~2x the PSF Y extent.")
    ("BlendTileOverlap", po::bool_switch(&BlendTileOverlap)->default_value(false), "Blend ~5 pixel in Overlap region between Tiles.")
    ("DevQuery,Q", "Show info and indices of available GPUs")
    ("help,h", "This help message.")
    ("version,v", "show version and quit")
    // ("GPUdevice", po::value<int>(&myGPUdevice)->default_value(0), "Index of GPU device to use (0=first device)")
    // ("UseOnlyHostMem", po::bool_switch(&UseOnlyHostMem)->default_value(false), "Just use Host Mapped Memory, and not GPU. For debugging only.")
    ;
    po::positional_options_description p;
    p.add("input-dir", 1);
    p.add("filename-pattern", 1);
    p.add("otf-file", 1);

    std::string commandline_string = __DATE__ ;
    commandline_string.append(" ");
    commandline_string.append(__TIME__);
    for (int i = 0; i < argc; i++) {
        commandline_string.append(" ");
        commandline_string.append(argv[i]);
    } // store commandline_string


    // Parse commandline option:
    po::variables_map varsmap;
    try {
        if (argc == 1)  { //if no arguments, show help.
            std::cout << progopts << "\n";
            return 0;
        }

        store(po::command_line_parser(argc, argv).
              options(progopts).positional(p).run(), varsmap);
        if (varsmap.count("help")) {
            std::cout << progopts << "\n";
            return 0;
        }

        if (varsmap.count("version")) {
            std::cout << version_number << "\n";
            return 0;
        }


        //****************************Query GPU devices***********************************
        if (varsmap.count("DevQuery")) {
            int deviceCount = 0;
            cudaGetDeviceCount(&deviceCount);
            // This function call returns 0 if there are no CUDA capable devices.
            if (deviceCount != 0)
                printf("Detected %d CUDA Capable device(s)\n", deviceCount);

            int dev, driverVersion = 0, runtimeVersion = 0;

            for (dev = 0; dev < deviceCount; ++dev)
            {
                cudaSetDevice(dev);
                cudaDeviceProp mydeviceProp;
                cudaGetDeviceProperties(&mydeviceProp, dev);
                printf("\nDevice %d: \"%s\"\n", dev, mydeviceProp.name);

                cudaDriverGetVersion(&driverVersion);
                cudaRuntimeGetVersion(&runtimeVersion);
                printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
                printf("  CUDA Capability Major/Minor version number:    %d.%d\n", mydeviceProp.major, mydeviceProp.minor);
                printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                       (float)mydeviceProp.totalGlobalMem / 1048576.0f, (unsigned long long) mydeviceProp.totalGlobalMem);
            }
            return 0; // added return because I want query to simply query and quit
        }

        notify(varsmap);

        // if (varsmap.count("crop")) {
        //   if (final_CropTo_boundaries.size() != 6)
        //     throw std::runtime_error("Exactly 6 integers are required for the -C or --crop flag!");
        //   std::copy(final_CropTo_boundaries.begin(), final_CropTo_boundaries.end(), std::ostream_iterator<int>(std::cout, ", "));
        //   std::cout << std::endl;
        // }
        // std::cout << bDoMaxIntProj.size() << std::endl;
    }
    catch (std::exception &e) {
        std::cout << "\n!!Error occurred: " << e.what() << std::endl;
        return 0;
    }

    bool bAdjustResolution = !bDontAdjustResolution;
    imgParamsOld = imgParams; // Copy image Params

    //****************************Check to see if we have a proper GPU device***********************************
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }
    if (deviceCount == 0)
        printf("There are no available device(s) that support CUDA\n");


    cudaSetDeviceFlags(cudaDeviceMapHost);
    size_t GPUfree;
    size_t GPUtotal;
    cudaMemGetInfo(&GPUfree, &GPUtotal);
    cudaDeviceProp mydeviceProp;
    cudaGetDeviceProperties(&mydeviceProp, myGPUdevice);

    #ifdef _WIN32
    SetConsoleTextAttribute(hConsole, 13); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
    #endif

    std::cout << std::endl << "Built : " << __DATE__ << " " << __TIME__ << ".  GPU " << GPUfree / (1024 * 1024) << " MB free / " << GPUtotal / (1024 * 1024) << " MB total on " << mydeviceProp.name << std::endl;

    CImg<> raw_image, raw_imageFFT, complexOTF, raw_deskewed, LSImage, sub_image, file_image, stitch_image, blend_weight_image;
    CImg<float> AverageLS;
    float dkr_otf, dkz_otf;
    float dkx, dky, dkz, rdistcutoff;
    fftwf_plan rfftplan=NULL, rfftplan_inv=NULL;
    CPUBuffer rotMatrix;
    double deskewFactor=0;
    bool bCrop = false;
    unsigned new_ny, new_nz, new_nx;
    int deskewedXdim = 0;
    cufftHandle rfftplanGPU = NULL, rfftplanInvGPU = NULL;
    GPUBuffer d_interpOTF(1, myGPUdevice, UseOnlyHostMem);
    GPUBuffer workArea(1, myGPUdevice, UseOnlyHostMem);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, myGPUdevice);

    unsigned nx, ny, nz = 0;

    int border_x = 0;
    int border_y = 0;
    int border_z = 0;

	float voxel_size [3];
	float voxel_size_decon [3];
	float imdz;
	

    if (lzw) {
        compression = 1;
    }
    //****************************Main processing***********************************
    // Loop over all matching input TIFFs, :
    try {


        std::cout << "Looking for files to process... " ;
        // Gather all files in 'datafolder' and matching the file name pattern:
        //bool MIPsOnly = (RL_iters == 0 && bDoMaxIntProj.size() == 3);

        std::vector< std::string > all_matching_files = gatherMatchingFiles(datafolder, filenamePattern, no_overwrite);
        std::cout << "Found " << all_matching_files.size() << " file(s)." << std::endl ;

        #ifdef _WIN32
        SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
        #endif
        if (all_matching_files.size() == 0) {
            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, 10); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
            #endif
            std::cout<< "\nNo files need processing!" << std::endl;
            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
            #endif
            return 0;
        }


        std::vector<std::string>::iterator next_it = all_matching_files.begin() + skip;//make a second incrementer that will be used to load the next raw image while we process.

        std::thread t1; // thread for loading files
        t1 = std::thread(load_next_thread, next_it->c_str());   //start loading the first file.

        std::thread tsave;  // thread for saving decon
        std::thread tDeskewsave; // thread for saving deskewed

        for (std::vector<std::string>::iterator it= all_matching_files.begin() + skip;
                it != all_matching_files.end(); it++) {

            int number_of_files_left = all_matching_files.end() - it;

            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, 10); // colors are 9=blue 10=green and so on to 15=white
            #endif
            std::cout << std::endl << "Loading raw_image: " << it - all_matching_files.begin() + 1 << " out of " << all_matching_files.size() << ".   ";
            if (it > all_matching_files.begin() + skip) // if this isn't the first iteration.
            {
                int seconds = number_of_files_left * iter_duration;
                int hours = ceil(seconds / (60 * 60));
                int minutes = ceil((seconds - (hours * 60 * 60)) / 60);
                std::cout << (int)iter_duration << " s/file.   " << number_of_files_left << " files left.  " << hours << " hours, " <<  minutes << " minutes remaining.";
            }

            std::cout <<  std::endl;
            #ifdef _WIN32
            SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
            #endif

            std::cout << std::endl << *it << std::endl;
            std::cout << "Waiting for separate thread to finish loading image... " ;
            t1.join();        // wait for loading thread to finish reading next_raw_image into memory.
            t1.~thread();     // destroy thread.
            std::cout << "Image loaded. Copying to 'file_image'... " ;
            file_image.assign(next_file_image, false); // Copy to file_image.  CImg<T>& assign(const CImg<t>& img, const bool is_shared)

            float img_max = file_image(0, 0);
            float img_min = file_image(0, 0);

            #ifndef NDEBUG
            cimg_forXYZ(file_image, x, y, z) {
                img_max = std::max(file_image(x, y, z), img_max);
                img_min = std::min(file_image(x, y, z), img_min);
            }

            std::cout << "         raw img max, min : " << std::setw(8) << img_max << ", " << std::setw(8) << img_min << std::endl;
            #endif



            std::cout << "Done." << std::endl;

            // start reading the next image
            next_it++; // increment next_it.  This will now have the next file to read.
            if (it < all_matching_files.end() - 1)  // If there are more files to process...
                t1 = std::thread(load_next_thread, next_it->c_str());		//start new thread and load the next file, while we process file_image

            //**************************** Tiling setup ***********************************
            int number_of_tiles = 1;
            int number_of_overlaps = 0;
            double tile_overlap = 0;


            if (tile_requested < 0) {
                tile_size = std::min(           128, file_image.height());
            }
            if (tile_requested > 0) {
                tile_size = std::min(tile_requested, file_image.height());
            }

            number_of_tiles = ceil((double)file_image.height() / ((double)tile_size - tile_overlap_requested)); // try for 20 pixel overlap
            number_of_overlaps = number_of_tiles - 1;
            tile_overlap = (((double)tile_size * number_of_tiles) - file_image.height()) / number_of_overlaps;
            tile_overlap = floor(tile_overlap); //


            if (tile_requested == 0) {
                number_of_tiles = 1;
                tile_size = file_image.height();
                tile_overlap = 0;
                number_of_overlaps = 0;
            }
            std::cout << std::endl << "# of tiles: " << number_of_tiles << ". # of overlaps: " << number_of_overlaps << ". Tile size: " << tile_size << " Y pix. Overlap: " << tile_overlap << " Y pix. " << std::endl;

            for (int tile_index = 0; tile_index < number_of_tiles; tile_index++) {
                int tile_y_offset = floor( (double)tile_index*(tile_size - tile_overlap) );
                //std::cout << std::endl << "tile_y_offset: " << tile_y_offset;

                if (number_of_tiles > 1) {

                    //int y_end = std::min(file_image.height(), tile_size + tile_y_offset);
                    int y_end = tile_size + tile_y_offset; // I think we can crop to outside the image, and the boundry conditions will fill it. This way each subimage is exactly the same size, which makes the rest of the code mighty happy.


                    raw_image = file_image.get_crop(0,	/* X start */
                                                    tile_y_offset,					/* Y start */
                                                    0,								/* Z start */
                                                    file_image.width() - 1,			/*  X end */
                                                    y_end - 1,						/*  Y end */
                                                    file_image.depth() - 1,			/*  Z end */
                                                    true); // get sub_image. raw_image = sub_image


                    //for (int y = tile_y_offset; y < y_end; ++y) {

                    //cimg_forXZ(file_image, x, z) {
                    //raw_image(x, y_raw_index, z) = (unsigned long) file_image(x, y, z);
                    //}
                    //y_raw_index = y_raw_index + 1;
                    //}
                    //raw_image = file_image.get_crop(0,
                    //	tile_y_offset,
                    //	0,
                    //	0,
                    //	file_image.width() - 1,
                    //	std::min(file_image.height(), tile_size + tile_y_offset) - 1,
                    //	file_image.depth() - 1,
                    //	0); // get sub_image. raw_image = sub_image
					#ifdef _WIN32
                    SetConsoleTextAttribute(hConsole, 10); // colors are 9=blue 10=green and so on to 15=white
					#endif
                    std::cout << std::endl << "Tile: " << tile_index + 1 << " out of " << number_of_tiles << ".   " << std::endl;
					#ifdef _WIN32
                    SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
                    #endif
					// raw_image.save(makeOutputFilePath(it->c_str(), "tile", "_tile").c_str()); //for debugging
                }
                else
                    raw_image.assign(file_image);

                // If it's the first input file, initialize a bunch including:
                // 1. crop image to make dimensions nice factorizable numbers
                // 2. calculate deskew parameters, new X dimensions
                // 3. calculate rotation matrix
                // 4. create FFT plans
                // 5. transfer constants into GPU device constant memory
                // 6. make 3D OTF array in device memory

                bool bDifferent_sized_raw_img = (nx != raw_image.width() || ny != raw_image.height() || nz != raw_image.depth() ); // Check if raw.image has changed size from first iteration

                if (it != all_matching_files.begin() + skip && bDifferent_sized_raw_img) {
                    #ifdef _WIN32
                    SetConsoleTextAttribute(hConsole, 14); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
                    #endif
                    std::cout << std::endl << "File " << it - all_matching_files.begin() + 1 << " has a different size" << std::endl;
                }
                std::cout << "raw_image size             : " << raw_image.width() << " x " << raw_image.height() << " x " << raw_image.depth() << std::endl;
                #ifdef _WIN32
                SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
                #endif


                //**************************** If first image OR if raw_img has changed size ***********************************
                if (nx == 0 || bDifferent_sized_raw_img ) {
                    nx = raw_image.width();
                    ny = raw_image.height();
                    nz = raw_image.depth();
                    imgParams = imgParamsOld;



                    //****************************Adjust resolution if desired***********************************

                    int step_size;

                    if (fabs(deskewAngle) > 0.0 || fabs(rotationAngle) > 0.0)
                    {
                        Pad = 0;	// Currently padding is disabled if we are deskewing or rotating.
                        std::cout << "Currently padding is disabled if we are deskewing or rotating." << std::endl;
                    }

                    if (Pad)
                        step_size =  1;
                    else
                        step_size = -1;

                    int startnx = nx;
                    int startny = ny;
                    int startnz = nz;

                    if (Pad) //pad by N number of border pixels
                    {
                        startnx = startnx + Pad * 2;
                        startny = startny + Pad * 2;
                        startnz = startnz + Pad * 2;
                        std::cout << "Min padding to use is " << Pad << " pixels." << std::endl;
                    }

                    if (RL_iters > 0 && (bAdjustResolution || Pad)) {
                        new_ny = findOptimalDimension(startny, step_size);
                        if (new_ny != startny) {
                            printf("new ny=%d\n", new_ny);
                            bCrop = true;
                        }

                        new_nz = findOptimalDimension(startnz, step_size);
                        if (new_nz != startnz) {
                            printf("new nz=%d\n", new_nz);
                            bCrop = true;
                        }

                        // only if no deskewing is happening do we want to change image width here
                        new_nx = startnx;
                        if (!(fabs(deskewAngle) > 0.0)) {
                            new_nx = findOptimalDimension(startnx, step_size);
                            if (new_nx != startnx) {
                                printf("new nx=%d\n", new_nx);
                                bCrop = true;
                            }
                        }
                    }
                    else {
                        new_nx = nx;
                        new_ny = ny;
                        new_nz = nz;
                    }


                    //****************************Load OTF to CPU RAM(assuming 3D rotationally averaged OTF)***********************************
                    std::cout <<  "Loading OTF... ";
                    complexOTF.assign(otffiles.c_str());
                    unsigned nr_otf = complexOTF.height();		// because it is rotationally averaged, kx = ky = kr
                    unsigned nz_otf = complexOTF.width() / 2;	// because we are storing complex data in the .tiff file as two pixels: the real "width" is half the number of pixels in a row.
                    dkr_otf = 1/((nr_otf-1)*2 * dr_psf);
                    dkz_otf = 1/(nz_otf * dz_psf);
                    std::cout << "nr x nz     : " << nr_otf << " x " << nz_otf << ". " << std::endl << std::endl ;


                    //****************************Construct deskew matrix***********************************
                    deskewedXdim = new_nx;
                    if (fabs(deskewAngle) > 0.0) {
                        float old_dz = imgParams.dz;
                        if (deskewAngle <0) deskewAngle += 180.;
                        deskewFactor = cos(deskewAngle * M_PI/180.) * imgParams.dz / imgParams.dr;
                        if (outputWidth ==0)
                            deskewedXdim += floor(new_nz * imgParams.dz *
                                                  fabs(cos(deskewAngle * M_PI/180.)) / imgParams.dr);
                        // TODO: sometimes deskewedXdim calc'ed this way is too large
                        else
                            deskewedXdim = outputWidth; // use user-provided output width if available

                        // Adjust resolution to optimal FFT size if desired
                        if (bAdjustResolution)
                            deskewedXdim = findOptimalDimension(deskewedXdim);

                        // update z step size:
                        imgParams.dz *= sin(deskewAngle * M_PI/180.);
                        if (fabs(deskewFactor) < 1)
                        {
                            #ifdef _WIN32
                            SetConsoleTextAttribute(hConsole, 14); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
                            #endif
                            printf("Warning : deskewFactor is < 1.  Check that angle, dz, and dr sizes are correct.\n");
                            #ifdef _WIN32
                            SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
                            #endif
                        }
                        printf("old dz = %f, dxy = %f, deskewFactor = %f, new nx = %d, new dz = %f\n", old_dz, imgParams.dr, deskewFactor, deskewedXdim, imgParams.dz);

                    }



                    //****************************Construct rotation matrix***********************************
                    if (fabs(rotationAngle) > 0.0) {
                        rotMatrix.resize(4*sizeof(float));
                        rotationAngle *= M_PI/180;
                        float stretch = imgParams.dr / imgParams.dz;
                        float *p = (float *)rotMatrix.getPtr();
                        p[0] = cos(rotationAngle) * stretch;
                        p[1] = sin(rotationAngle) * stretch;
                        p[2] = -sin(rotationAngle);
                        p[3] = cos(rotationAngle);
                    }



                    if (wiener >0) {
                        //****************************Wiener filter instead of RL (not usually used.):***********************************
                        raw_imageFFT.assign(deskewedXdim+2, new_ny, new_nz);

                        if (!fftwf_init_threads()) { /* one-time initialization required to use threads */
                            printf("Error returned by fftwf_init_threads()\n");
                        }

                        fftwf_plan_with_nthreads(8);
                        rfftplan = fftwf_plan_dft_r2c_3d(new_nz, new_ny, deskewedXdim,
                                                         raw_image.data(),
                                                         (fftwf_complex *) raw_imageFFT.data(),
                                                         FFTW_ESTIMATE);
                        rfftplan_inv = fftwf_plan_dft_c2r_3d(new_nz, new_ny, deskewedXdim,
                                                             (fftwf_complex *) raw_imageFFT.data(),
                                                             raw_image.data(),
                                                             FFTW_ESTIMATE);
                    }
                    else if (RL_iters>0) {

                        //****************************RL decon***********************************

                        //****************************Create reusable cuFFT plans***********************************
                        //
                        size_t workSize = 0;
                        cufftResult cuFFTErr;

                        if (cufftGetSize(rfftplanGPU, &workSize) == CUFFT_SUCCESS) { // if plan existed before, destroy it before creating a new plan.
                            cufftDestroy(rfftplanGPU);
                            std::cout << "Destroying rfftplanGPU." << std::endl;
                        }
                        if (cufftGetSize(rfftplanInvGPU, &workSize) == CUFFT_SUCCESS) { // if plan existed before, destroy it before creating a new plan.
                            cufftDestroy(rfftplanInvGPU);
                            std::cout << "Destroying rfftplanInvGPU." << std::endl;
                        }

                        size_t GPUfree_prev;
                        cudaMemGetInfo(&GPUfree_prev, &GPUtotal);

                        unsigned nz_final;
                        if (bDupRevStack)  // to reduce severe Z ringing
                            nz_final = new_nz*2;
                        else
                            nz_final = new_nz;

                        if (0) {
                            // old way,  just autoallocate FFTplans.
                            // This will always be on the GPU, and the two plans 
                            // (even though are serially exectuted) cannot share workspace.
                            cuFFTErr = cufftPlan3d(&rfftplanGPU, nz_final, new_ny, deskewedXdim, CUFFT_R2C);
                            if (cuFFTErr != CUFFT_SUCCESS) {
                                // std::cerr << "cufftPlan3d() r2c failed. Error code: " << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;
                                cudaMemGetInfo(&GPUfree, &GPUtotal);
                                std::cerr << "GPU " << GPUfree / (1024 * 1024) << " MB free / " << GPUtotal / (1024 * 1024) << " MB total. " << std::endl;

                                cufftEstimate3d(nz_final, new_ny, deskewedXdim, CUFFT_R2C, &workSize);
                                std::cerr << "R2C FFT Plan desires " << workSize / (1024 * 1024) << " MB. " << std::endl;
                                throw std::runtime_error("cufftPlan3d() r2c failed.");
                            }


                            cuFFTErr = cufftPlan3d(&rfftplanInvGPU, nz_final, new_ny, deskewedXdim, CUFFT_C2R);
                            if (cuFFTErr != CUFFT_SUCCESS) {
                                // std::cerr << "cufftPlan3d() c2r failed. Error code: " << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;
                                cudaMemGetInfo(&GPUfree, &GPUtotal);
                                std::cerr << "GPU " << GPUfree / (1024 * 1024) << " MB free / " << GPUtotal / (1024 * 1024) << " MB total. " << std::endl;

                                cufftEstimate3d(nz_final, new_ny, deskewedXdim, CUFFT_R2C, &workSize);
                                std::cerr << "C2R FFT Plan desires " << workSize / (1024 * 1024) << " MB. " << std::endl;
                                throw std::runtime_error("cufftPlan3d() c2r failed.");
                            }
                        }
                        else { 
                            // new way, share FFTplan workarea.
                            // Possibly put this area on the Host RAM if it doesn't fit in GPU RAM.
                            int autoAllocate   = 0;	// don't allocate, we want to specify the workspace ourselves.
                            size_t workSizeR2C = 0; // size of R2C plan to be filled in by cuFFT
                            size_t workSizeC2R = 0; // size of C2R plan to be filled in by cuFFT
                            workSize    = 0; // necessary size of plan to enable R2C or C2R (i.e. max of these)

                            cuFFTErr = cufftCreate(&rfftplanGPU);							// create object.
                            cuFFTErr = cufftSetAutoAllocation(rfftplanGPU, autoAllocate);
                            cuFFTErr = cufftMakePlan3d(rfftplanGPU, new_nz, new_ny, deskewedXdim, CUFFT_R2C, &workSizeR2C);   // make plan, retrieve needed workSize
                            if (cuFFTErr != CUFFT_SUCCESS) {
                                std::cerr << "cufftMakePlan3d() r2c failed. Error code: " << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;
                                cudaMemGetInfo(&GPUfree, &GPUtotal);
                                std::cerr << "GPU " << GPUfree / (1024 * 1024) << " MB free / " << GPUtotal / (1024 * 1024) << " MB total. " << std::endl;

                                cuFFTErr = cufftEstimate3d(new_nz, new_ny, deskewedXdim, CUFFT_R2C, &workSizeR2C);
                                if (cuFFTErr != CUFFT_SUCCESS)
                                    std::cerr << "cufftEstimate3d() failed. " << new_nz << " x " << new_ny << " x " << deskewedXdim << " Error code: " << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;

                                std::cerr << "R2C FFT Plan desires " << workSizeR2C / (1024 * 1024) << " MB. " << std::endl;
                                throw std::runtime_error("cufftMakePlan3d() r2c failed.");
                            }

                            cuFFTErr = cufftCreate(&rfftplanInvGPU);						// create object.
                            if (cuFFTErr != CUFFT_SUCCESS)
                                std::cerr << "cufftCreate(&rfftplanInvGPU) failed. Error code: " << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;

                            cuFFTErr = cufftSetAutoAllocation(rfftplanInvGPU, autoAllocate);
                            if (cuFFTErr != CUFFT_SUCCESS)
                                std::cerr << "cufftSetAutoAllocation(rfftplanInvGPU, autoAllocate) failed. Error code: " << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;

                            cuFFTErr = cufftMakePlan3d(rfftplanInvGPU, new_nz, new_ny, deskewedXdim, CUFFT_C2R, &workSizeC2R);// make plan, get workSize
                            if (cuFFTErr != CUFFT_SUCCESS) {
                                std::cerr << "cufftMakePlan3d() c2r failed. " << new_nz << " x " << new_ny << " x " << deskewedXdim << " Error code: " << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;
                                cudaMemGetInfo(&GPUfree, &GPUtotal);
                                std::cerr << "GPU " << GPUfree / (1024 * 1024) << " MB free / " << GPUtotal / (1024 * 1024) << " MB total. " << std::endl;

                                cuFFTErr = cufftEstimate3d(new_nz, new_ny, deskewedXdim, CUFFT_C2R, &workSizeC2R);
                                if (cuFFTErr != CUFFT_SUCCESS)
                                    std::cerr << "cufftEstimate3d() failed. " << new_nz << " x " << new_ny << " x " << deskewedXdim << " Error code: " << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;

                                std::cerr << "C2R FFT Plan desires " << workSizeC2R / (1024 * 1024) << " MB. " << std::endl;
                                throw std::runtime_error("cufftMakePlan3d() c2r failed.");
                            }

                            if (workSizeC2R > workSizeR2C) {
                                workSize = workSizeC2R;    // set workSize to max of C2R and R2C plans.
                            }
                            else {
                                workSize = workSizeR2C;
                            }

                            if (workArea.getSize() != workSize) {  // do we need to (re)allocate workArea?
                                workArea.resize(workSize);
                                std::cout << "FFTplan workarea allctd.   (";
                                cudaMemGetInfo(&GPUfree, &GPUtotal);
                                std::cout << std::setw(4) << workArea.getSize() / (1024 * 1024) << "MB)" << std::setw(7) << GPUfree / (1024 * 1024) << "MB free " << std::endl;
                            }

                            cuFFTErr = cufftSetWorkArea(rfftplanGPU, workArea.getPtr());
                            if (cuFFTErr != CUFFT_SUCCESS) {
                                std::cerr << "cufftSetWorkArea() r2c failed. Error code: " << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;
                                cudaMemGetInfo(&GPUfree, &GPUtotal);
                                std::cerr << "GPU " << GPUfree / (1024 * 1024) << " MB free / " << GPUtotal / (1024 * 1024) << " MB total. " << std::endl;

                                cufftEstimate3d(new_nz, new_ny, deskewedXdim, CUFFT_R2C, &workSizeR2C);
                                std::cerr << "R2C FFT Plan desires " << workSizeR2C / (1024 * 1024) << " MB. " << std::endl;
                                throw std::runtime_error("cufftSetWorkArea() r2c failed.");
                            }

                            cuFFTErr = cufftSetWorkArea(rfftplanInvGPU, workArea.getPtr());
                            if (cuFFTErr != CUFFT_SUCCESS) {
                                std::cerr << "cufftSetWorkArea() c2r failed. Error code: " << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;
                                cudaMemGetInfo(&GPUfree, &GPUtotal);
                                std::cerr << "GPU " << GPUfree / (1024 * 1024) << " MB free / " << GPUtotal / (1024 * 1024) << " MB total. " << std::endl;

                                cufftEstimate3d(new_nz, new_ny, deskewedXdim, CUFFT_C2R, &workSizeC2R);
                                std::cerr << "C2R FFT Plan desires " << workSizeC2R / (1024 * 1024) << " MB. " << std::endl;
                                throw std::runtime_error("cufftSetWorkArea() c2r failed.");
                            }

                        } // end FFT plan allocation method (either normal way or "shared" plan way)

                        std::cout << "FFT plans allocated.    ";
                        cudaMemGetInfo(&GPUfree, &GPUtotal);
                        std::cout << std::setw(8) << (GPUfree_prev - GPUfree) / (1024 * 1024) << "MB" << std::setw(8) << GPUfree / (1024 * 1024)
                                  << "MB free" << " nz=" << new_nz << ", ny=" << new_ny << ", nx=" << deskewedXdim << std::endl;

                    }  //else if (RL_iters>0)

                    //****************Transfer a bunch of constants to device, including OTF array:*******************

                    dkx = 1.0/(imgParams.dr * deskewedXdim);
                    dky = 1.0/(imgParams.dr * new_ny);

                    unsigned nz_const;
                    if (bDupRevStack)
                        nz_const = new_nz*2;
                    else
                        nz_const = new_nz;
                    dkz = 1.0/(imgParams.dz * nz_const);

                    rdistcutoff = 2*NA/(imgParams.wave); // lateral resolution limit in 1/um
                    float eps = std::numeric_limits<float>::epsilon();

                    transferConstants(deskewedXdim, new_ny,
                                      nz_const,
                                      complexOTF.height(), complexOTF.width()/2,
                                      dkx/dkr_otf, dky/dkr_otf, dkz/dkz_otf,
                                      eps, complexOTF.data());

                    d_interpOTF.resize(nz_const * new_ny * (deskewedXdim+2) * sizeof(float)); // allocate memory
                    makeOTFarray(d_interpOTF, deskewedXdim, new_ny, nz_const); // interpolate.  This reads from the complexOTF.data sent to the GPU from transferConstants().
                    std::cout << "d_interpOTF allocated.  ";
                    cudaMemGetInfo(&GPUfree, &GPUtotal);
                    std::cout << std::setw(8) << d_interpOTF.getSize() / (1024 * 1024) << "MB" << std::setw(8) << GPUfree / (1024 * 1024) << "MB free" << std::endl;

                    //****************************Prepare Light Sheet Correction if desired ***********************

                    if (LSfile.size()) {

                        const float my_min = 0.2;

                        std::cout << std::endl << "Loading LS Correction      : ";
                        LSImage.assign(LSfile.c_str());
                        std::cout << LSImage.width() << " x " << LSImage.height() << " x " << LSImage.depth() << ". " << std::endl;
                        AverageLS.resize(LSImage.width(), LSImage.height(), 1, 1, -1); //Set size of AverageLS
                        AverageLS.fill(0); //fill with zeros

                        cimg_forXYZ(LSImage, x, y, z) {
                            AverageLS(x, y) = AverageLS(x, y) + LSImage(x, y, z); //sum image

                            if (z == (LSImage.depth() - 1)) // if this is the last Z slice
                                AverageLS(x, y) = AverageLS(x, y) / LSImage.depth(); //divide by number of slices
                        }

                        const int LSIx = AverageLS.width();
                        const int LSIy = AverageLS.height();

                        int LSIborderx = (LSIx - nx) / 2;
                        int LSIbordery = (LSIy - ny) / 2;

                        std::cout << "Excess LS border to crop   : " << LSIborderx << " x " << LSIbordery << std::endl;

                        AverageLS.crop(LSIborderx, LSIbordery, LSIx - LSIborderx - 1, LSIy - LSIbordery - 1); //crop it

                        cimg_forXY(AverageLS, x, y)
                        AverageLS(x, y) = AverageLS(x, y) - background;


                        img_max = AverageLS(0, 0);
                        img_min = AverageLS(0, 0);
                        cimg_forXY(AverageLS, x, y) {
                            img_max = std::max(AverageLS(x, y), img_max);
                            img_min = std::min(AverageLS(x, y), img_min);
                        }
                        std::cout << "               LS max, min : " << std::setw(8) << img_max << ", " << std::setw(8) << img_min << std::endl;

                        const float normalization = img_max;

                        img_max = -9999999;
                        img_min = 9999999;
                        cimg_forXY(AverageLS, x, y) {
                            AverageLS(x, y) = AverageLS(x, y) / normalization;
                            img_max = std::max(AverageLS(x, y), img_max);
                            img_min = std::min(AverageLS(x, y), img_min);
                        }
                        std::cout << "       After normalization : " << std::setw(8) << img_max << ", " << std::setw(8) << img_min << std::endl;


                        AverageLS.max(my_min); // set everything below 0.2 to 0.2.  This will prevent divide by zero.
                    } // end if (LSfile.size())

                } // end if (nx == 0 || bDifferent_sized_raw_img )

                //****************************Apply Light Sheet Correction if desired ***********************
                if (LSfile.size()) {
                    CImg<float> Temp(raw_image); // Copy raw_image to a floating point image

                    cimg_forXYZ(Temp, x, y, z)
                    Temp(x, y, z) = Temp(x, y, z) - background; // subtract background. min value =0.

                    Temp.max((float)0); // set min value to 0.  have to cast zero to avoid compiler complaint.

                    if (it == all_matching_files.begin()) {
                        img_max = AverageLS(0, 0);
                        img_min = AverageLS(0, 0);

                        cimg_forXY(AverageLS, x, y) {
                            img_max = std::max(AverageLS(x, y), img_max);
                            img_min = std::min(AverageLS(x, y), img_min);
                        }
                        std::cout << "LSC size                   : " << AverageLS.width() << " x " << AverageLS.height() << " x " << AverageLS.depth() << std::endl;
                        std::cout << "LSC max, min               : " << img_max << ", " << img_min << std::endl;
                    }

                    std::cout << "Dividing by LSC... ";

                    Temp.div(AverageLS);
                    std::cout << " Done." << std::endl;

                    cimg_forXYZ(Temp, x, y, z)
                    raw_image(x, y, z) = Temp(x, y, z) + background; //replace background and copy back to U16 raw_image.
                } // end if applying LS correction

				voxel_size[0] = imgParams.dr;
				voxel_size[1] = imgParams.dr;
				voxel_size[2] = imgParams.dz;
                imdz = imgParams.dz;
                if (rotMatrix.getSize()) {
                    imdz = imgParams.dr;
                }
				voxel_size_decon[0] = imgParams.dr;
				voxel_size_decon[1] = imgParams.dr;
				voxel_size_decon[2] = imdz;


                //****************************Pad image.  Use Mirror image in padded border region***********************************

                if (Pad) {
                    border_x = (new_nx - nx) / 2;   // get border size
                    border_y = (new_ny - ny) / 2;
                    border_z = (new_nz - nz) / 2;

                    std::cout << std::endl <<  "Create padded img.  Border : " << border_x << " x " << border_y << " x " << border_z << ". " << std::endl;
                    CImg<> raw_original(raw_image);       //copy from raw image
                    std::cout << "Image with padding size    : " << new_nx << " x " << new_ny << " x " << new_nz << ". ";
                    raw_image.resize(new_nx, new_ny, new_nz);     //resize with border

                    int x_raw;
                    int y_raw;
                    int z_raw;

                    int i_nx = (int)nx;
                    int i_ny = (int)ny;
                    int i_nz = (int)nz;

                    // std::cout << "Line:" << __LINE__ << std::endl;
                    std::cout << "Copy values... ";
                    cimg_forXYZ(raw_image, x, y, z) // for every pixel in the new image, copy value from original image
                    {
                        x_raw = abs(x - border_x);
                        y_raw = abs(y - border_y);
                        z_raw = abs(z - border_z);

                        if (x_raw >= i_nx)
                            x_raw = i_nx - (x_raw - i_nx) - 1;

                        if (y_raw >= i_ny)
                            y_raw = i_ny - (y_raw - i_ny) - 1;

                        if (z_raw >= i_nz)
                            z_raw = i_nz - (z_raw - i_nz) - 1;

                        //raw_image(x, y, z) = x_raw;
                        raw_image(x, y, z) = raw_original(x_raw, y_raw, z_raw);
                    }
                    //***debug padded image.
                    if (false) {
                        std::cout << "Saving padded image... " << std::endl;
                        makeNewDir("Padded");
                        CImg<unsigned short> uint16Img(raw_image);
                        uint16Img.save_tiff(makeOutputFilePath(*it, "Padded", "_padded").c_str(), 0, voxel_size, "");
                    }
                    std::cout << "Done." << std::endl;
                } // End Pad image creation.

                // moved here from boostfs.cpp in order to make it conditional on actually performing decon
                // this MAY cause some bugs ... but haven't seen any since adding a long time ago
                if (RL_iters || rotMatrix.getSize()) {
                    if (it == all_matching_files.begin()) //make directory only on first call, and if we are saving Deskewed.
                        makeNewDir("GPUdecon");
                }

                // initialize the raw_deskewed size everytime in case it is cropped on an earlier iteration
                if (bSaveDeskewedRaw && (fabs(deskewAngle) > 0.0) ) {
                    raw_deskewed.assign(deskewedXdim, new_ny, new_nz);

                    if (it == all_matching_files.begin() + skip) //make directory only on first call, and if we are saving Deskewed.
                        makeNewDir("Deskewed");
                }

                // If deskew is to happen, it'll be performed inside RichardsonLucy_GPU() on GPU;
                // but here raw data's x dimension is still just "new_nx"
                if (bCrop)
                    raw_image.crop(0, 0, 0, 0, new_nx-1, new_ny-1, new_nz-1, 0);

                if (wiener >0) { // plain 1-step Wiener filtering
                    raw_image -= background;
                    fftwf_execute_dft_r2c(rfftplan, raw_image.data(), (fftwf_complex *) raw_imageFFT.data());

                    wienerfilter(raw_imageFFT,
                                 dkx, dky, dkz,
                                 complexOTF,
                                 dkr_otf, dkz_otf,
                                 rdistcutoff, wiener);

                    fftwf_execute_dft_c2r(rfftplan_inv, (fftwf_complex *) raw_imageFFT.data(), raw_image.data());
                    raw_image /= raw_image.size();
                }
                else if (RL_iters || raw_deskewed.size() || rotMatrix.getSize()) {
                    #ifndef NDEBUG
                    img_max = raw_image(0, 0, 0);
                    img_min = raw_image(0, 0, 0);
                    cimg_forXYZ(raw_image, x, y, z) {
                        img_max = std::max(raw_image(x, y, z), img_max);
                        img_min = std::min(raw_image(x, y, z), img_min);
                    }
                    std::cout << "RL_image : " << raw_image.width() << " x " << raw_image.height() << " x " << raw_image.depth() << ". " << std::endl;
                    std::cout << "         RL_image max, min : " << std::setw(8) << img_max << ", " << std::setw(8) << img_min << std::endl;
                    #endif
                    float my_median = 1;
                    if (bFlatStartGuess)
                        my_median = raw_image.median();

                    #ifdef USE_NVTX
                    cudaProfilerStart();
                    #endif

                    //************************************************************************************
                    //****************************Run RL GPU**********************************************
                    //************************************************************************************
                    RichardsonLucy_GPU(raw_image, background, d_interpOTF, RL_iters, deskewFactor,
                                       deskewedXdim, extraShift, napodize, nZblend, rotMatrix,
                                       rfftplanGPU, rfftplanInvGPU, raw_deskewed, &deviceProp,
                                       bFlatStartGuess, my_median, bDoRescale, padVal, bDupRevStack,
                                       UseOnlyHostMem, myGPUdevice);
                    #ifdef USE_NVTX
                    cudaProfilerStop();
                    #endif
                }
                else {
                    std::cerr << "Nothing is performed\n";
                }

                #ifndef NDEBUG
                img_max = raw_image(0, 0, 0);
                img_min = raw_image(0, 0, 0);
                cimg_forXYZ(raw_image, x, y, z) {
                    img_max = std::max(raw_image(x, y, z), img_max);
                    img_min = std::min(raw_image(x, y, z), img_min);
                }
                std::cout << "output_image : " << raw_image.width() << " x " << raw_image.height() << " x " << raw_image.depth() << ". " << std::endl;
                std::cout << "         output_image max, min : " << std::setw(8) << img_max << ", " << std::setw(8) << img_min << std::endl;
                #endif

                //****************************Stitch tiles***********************************

                if (number_of_tiles > 1) { // are we tiling?

                    if (tile_index == 0) {
                        stitch_image.assign(raw_image.width(), file_image.height(), raw_image.depth()); // initialize destination stitch_image.
                        std::cout << "         stitch_image : " << stitch_image.width() << " x " << stitch_image.height() << " x " << stitch_image.depth() << ". " << std::endl;
                        stitch_image.fill((float)0.0);

                        blend_weight_image.assign(raw_image.width(), file_image.height(), raw_image.depth()); // initialize destination blend_weight_image.
                        blend_weight_image.fill(0);
                    }
                    // int no_zone    = floor((float)tile_overlap / 3) ;  // 1/3 no_zone, 1/3 blend zone, 1/3 no_zone
                    int no_zone    = ceil((tile_overlap - 4) / 2.0) ;  //  no_zone, 4 pixel blend zone, yes_zone
                    int blend_zone = tile_overlap - no_zone - no_zone;

                    std::cout << "           tile_image : " << raw_image.width() << " x " << raw_image.height() << " x " << raw_image.depth() << ". tile_overlap: " << tile_overlap << ". blend_zone: " << blend_zone << ". tile_y_offset:" << tile_y_offset << ". " << std::endl;
                    bool is_first_tile = tile_index == 0;
                    bool is_last_tile = tile_index + 1 >= number_of_tiles;

                    cimg_forXYZ(raw_image, x, y, z) {
                        float blend = 1;

                        if (is_first_tile && y <= tile_overlap)
                            blend = 1;// front overlap region of 1st tile
                        else if (is_last_tile && y >= raw_image.height() - tile_overlap )
                            blend = 1;// back overlap region of last tile
                        else {
                            if (y <= no_zone) // front no_zone?
                                blend = 0;

                            else if(y > no_zone && y <= no_zone + blend_zone) // front blend region?
                                blend = (double)(y - no_zone) / blend_zone;

                            else if (y >= raw_image.height() - no_zone) // back no zone?
                                blend = 0;

                            else blend = 1; // middle zone or back blend zone
                        }

                        if (y + tile_y_offset < stitch_image.height()) { /* are we inside of the destination image?*/

                            if (BlendTileOverlap) { // blend the overlap?
                                if (blend > 0 && blend < 1 && blend_weight_image(x, y + tile_y_offset, z) > 0) { // need to blend?
                                    stitch_image(x, y + tile_y_offset, z) = (raw_image(x, y, z) * blend) + (stitch_image(x, y + tile_y_offset, z) * (1.0 - blend)); // insert into image with blend in overlap region
                                    //stitch_image(x, y + tile_y_offset, z) = blend * 1000;
                                }

                                else if (blend > 0 && blend_weight_image(x, y + tile_y_offset, z) == 0) // just put this pixel into the image
                                    stitch_image(x, y + tile_y_offset, z) = raw_image(x, y, z);

                                else if (blend == 1)
                                    stitch_image(x, y + tile_y_offset, z) = raw_image(x, y, z); // just put this pixel into the image

                                if (blend > 0)
                                    blend_weight_image(x, y + tile_y_offset, z) = 1;
                            }
                            else // overlap is either 1 tile or the other.
                            {
                                bool is_past_front_overlap = (y >= floor((float)tile_overlap / 2.0) || is_first_tile);
                                bool is_before_end_overlap = (y < floor(raw_image.height() - (float)tile_overlap / 2.0) || is_last_tile);

                                if (is_past_front_overlap && is_before_end_overlap)
                                    stitch_image(x, y + tile_y_offset, z) = raw_image(x, y, z); // insert into image directly without blend in overlap region
                            }
                        }

                    } // end loop on pixels
                } // end if (number_of_tiles > 1)
                else
                    stitch_image.assign(raw_image, false); // no tiling, so just do image copy.

            } // end for (int tile_index = 0; ... ) tiling loop

            //****************************Crop***********************************

            //remove padding
            if (Pad)
                stitch_image.crop(
                    border_x, border_y,         	// X, Y
                    border_z, 0,			        // Z, C
                    border_x + nx, border_y + ny,	// X, Y
                    border_z + nz, 0);			    // Z, C

            if (! final_CropTo_boundaries.empty()) {
                stitch_image.crop(final_CropTo_boundaries[0], final_CropTo_boundaries[2],   // X, Y
                                  final_CropTo_boundaries[4], 0,							// Z, C
                                  final_CropTo_boundaries[1], final_CropTo_boundaries[3],	// X, Y
                                  final_CropTo_boundaries[5], 0);							// Z, C
                if (raw_deskewed.size()) {
                    // std::cout << std::endl << "The 'raw_deskewed' buffer uses " << raw_deskewed.size()*sizeof(float) << " bytes." << std::endl;
                    raw_deskewed.crop(final_CropTo_boundaries[0], final_CropTo_boundaries[2],
                                      final_CropTo_boundaries[4], 0,
                                      final_CropTo_boundaries[1], final_CropTo_boundaries[3],
                                      final_CropTo_boundaries[5], 0);
                }
            }

            if (tsave.joinable()) {
                tsave.join();           // wait for previous saving thread to finish.
                tsave.~thread();        // destroy thread.
            }

            if (tDeskewsave.joinable()) {
                tDeskewsave.join();     // wait for previous saving thread to finish.
                tDeskewsave.~thread();  // destroy thread.
            }


            //****************************Save Deskewed Raw***********************************
            if (bSaveDeskewedRaw && (fabs(deskewAngle) > 0.0)) {
				std::string tiff_description = make_Image_Description(raw_deskewed.depth(), voxel_size[2]);

                if (!bSaveUshort) {
                    DeskewedToSave.assign(raw_deskewed);
                    tDeskewsave = std::thread(DeSkewsave_in_thread, *it, voxel_size, tiff_description.c_str()); //start saving "Deskewed To Save" file.                    
                }
                else {
                    CImg<unsigned short> uint16Img(raw_deskewed); // convert to U16 then save.
                    uint16Img.save_tiff(makeOutputFilePath(*it, "Deskewed", "_deskewed").c_str(), compression, voxel_size, tiff_description.c_str());
                }
            }


            //****************************Save Deskewed MIPs***********************************
            if (bDoRawMaxIntProj.size() == 3 && bSaveDeskewedRaw) {
                if(it == all_matching_files.begin() && (bDoRawMaxIntProj[0] || bDoRawMaxIntProj[1] || bDoRawMaxIntProj[2]))
                    makeNewDir("Deskewed/MIPs");

				float MIPvoxel_size_YZ[3] = { voxel_size[1], voxel_size[2], voxel_size[0] }; // Y,Z,X voxel
				float MIPvoxel_size_XZ[3] = { voxel_size[0], voxel_size[2], voxel_size[1] }; // X,Z,Y voxel
				float MIPvoxel_size_XY[3] = { voxel_size[0], voxel_size[1], voxel_size[2] }; // X,Y,Z voxel
				
                if (bDoRawMaxIntProj[0]) {
                    CImg<> proj = MaxIntProj(raw_deskewed, 0); // YZ projection
					std::string tiff_description = make_Image_Description(1, MIPvoxel_size_YZ[2]); // YZ projection

					proj.save_tiff(makeOutputFilePath(*it, "Deskewed/MIPs", "_MIP_x").c_str(), compression, MIPvoxel_size_YZ, tiff_description.c_str());
                }
                if (bDoRawMaxIntProj[1]) {
                    CImg<> proj = MaxIntProj(raw_deskewed, 1); // XZ projection
					std::string tiff_description = make_Image_Description(1, MIPvoxel_size_XZ[2]); // XZ projection

					proj.save_tiff(makeOutputFilePath(*it, "Deskewed/MIPs", "_MIP_y").c_str(), compression, MIPvoxel_size_XZ, tiff_description.c_str());
                }
                if (bDoRawMaxIntProj[2]) {
                    CImg<> proj = MaxIntProj(raw_deskewed, 2); // XY projection
					std::string tiff_description = make_Image_Description(1, MIPvoxel_size_XY[2]); // XY projection

                    proj.save_tiff(makeOutputFilePath(*it, "Deskewed/MIPs", "_MIP_z").c_str(), compression, MIPvoxel_size_XY, tiff_description.c_str());
                }

            }

            //****************************Save decon MIPs***********************************

            // I had to modify this a bit to save the behavior of --saveDeskewedRaw when NO RL is being performed...
            // otherwise, it was trying to create folders it shouldn't...
            // this might have unintended side effects (notably with weiner filtering only... and maybe others)
            if (bDoMaxIntProj.size() == 3 && RL_iters && (bDoMaxIntProj[0] || bDoMaxIntProj[1] || bDoMaxIntProj[2])) {
                if (it == all_matching_files.begin() + skip) {
                    makeNewDir("GPUdecon/MIPs");
                }

				float MIPvoxel_size_YZ[3] = { voxel_size_decon[1], voxel_size_decon[2], voxel_size_decon[0] }; // Y,Z,X decon voxel
				float MIPvoxel_size_XZ[3] = { voxel_size_decon[0], voxel_size_decon[2], voxel_size_decon[1] }; // X,Z,Y decon voxel
				float MIPvoxel_size_XY[3] = { voxel_size_decon[0], voxel_size_decon[1], voxel_size_decon[2] }; // X,Y,Z decon voxel

				
                if (bDoMaxIntProj[0]) {
                    CImg<> proj = MaxIntProj(stitch_image, 0); // YZ projection
					std::string tiff_description = make_Image_Description(1, MIPvoxel_size_YZ[2]); // YZ projection
                    if (bSaveUshort) {
                        CImg<unsigned short> uint16Img(proj);
                        uint16Img.save_tiff(makeOutputFilePath(*it, "GPUdecon/MIPs", "_MIP_x").c_str(), compression, MIPvoxel_size_YZ, tiff_description.c_str());
                    }
                    else
						proj.save_tiff(     makeOutputFilePath(*it, "GPUdecon/MIPs", "_MIP_x").c_str(), compression, MIPvoxel_size_YZ, tiff_description.c_str());
                    
                }
                if (bDoMaxIntProj[1]) {
                    CImg<> proj = MaxIntProj(stitch_image, 1); // XZ projection
					std::string tiff_description = make_Image_Description(1, MIPvoxel_size_XZ[2]); // XZ projection
                    if (bSaveUshort) {
                        CImg<unsigned short> uint16Img(proj);
                        uint16Img.save_tiff(makeOutputFilePath(*it, "GPUdecon/MIPs", "_MIP_y").c_str(), compression, MIPvoxel_size_XZ, tiff_description.c_str());
                    }
                    else
						proj.save_tiff(     makeOutputFilePath(*it, "GPUdecon/MIPs", "_MIP_y").c_str(), compression, MIPvoxel_size_XZ, tiff_description.c_str());
                    
                }
                if (bDoMaxIntProj[2]) {
                    CImg<> proj = MaxIntProj(stitch_image, 2); // XY projection
					std::string tiff_description = make_Image_Description(1, MIPvoxel_size_XY[2]); // XY projection
                    if (bSaveUshort) {
                        CImg<unsigned short> uint16Img(proj);
                        uint16Img.save_tiff(makeOutputFilePath(*it, "GPUdecon/MIPs", "_MIP_z").c_str(), compression, MIPvoxel_size_XY, tiff_description.c_str());
                    }
                    else
                        proj.save_tiff(     makeOutputFilePath(*it, "GPUdecon/MIPs", "_MIP_z").c_str(), compression, MIPvoxel_size_XY, tiff_description.c_str());
                    
                }
            }

            //****************************Save Decon Image***********************************
            if (RL_iters || rotMatrix.getSize()) {
                // Stupid to redefine these here... but couldn't get the Z voxel size to work
                // correctly in ImageJ otherwise...

                if (!bSaveUshort) {

                    ToSave.assign(stitch_image); //copy decon image (i.e. stitch_image) to new image space, ToSave, for saving.                  

                    tsave = std::thread(save_in_thread, *it, voxel_size_decon, imdz); //start saving float "ToSave" file.
                }
                else {
                    U16ToSave = stitch_image;
                    tsave = std::thread(U16save_in_thread, *it, voxel_size_decon, imdz); //start saving U16 "U16ToSave" file.
                }
            }

            /// please leave this here for LLSpy
            printf(">>>file_finished\n");

            iter_duration = (std::clock() - start_t) / (double)CLOCKS_PER_SEC / (it - (all_matching_files.begin() + skip) + 1 );
        } // end for (std::vector<std::string>::iterator it= all_matching_files.begin() + skip;

        if (tsave.joinable()) {
            tsave.join();               // wait for previous saving thread to finish.
            tsave.~thread();            // destroy thread.
        } //Make sure we have finished saving.

        if (tDeskewsave.joinable()) {
            tDeskewsave.join();         // wait for previous saving thread to finish.
            tDeskewsave.~thread();      // destroy thread.
        } //Make sure we have finished saving.

        duration = (std::clock() - start_t) / (double)CLOCKS_PER_SEC;

        #ifdef _WIN32
        SetConsoleTextAttribute(hConsole, 10); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
        #endif
        std::cout << "*** Finished! Elapsed " << duration << " seconds.  ";
        if (skip != 0)
            std::cout << "Skipped " << skip << "images.  ";
        std::cout << "Processed " << all_matching_files.size() - skip << " images.  " << duration / all_matching_files.size() << " seconds per image. ***" << std::endl;
        #ifdef _WIN32
        SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
        #endif

    } // end try {} block
    catch (std::exception &e) {
        std::cerr << "\n!!Error occurred: " << e.what() << std::endl;
        return 0;
    }
    return 0;
}


CImg<> MaxIntProj(CImg<> &input, int axis)
{
    CImg <> out;

    if (axis==0) {
        CImg<> maxvals(input.height(), input.depth());
        maxvals = -1e10;
        #pragma omp parallel for
        cimg_forYZ(input, y, z) for (int x=0; x<input.width(); x++) {
            if (input(x, y, z) > maxvals(y, z))
                maxvals(y, z) = input(x, y, z);
        }
        return maxvals;
    }
    else if (axis==1) {
        CImg<> maxvals(input.width(), input.depth());
        maxvals = -1e10;
        #pragma omp parallel for
        cimg_forXZ(input, x, z) for (int y=0; y<input.height(); y++) {
            if (input(x, y, z) > maxvals(x, z))
                maxvals(x, z) = input(x, y, z);
        }
        return maxvals;
    }

    else if (axis==2) {
        CImg<> maxvals(input.width(), input.height());
        maxvals = -1e10;
        #pragma omp parallel for
        cimg_forXY(input, x, y) for (int z=0; z<input.depth(); z++) {
            if (input(x, y, z) > maxvals(x, y))
                maxvals(x, y) = input(x, y, z);
        }
        return maxvals;
    }
    else {
        throw std::runtime_error("unknown axis number in MaxIntProj()");
    }
    return out;
}

