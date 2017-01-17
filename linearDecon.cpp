#include "linearDecon.h"
#include <exception>
#include <ctime>



// Disable silly warnings on some Microsoft VC++ compilers.
#pragma warning(disable : 4244) // Disregard loss of data from float to int.
#pragma warning(disable : 4267) // Disregard loss of data from size_t to unsigned int.
#pragma warning(disable : 4305) // Disregard loss of data from double to float.

CImg<> next_raw_image;
CImg<> ToSave;

int load_next_thread(const char* my_path)
{
	next_raw_image.assign(my_path);
	if (false)
	{
		float img_max = next_raw_image(0, 0, 0);
		float img_min = next_raw_image(0, 0, 0);
		cimg_forXYZ(next_raw_image, x, y, z){
			img_max = std::max(next_raw_image(x, y, z), img_max);
			img_min = std::min(next_raw_image(x, y, z), img_min);
		}
		std::cout << "next_raw_image : " << next_raw_image.width() << " x " << next_raw_image.height() << " x " << next_raw_image.depth() << ". " << std::endl;
		std::cout << "         next_raw_image max, min : " << std::setw(8) << img_max << ", " << std::setw(8) << img_min << std::endl;
		std::cout << "Loaded from " << my_path << std::endl;
	}

	return 0;
}


int save_in_thread(std::string inputFileName)
{
	ToSave.save(makeOutputFilePath(inputFileName).c_str());

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
  /* 'g' is the raw data's FFT (half kx axis); 
     it is also the result upon return */
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
	std::clock_t start_t;
	double duration;
	double iter_duration = 0;

	HANDLE  hConsole;
	hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	start_t = std::clock();

  int napodize, nZblend;
  float background;
  float NA=1.2;
  ImgParams imgParams;
  float dz_psf, dr_psf;
  float wiener;

  int myGPUdevice;
  int RL_iters=0;
  bool bSaveDeskewedRaw = false;
  bool bDontAdjustResolution = false;
  bool bDevQuery = false;
  float deskewAngle=0.0;
  float rotationAngle=0.0;
  unsigned outputWidth;
  int extraShift=0;
  std::vector<int> final_CropTo_boundaries;
  bool bSaveUshort = false;
  std::vector<bool> bDoMaxIntProj;
  std::vector< CImg<> > MIprojections;
  int Pad = 0;
  bool bFlatStartGuess = false;
  bool No_Bleach_correction = false;
  bool UseOnlyHostMem = false;
  int skip = 0;

  TIFFSetWarningHandler(NULL);

  std::string datafolder, datafolderB, filenamePattern, filenamePatternB, otffiles, otffilesB, LSfile;
  po::options_description progopts;
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
	  ("NA,n", po::value<float>(&NA)->default_value(1.2), "Numerical aperture")
	  ("RL,i", po::value<int>(&RL_iters)->default_value(15), "Run Richardson-Lucy, and set how many iterations")
	  ("deskew,D", po::value<float>(&deskewAngle)->default_value(0.0), "Deskew angle; if not 0.0 then perform deskewing before deconv")
	  ("width,w", po::value<unsigned>(&outputWidth)->default_value(0), "If deskewed, the output image's width")
	  ("shift,x", po::value<int>(&extraShift)->default_value(0), "If deskewed, the output image's extra shift in X (positive->left")
	  ("rotate,R", po::value<float>(&rotationAngle)->default_value(0.0), "Rotation angle; if not 0.0 then perform rotation around y axis after deconv")
	  ("saveDeskewedRaw,S", po::bool_switch(&bSaveDeskewedRaw)->default_value(false), "Save deskewed raw data to files")
	  // ("crop,C", po::value< std::vector<int> >(&final_CropTo_boundaries)->multitoken(), "takes 6 integers separated by space: x1 x2 y1 y2 z1 z2; crop final image size to [x1:x2, y1:y2, z1:z2]")
	  // ("MIP,M", po::value< std::vector<bool> >(&bDoMaxIntProj)->multitoken(), "takes 3 binary numbers separated by space to indicate whether save a max-intensity projection along x, y, or z axis")
	  ("crop,C", fixed_tokens_value< std::vector<int> >(&final_CropTo_boundaries, 6, 6), "Crop final image size to [x1:x2, y1:y2, z1:z2]; takes 6 integers separated by space: x1 x2 y1 y2 z1 z2; ")
	  ("MIP,M", fixed_tokens_value< std::vector<bool> >(&bDoMaxIntProj, 3, 3), "Save max-intensity projection along x, y, or z axis; takes 3 binary numbers separated by space: 0 0 1")
    ("uint16,u", po::bool_switch(&bSaveUshort)->implicit_value(true), "Save result in uint16 format; should be used only if no actual decon is performed")
    ("input-dir", po::value<std::string>(&datafolder)->required(), "Folder of input images")
	("input-dirB", po::value<std::string>(&datafolderB), "Folder of input (second view) B images")
    ("otf-file", po::value<std::string>(&otffiles)->required(), "OTF file")
	("otf-fileB", po::value<std::string>(&otffilesB), "OTF B (second view) file")
    ("filename-pattern", po::value<std::string>(&filenamePattern)->required(), "File name pattern to find input images to process")
	("filename-patternB", po::value<std::string>(&filenamePatternB), "File name pattern to find input (second view) B images to process")
	("DoNotAdjustResForFFT,a", po::bool_switch(&bDontAdjustResolution)->default_value(false), "Don't change data resolution size. Otherwise data is cropped to perform faster, more memory efficient FFT: size factorable into 2,3,5,7)")
	("DevQuery,q", po::bool_switch(&bDevQuery)->default_value(false), "Show info and indices of available GPUs")
	("GPUdevice", po::value<int>(&myGPUdevice)->default_value(0), "Index of GPU device to use (0=first device)")
	("Pad", po::value<int>(&Pad)->default_value(0), "Pad the image data with mirrored values to avoid edge artifacts")
	("LSC", po::value<std::string>(&LSfile), "Lightsheet correction file")
	("FlatStart", po::bool_switch(&bFlatStartGuess)->default_value(false), "Start the RL from a guess that is a flat image filled with the median image value.  This may supress noise.")
	("NoBleachCorrection", po::bool_switch(&No_Bleach_correction)->default_value(false), "Does not apply bleach correction when running multiple images in a single batch.")
	("skip", po::value<int>(&skip)->default_value(0), "Skip the first 'skip' number of files.")
	// ("UseOnlyHostMem", po::bool_switch(&UseOnlyHostMem)->default_value(false), "Just use Host Mapped Memory, and not GPU. For debugging only.")
    ("help,h", "This help message.")
    ;
  po::positional_options_description p;
  p.add("input-dir", 1);
  p.add("filename-pattern", 1);
  p.add("otf-file", 1);

  
  std::string commandline_string = __DATE__ ;
  commandline_string.append(" ");
  commandline_string.append(__TIME__);
  for (int i = 0; i < argc; i++){
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
  
  //****************************Query GPU devices***********************************
  if (bDevQuery) {
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
	  cudaSetDevice(myGPUdevice);
  }


  cudaSetDeviceFlags(cudaDeviceMapHost);
  size_t GPUfree;
  size_t GPUtotal;
  cudaMemGetInfo(&GPUfree, &GPUtotal);
  cudaDeviceProp mydeviceProp;
  cudaGetDeviceProperties(&mydeviceProp, myGPUdevice);

  SetConsoleTextAttribute(hConsole, 13); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
  std::cout << std::endl << "Built : " << __DATE__ << " " << __TIME__ << ".  GPU " << GPUfree / (1024 * 1024) << " MB free / " << GPUtotal / (1024 * 1024) << " MB total on " << mydeviceProp.name << std::endl;

  CImg<> raw_image, raw_imageFFT, complexOTF, raw_deskewed, LSImage;
  CImg<float> AverageLS;
  float dkr_otf, dkz_otf;
  float dkx, dky, dkz, rdistcutoff;
  fftwf_plan rfftplan=NULL, rfftplan_inv=NULL;
  CPUBuffer rotMatrix;
  double deskewFactor=0;
  bool bCrop = false;
  unsigned new_ny, new_nz, new_nx;
  int deskewedXdim = 0;
  cufftHandle rfftplanGPU, rfftplanInvGPU;
  GPUBuffer d_interpOTF(0, myGPUdevice, UseOnlyHostMem);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, myGPUdevice);

  unsigned nx, ny, nz;

  int border_x = 0;
  int border_y = 0;
  int border_z = 0;

  //****************************Main processing***********************************
  // Loop over all matching input TIFFs, :
  try {
	

	std::cout << "Looking for files to process... " ;
    // Gather all files in 'datafolder' and matching the file name pattern:
    std::vector< std::string > all_matching_files = gatherMatchingFiles(datafolder, filenamePattern);

	std::cout << "Found " << all_matching_files.size() << " file(s)." << std::endl ;
	SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c



	std::vector<std::string>::iterator next_it = all_matching_files.begin() + skip;//make a second incrementer that will be used to load the next raw image while we process.
	
	std::thread t1;
	t1 = std::thread(load_next_thread, next_it->c_str());						//start loading the first file.
	std::thread tsave;	// make thread for saving decon
	
    for (std::vector<std::string>::iterator it= all_matching_files.begin() + skip;
         it != all_matching_files.end(); it++) {
		
	  
	  int number_of_files_left = all_matching_files.end() - it;

	  
	  SetConsoleTextAttribute(hConsole, 10); // colors are 9=blue 10=green and so on to 15=white 
	  std::cout << std::endl << "Loading raw_image: " << it - all_matching_files.begin() + 1 << " out of " << all_matching_files.size() << ".   ";
	  if (it > all_matching_files.begin() + skip) // if this isn't the first iteration.
	  {
		  int seconds = number_of_files_left * iter_duration;
		  int hours = ceil(seconds / (60 * 60));
		  int minutes = ceil((seconds - (hours * 60 * 60)) / 60);
		  std::cout << (int)iter_duration << " s/file.   " << number_of_files_left << " files left.  " << hours << " hours, " <<  minutes << " minutes remaining.";
	  }
	    
	  std::cout <<	std::endl;
	  SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c


	  std::cout << std::endl << *it << std::endl;
	  std::cout << "Waiting for separate thread to finish loading image... " ;
	  t1.join();		// wait for loading thread to finish reading next_raw_image into memory.
	  t1.~thread();		// destroy thread.

	  std::cout << "Image loaded. Copying to raw... " ;
	  raw_image.assign(next_raw_image); // Copy to raw_image. 
	  
	  float img_max = raw_image(0, 0);
	  float img_min = raw_image(0, 0);

#ifndef NDEBUG
	  cimg_forXYZ(raw_image, x, y, z){
		  img_max = std::max(raw_image(x, y, z), img_max);
		  img_min = std::min(raw_image(x, y, z), img_min);
	  }

	  std::cout << "         raw img max, min : " << std::setw(8) << img_max << ", " << std::setw(8) << img_min << std::endl;
#endif



	//std::swap(raw_image, next_raw_image); // Swap pointers.
	std::cout << "Done." << std::endl;

	// start reading the next image
	next_it++; // increment next_it.  This will now have the next file to read.
	if (it < all_matching_files.end() - 1)  // If there are more files to process...
		t1 = std::thread(load_next_thread, next_it->c_str());		//start new thread and load the next file, while we process raw_image
	  

      // If it's the first input file, initialize a bunch including:
      // 1. crop image to make dimensions nice factorizable numbers
      // 2. calculate deskew parameters, new X dimensions
      // 3. calculate rotation matrix
      // 4. create FFT plans
      // 5. transfer constants into GPU device constant memory
      // 6. make 3D OTF array in device memory

	bool bDifferent_sized_raw_img = (nx != raw_image.width() || ny != raw_image.height() || nz != raw_image.depth() ); // Check if raw.image has changed size from first iteration

	if (it != all_matching_files.begin() + skip && bDifferent_sized_raw_img) {
		SetConsoleTextAttribute(hConsole, 14); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
		std::cout << std::endl << "File " << it - all_matching_files.begin() + 1 << " has a different size" << std::endl;
	}
	std::cout << "raw_image size             : " << raw_image.width() << " x " << raw_image.height() << " x " << raw_image.depth() << std::endl;
	SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
				

	  
	  //**************************** If first image OR if raw_img has changed size ***********************************
      if (it == all_matching_files.begin() + skip || bDifferent_sized_raw_img ) {
        nx = raw_image.width();
        ny = raw_image.height();
        nz = raw_image.depth();

		

			
		//****************************Adjust resolution if desired***********************************
		
		int step_size;
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
		unsigned nr_otf = complexOTF.height();
        unsigned nz_otf = complexOTF.width() / 2;
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
			  SetConsoleTextAttribute(hConsole, 14); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
			  printf("Warning : deskewFactor is < 1.  Check that angle, dz, and dr sizes are correct.\n");
			  SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
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

			if (cufftGetSize(rfftplanGPU, &workSize) == CUFFT_SUCCESS){ // if plan existed before, destroy it before creating a new plan.
				cufftDestroy(rfftplanGPU);
				std::cout << "Destroying rfftplanGPU." << std::endl;
			}
			if (cufftGetSize(rfftplanInvGPU, &workSize) == CUFFT_SUCCESS){ // if plan existed before, destroy it before creating a new plan.
				cufftDestroy(rfftplanInvGPU);
				std::cout << "Destroying rfftplanInvGPU." << std::endl;
			}

			size_t GPUfree_prev;
			cudaMemGetInfo(&GPUfree_prev, &GPUtotal);

			
		  cuFFTErr = cufftPlan3d(&rfftplanGPU, new_nz, new_ny, deskewedXdim, CUFFT_R2C);
          if (cuFFTErr != CUFFT_SUCCESS) {
				std::cerr << "cufftPlan3d() r2c failed. Error code: " << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;
				cudaMemGetInfo(&GPUfree, &GPUtotal);
				std::cerr << "GPU " << GPUfree / (1024 * 1024) << " MB free / " << GPUtotal / (1024 * 1024) << " MB total. " << std::endl;
				
				cufftEstimate3d(new_nz, new_ny, deskewedXdim, CUFFT_R2C, &workSize);
				std::cerr << "R2C FFT Plan desires " << workSize / (1024 * 1024) << " MB. " << std::endl;
				throw std::runtime_error("cufftPlan3d() r2c failed.");
          }



          cuFFTErr = cufftPlan3d(&rfftplanInvGPU, new_nz, new_ny, deskewedXdim, CUFFT_C2R);
          if (cuFFTErr != CUFFT_SUCCESS) {
			  std::cerr << "cufftPlan3d() c2r failed. Error code: " << cuFFTErr << " : " << _cudaGetErrorEnum(cuFFTErr) << std::endl;
			cudaMemGetInfo(&GPUfree, &GPUtotal);
			std::cerr << "GPU " << GPUfree / (1024 * 1024) << " MB free / " << GPUtotal / (1024 * 1024) << " MB total. " << std::endl;

			cufftEstimate3d(new_nz, new_ny, deskewedXdim, CUFFT_R2C, &workSize);
			std::cerr << "C2R FFT Plan desires " << workSize / (1024 * 1024) << " MB. " << std::endl;
			throw std::runtime_error("cufftPlan3d() c2r failed.");
          }
		  std::cout << "FFT plans allocated.    ";
		  cudaMemGetInfo(&GPUfree, &GPUtotal);
		  std::cout << std::setw(8) << (GPUfree_prev - GPUfree) / (1024 * 1024) << "MB" << std::setw(8) << GPUfree / (1024 * 1024) << "MB free" << std::endl;

        }


		//****************************Transfer a bunch of constants to device, including OTF array:***********************************
		// 
		dkx = 1.0/(imgParams.dr * deskewedXdim);
        dky = 1.0/(imgParams.dr * new_ny);
        dkz = 1.0/(imgParams.dz * new_nz);
        rdistcutoff = 2*NA/(imgParams.wave); // lateral resolution limit in 1/um
		float eps = std::numeric_limits<float>::epsilon();

        transferConstants(deskewedXdim, new_ny, new_nz,
                          complexOTF.height(), complexOTF.width()/2,
                          dkx/dkr_otf, dky/dkr_otf, dkz/dkz_otf,
                          eps, complexOTF.data());

        // make a 3D interpolated OTF array on GPU:
        d_interpOTF.resize(new_nz * new_ny * (deskewedXdim+2) * sizeof(float)); // allocate memory
        makeOTFarray(d_interpOTF, deskewedXdim, new_ny, new_nz);				// interpolate
		std::cout << "d_interpOTF allocated.  ";
		cudaMemGetInfo(&GPUfree, &GPUtotal);
		std::cout << std::setw(8) << d_interpOTF.getSize() / (1024 * 1024) << "MB" << std::setw(8) << GPUfree / (1024 * 1024) << "MB free" << std::endl;

		//****************************Prepare Light Sheet Correction if desired ***********************
		if (LSfile.size()){

			const float my_min = 0.2;

			std::cout << std::endl << "Loading LS Correction      : ";
			LSImage.assign(LSfile.c_str());
			std::cout << LSImage.width() << " x " << LSImage.height() << " x " << LSImage.depth() << ". " << std::endl;
			AverageLS.resize(LSImage.width(), LSImage.height(), 1, 1, -1); //Set size of AverageLS
			AverageLS.fill(0); //fill with zeros


			cimg_forXYZ(LSImage, x, y, z){
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
			cimg_forXY(AverageLS, x, y){
				img_max = std::max(AverageLS(x, y), img_max);
				img_min = std::min(AverageLS(x, y), img_min);
			}
			std::cout << "               LS max, min : " << std::setw(8) << img_max << ", " << std::setw(8) << img_min << std::endl;


			const float normalization = img_max;

			img_max = -9999999;
			img_min = 9999999;
			cimg_forXY(AverageLS, x, y){
				AverageLS(x, y) = AverageLS(x, y) / normalization;
				img_max = std::max(AverageLS(x, y), img_max);
				img_min = std::min(AverageLS(x, y), img_min);
			}
			std::cout << "       After normalization : " << std::setw(8) << img_max << ", " << std::setw(8) << img_min << std::endl;


			AverageLS.max(my_min); // set everything below 0.2 to 0.2.  This will prevent divide by zero.
		} // end if prepare LS correction




      } // end "if this is the first iteration" (it == all_matching_files.begin())


	  //****************************Apply Light Sheet Correction if desired ***********************
	  if (LSfile.size()){
		  CImg<float> Temp(raw_image); // Copy raw_image to a floating point image

		  cimg_forXYZ(Temp, x, y, z)
			  Temp(x, y, z) = Temp(x, y, z) - background; // subtract background. min value =0.

		  Temp.max((float)0); // set min value to 0.  have to cast zero to avoid compiler complaint.

		  if (it == all_matching_files.begin()){
			  img_max = AverageLS(0, 0);
			  img_min = AverageLS(0, 0);

			  cimg_forXY(AverageLS, x, y){
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


	  //****************************Pad image.  Use Mirror image in padded border region***********************************

	  if (Pad){
		  border_x = (new_nx - nx) / 2;   // get border size
		  border_y = (new_ny - ny) / 2;
		  border_z = (new_nz - nz) / 2;

		  std::cout << std::endl <<  "Create padded img.  Border : " << border_x << " x " << border_y << " x " << border_z << ". " << std::endl;
		  CImg<> raw_original(raw_image);		//copy from raw image
		  std::cout << "Image with padding size    : " << new_nx << " x " << new_ny << " x " << new_nz << ". ";
		  raw_image.resize(new_nx, new_ny, new_nz);		//resize with border

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
		  if (false){
			  std::cout << "Saving padded image... " << std::endl;
			  makeDeskewedDir("Padded");
			  CImg<unsigned short> uint16Img(raw_image);
			  uint16Img.SetDescription(commandline_string);
			  uint16Img.save(makeOutputFilePath(*it, "Padded", "_padded").c_str());
		  }
		  std::cout << "Done." << std::endl;
	  } // End Pad image creation.


	  // initialize the raw_deskewed size everytime in case it is cropped on an earlier iteration
	  if (bSaveDeskewedRaw && (fabs(deskewAngle) > 0.0) ) {
		  raw_deskewed.assign(deskewedXdim, new_ny, new_nz);

		  if (it == all_matching_files.begin() + skip) //make directory only on first call, and if we are saving Deskewed.
			   makeDeskewedDir("Deskewed");
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
		  cimg_forXYZ(raw_image, x, y, z){
			  img_max = std::max(raw_image(x, y, z), img_max);
			  img_min = std::min(raw_image(x, y, z), img_min);
		  }
		  std::cout << "RL_image : " << raw_image.width() << " x " << raw_image.height() << " x " << raw_image.depth() << ". " << std::endl;
		  std::cout << "         RL_image max, min : " << std::setw(8) << img_max << ", " << std::setw(8) << img_min << std::endl;
#endif
		  float my_median = 1;
		  if (bFlatStartGuess)
			my_median = raw_image.median();

		  //**********************************************************************************************************
		  //****************************Run RL GPU********************************************************************
		  //**********************************************************************************************************
        RichardsonLucy_GPU(raw_image, background, d_interpOTF, RL_iters, deskewFactor,
                           deskewedXdim, extraShift, napodize, nZblend, rotMatrix,
						   rfftplanGPU, rfftplanInvGPU, raw_deskewed, &deviceProp, myGPUdevice, 
						   bFlatStartGuess, my_median, No_Bleach_correction, UseOnlyHostMem);
      }
      else {
        std::cerr << "Nothing is performed\n";
        break;
      }
	  
#ifndef NDEBUG
	  img_max = raw_image(0, 0, 0);
	  img_min = raw_image(0, 0, 0);
	  cimg_forXYZ(raw_image, x, y, z){
		  img_max = std::max(raw_image(x, y, z), img_max);
		  img_min = std::min(raw_image(x, y, z), img_min);
	  }
	  std::cout << "output_image : " << raw_image.width() << " x " << raw_image.height() << " x " << raw_image.depth() << ". " << std::endl;
	  std::cout << "         output_image max, min : " << std::setw(8) << img_max << ", " << std::setw(8) << img_min << std::endl;
#endif

	  //****************************Crop***********************************

	  //remove padding
	  if (Pad)
		  raw_image.crop(
			  border_x, border_y,	// X, Y
			  border_z, 0,			// Z, C
			  border_x + nx, border_y + ny,	// X, Y
			  border_z + nz, 0);			// Z, C
				
	  if (! final_CropTo_boundaries.empty()) {
        raw_image.crop(final_CropTo_boundaries[0], final_CropTo_boundaries[2],  // X, Y
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



	  if (tsave.joinable()){
		  tsave.join();		// wait for previous saving thread to finish.
		  tsave.~thread();		// destroy thread.
	  }


	  //****************************Save Deskewed Raw***********************************
      if (bSaveDeskewedRaw) {
		  if (!bSaveUshort){
			  raw_deskewed.SetDescription(commandline_string);
			  raw_deskewed.save(makeOutputFilePath(*it, "Deskewed", "_deskewed").c_str());
		  }
        else {
          CImg<unsigned short> uint16Img(raw_deskewed);
		  uint16Img.SetDescription(commandline_string);
          uint16Img.save(makeOutputFilePath(*it, "Deskewed", "_deskewed").c_str());
         }
      }

	  
	  //****************************Save MIPs***********************************
      if (it == all_matching_files.begin() + skip && bDoMaxIntProj.size())
        makeDeskewedDir("GPUdecon/MIPs");

      if (bDoMaxIntProj.size() == 3) {
        if (bDoMaxIntProj[0]) {
          CImg<> proj = MaxIntProj(raw_image, 0);
		  proj.SetDescription(commandline_string);
		  proj.save(makeOutputFilePath(*it, "GPUdecon/MIPs", "_MIP_x").c_str());
        }
        if (bDoMaxIntProj[1]) {
          CImg<> proj = MaxIntProj(raw_image, 1);
		  proj.save(makeOutputFilePath(*it, "GPUdecon/MIPs", "_MIP_y").c_str());
        }
        if (bDoMaxIntProj[2]) {
          CImg<> proj = MaxIntProj(raw_image, 2);
		  proj.SetDescription(commandline_string);		 
          proj.save(makeOutputFilePath(*it, "GPUdecon/MIPs", "_MIP_z").c_str());
        }
      }

	  //****************************Save Decon Image***********************************
	  if (RL_iters || rotMatrix.getSize()) {
		  if (!bSaveUshort){

			  ToSave.assign(raw_image); //copy decon image (i.e. raw_image) to new image space for saving.
			  ToSave.SetDescription(commandline_string);
			  // ToSave.save(makeOutputFilePath(*it).c_str());

			  tsave = std::thread(save_in_thread, *it); //start saving "To Save" file.
		  }
		  else {
			  CImg<unsigned short> uint16Img(raw_image);
			  uint16Img.SetDescription(commandline_string);
			  uint16Img.save(makeOutputFilePath(*it).c_str());
		  }
	  }


	  iter_duration = (std::clock() - start_t) / (double)CLOCKS_PER_SEC / (it - (all_matching_files.begin() + skip) + 1 );
    } // iteration over all_matching_files

	if (tsave.joinable()){
		tsave.join();		// wait for previous saving thread to finish.
		tsave.~thread();	// destroy thread.
	} //Make sure we have finished saving.
	


	duration = (std::clock() - start_t) / (double)CLOCKS_PER_SEC;

	SetConsoleTextAttribute(hConsole, 10); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
	std::cout << "*** Finished! Elapsed " << duration << " seconds.  ";
	if (skip != 0)
		std::cout << "Skipped " << skip << "images.  ";
	std::cout << "Processed " << all_matching_files.size() - skip << " images.  " << duration / all_matching_files.size() << " seconds per image. ***" << std::endl;
	SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c

  } // try {} block
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
