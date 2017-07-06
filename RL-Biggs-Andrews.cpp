#include "linearDecon.h"


#include <vector>
#include <algorithm>  // max(), min()
#include <limits>  // epsilon()

//#include <helper_timer.h>

bool notGoodDimension(unsigned num)
/*! Good dimension is defined as one that can be fatorized into 2s, 3s, 5s, and 7s
  According to CUFFT manual, such dimension would warranty fast FFT
*/
{
  if (num==2 || num==3 || num==5 || num==7)
    return false;
  else if (num % 2 == 0) return notGoodDimension(num / 2);
  else if (num % 3 == 0) return notGoodDimension(num / 3);
  else if (num % 5 == 0) return notGoodDimension(num / 5);
  else if (num % 7 == 0) return notGoodDimension(num / 7);
  else
    return true;
}

unsigned findOptimalDimension(unsigned inSize, int step)
/*!
  "step" can be positive or negative.  By default this is -1 (as defined in function declaration : ...int step=-1);
*/
{
  unsigned outSize = inSize;
  while (notGoodDimension(outSize))
    outSize += step;

  return outSize;
}

// static globals for photobleach correction:
static double intensity_overall0 = 0.;
static bool bFirstTime = true;







//***************************************************************************************************************
//********************************************* RichardsonLucy_GPU  *********************************************
//***************************************************************************************************************


void RichardsonLucy_GPU(CImg<> & raw, float background, 
                        GPUBuffer & d_interpOTF, int nIter,
                        double deskewFactor, int deskewedNx, int extraShift,
                        int napodize, int nZblend,
                        CPUBuffer &rotationMatrix, cufftHandle rfftplanGPU, 
                        cufftHandle rfftplanInvGPU, CImg<> & raw_deskewed,
						cudaDeviceProp *devProp, int myGPUdevice, bool bFlatStartGuess, float my_median,
						bool No_Bleach_correction, bool UseOnlyHostMem)
{
	size_t free; //for GPU memory profiling
	size_t total;//for GPU memory profiling

  // "raw" contains the raw image, also used as the initial guess X_0
  unsigned int nx = raw.width();
  unsigned int ny = raw.height();
  unsigned int nz = raw.depth();

  unsigned int nxy = nx * ny;
  unsigned int nxy2 = (nx/2 + 1)*ny; // x=N3, y=N2, z=N1 see: http://docs.nvidia.com/cuda/cufft/#multi-dimensional

#ifndef NDEBUG
#ifdef _WIN32
  StopWatchWin stopwatch;
#else
  StopWatchLinux stopwatch;
#endif
  stopwatch.start();
#endif

   // allocate buffers in GPU device 0
  GPUBuffer X_k(nz * nxy * sizeof(float), myGPUdevice, false); // Estimate after RL iteration
  std::cout << "X_k allocated.          " ;
  cudaMemGetInfo(&free, &total);
  std::cout << std::setw(8) << X_k.getSize() / (1024 * 1024) << "MB" << std::setw(8) << free / (1024 * 1024) << "MB free " ;


  std::cout << "Pinning Host RAM.  ";
  cutilSafeCall(cudaHostRegister(raw.data(), nz*nxy*sizeof(float), cudaHostRegisterPortable)); //pin the host RAM
  // transfer host data to GPU
  std::cout << "Copy raw.data to X_k HostToDevice.  ";
  cutilSafeCall(cudaMemcpy(X_k.getPtr(), raw.data(), nz*nxy*sizeof(float),
	  cudaMemcpyDefault));
  std::cout << "Done.  " << std::endl;

#ifndef NDEBUG
  printf("%f msecs\n", stopwatch.getTime());
#endif

  if (nIter > 0 || raw_deskewed.size()>0 || rotationMatrix.getSize()) {
    apodize_GPU(&X_k, nx, ny, nz, napodize);

	//**************************** Background subtraction ***********************************
    // background subtraction (including thresholding by 0):
    // printf("background=%f\n", background);
    backgroundSubtraction_GPU(X_k, nx, ny, nz, background, devProp->maxGridSize[2]);


	
	//**************************** Bleach correction ***********************************

    // Calculate sum for bleach correction:
    double intensity_overall = meanAboveBackground_GPU(X_k, nx, ny, nz, devProp->maxGridSize[2], myGPUdevice);
    
	if (bFirstTime || No_Bleach_correction) {
      intensity_overall0 = intensity_overall;
      bFirstTime = false;
    }
    else
      rescale_GPU(X_k, nx, ny, nz, intensity_overall0/intensity_overall, devProp->maxGridSize[2]);
#ifndef NDEBUG
    printf("intensity_overall=%lf\n", intensity_overall);
#endif
  

	//**************************** Deskew ***********************************
    if (fabs(deskewFactor) > 0.0) { //then deskew raw data along x-axis first:

		GPUBuffer deskewedRaw(nz * ny * deskewedNx * sizeof(float), myGPUdevice, UseOnlyHostMem);
		std::cout << "deskewedRaw allocated.  ";
		cudaMemGetInfo(&free, &total);
		std::cout << std::setw(8) << X_k.getSize() / (1024 * 1024) << "MB" << std::setw(8) << free / (1024 * 1024) << "MB free" ;
		
		std::cout << " Deskewing. ";
		deskew_GPU(X_k, nx, ny, nz, deskewFactor, deskewedRaw, deskewedNx, extraShift);

      // update raw (i.e., X_k) and its dimension variables.
		std::cout << "Copy deskewedRaw back to X_k. ";
		X_k = deskewedRaw;
		
	  
		nx = deskewedNx;
		nxy = nx * ny;
		nxy2 = (nx / 2 + 1)*ny; // x=N3, y=N2, z=N1 see: http://docs.nvidia.com/cuda/cufft/#multi-dimensional

		cutilSafeCall(cudaHostUnregister(raw.data()));
		raw.clear();
		raw.assign(nx, ny, nz, 1);

		if (nIter > 0)
	        cutilSafeCall(cudaHostRegister(raw.data(), nz*nxy*sizeof(float), cudaHostRegisterPortable));

		if (raw_deskewed.size()>0) {
			// save deskewed raw data into "raw_deskewed"; if no decon iteration is requested, then return immediately.
			std::cout << "Copy X_k into raw_deskewed. " ;
			cutilSafeCall(cudaMemcpy(raw_deskewed.data(), X_k.getPtr(),
				nz*nxy*sizeof(float), cudaMemcpyDefault));
        if (nIter == 0)
          return;
      }
		std::cout << "Done." << std::endl;
    }


	//**************************** Z blend ***********************************

    if (nZblend > 0)
      zBlend_GPU(X_k, nx, ny, nz, nZblend);
  }


  GPUBuffer rawGPUbuf(X_k, myGPUdevice, UseOnlyHostMem);  // make a copy of raw image
  
  std::cout << "rawGPUbuf allocated.    " ;
  cudaMemGetInfo(&free, &total);
  std::cout << std::setw(8) << rawGPUbuf.getSize() / (1024*1024) << "MB" << std::setw(8) << free / (1024 * 1024) << "MB free" << std::endl;
  

  GPUBuffer X_kminus1(nz * nxy * sizeof(float), myGPUdevice, UseOnlyHostMem); // guess at the end of previous RL iteration
  std::cout << "X_kminus1 allocated.    " ;
  cudaMemGetInfo(&free, &total);
  std::cout << std::setw(8) << X_kminus1.getSize() / (1024 * 1024) << "MB" << std::setw(8) << free / (1024 * 1024) << "MB free" << std::endl;


  GPUBuffer Y_k(nz * nxy * sizeof(float), myGPUdevice, false); // guess at beginning of RL iteration
  std::cout << "Y_k allocated.          " ;
  cudaMemGetInfo(&free, &total);
  std::cout << std::setw(8) << Y_k.getSize() / (1024 * 1024) << "MB" << std::setw(8) << free / (1024 * 1024) << "MB free" << std::endl;

  GPUBuffer CC(nz * nxy * sizeof(float), myGPUdevice, UseOnlyHostMem); // RL factor to apply to Y_k to get X_k
  std::cout << "CC allocated.           ";
  cudaMemGetInfo(&free, &total);
  std::cout << std::setw(8) << CC.getSize() / (1024 * 1024) << "MB" << std::setw(8) << free / (1024 * 1024) << "MB free" << std::endl;

  GPUBuffer G_kminus1(nz * nxy * sizeof(float), myGPUdevice, UseOnlyHostMem); // X_k - Y_k (RL change)
  std::cout << "G_kminus1 allocated.    ";
  cudaMemGetInfo(&free, &total);
  std::cout << std::setw(8) << G_kminus1.getSize() / (1024 * 1024) << "MB" << std::setw(8) << free / (1024 * 1024) << "MB free" << std::endl;

  GPUBuffer G_kminus2(nz * nxy * sizeof(float), myGPUdevice, UseOnlyHostMem); // previous X_k - Y_k (change between prediction and acceleration)
  std::cout << "G_kminus2 allocated.    ";
  cudaMemGetInfo(&free, &total);
  std::cout << std::setw(8) << G_kminus2.getSize() / (1024 * 1024) << "MB" << std::setw(8) << free / (1024 * 1024) << "MB free" << std::endl;


/*  testing using 2D texture for OTF interpolation
  CImg<> realpart(otf.width()/2, otf.height()), imagpart(realpart);
#pragma omp parallel for  
  cimg_forXY(realpart, x, y) {
    realpart(x, y) = otf(2*x  , y);
    imagpart(x, y) = otf(2*x+1, y);
  }

  prepareOTFtexture(realpart.data(), imagpart.data(), realpart.width(), realpart.height());
  
  CPUBuffer interpOTF(d_interpOTF); //.getSize());
  // d_interpOTF.set(&interpOTF, 0, interpOTF.getSize(), 0);

  CImg<float> otfarr((float *) interpOTF.getPtr(), nz*2, nx/2+1);
  otfarr.save("interpOTF.tif");

  return;
  debugging code ends
*/
  // Allocate GPU buffer for temp FFT result
  GPUBuffer fftGPUbuf(nz * nxy2 * sizeof(cuFloatComplex), myGPUdevice, UseOnlyHostMem); // This is the complex FFT output.
  std::cout << "fftGPUbuf allocated.    ";
  cudaMemGetInfo(&free, &total);
  std::cout << std::setw(8) << fftGPUbuf.getSize() / (1024 * 1024) << "MB" << std::setw(8) << free / (1024 * 1024) << "MB free" << std::endl;


  double lambda=0; // acceleration factor
  float eps = std::numeric_limits<float>::epsilon();
  
  
  
  //****************************************************************************
  //****************************RL Iterations ***********************************
  //****************************************************************************
  
  // R-L iteration
  for (int k = 0; k < nIter; k++) {
    std::cout << "Iteration " << k << ". ";
    // a. Make an image predictions for the next iteration    
	if (k > 1) {
		lambda = calcAccelFactor(G_kminus1, G_kminus2, nx, ny, nz, eps, myGPUdevice); // (G_km1 dot G_km2) / (G_km2 dot G_km2)
		lambda = std::max(std::min(lambda, 1.), 0.); // stability enforcement

		printf("Lambda = %.2f. ", lambda);

		updatePrediction(Y_k, X_k, X_kminus1, lambda, nx, ny, nz, devProp->maxGridSize[2]); // Y_k = X_k + lambda * (X_k - X_kminus1)
	}
	
	else if (bFlatStartGuess && k == 0)
	{
		std::cout << "Median. ";
		CImg<float> FlatStartGuess(raw, "xyzc", my_median); //create a buffer and fill with median value of image.
		std::cout << "Copy Median to Y_k. ";
		cutilSafeCall(cudaHostRegister(FlatStartGuess.data(), nz*nxy*sizeof(float), cudaHostRegisterPortable)); //pin the host RAM
		// transfer host data to GPU
		cutilSafeCall(cudaMemcpy(Y_k.getPtr(), FlatStartGuess.data(), nz*nxy*sizeof(float),
			cudaMemcpyDefault));
		cutilSafeCall(cudaHostUnregister(FlatStartGuess.data()));
		~FlatStartGuess;
	}
	
	else
	{
		std::cout << "Copy X_k to Y_k. ";
		Y_k = X_k; // copy data (not pointer) from X_k to Y_k (= operator has been redefined)
	}


	std::cout << "Copy X_k to X_k-1. ";
    cutilSafeCall(cudaMemcpyAsync(X_kminus1.getPtr(), X_k.getPtr(),				//copy previous guess to X_kminus1
		X_k.getSize(), cudaMemcpyDefault));
	
	if (k > 0){
		std::cout << "Copy G_k-1 to G_k-2. ";
		cutilSafeCall(cudaMemcpyAsync(G_kminus2.getPtr(), G_kminus1.getPtr(),
			G_kminus1.getSize(), cudaMemcpyDefault));
	}
    
	std::cout << "Filter1. ";
	// b.  Make core for the LR estimation ( raw/reblurred_current_estimation )
    CC = Y_k;
    filterGPU(CC, nx, ny, nz, rfftplanGPU, rfftplanInvGPU, fftGPUbuf, d_interpOTF,
              false, devProp->maxGridSize[2]);

	
    // if (k==0) {
    //   CPUBuffer CC_cpu(CC);
    //   CImg<> CCimg((float *) CC_cpu.getPtr(), nx, ny, nz, 1, true);
    //   CCimg.save_tiff("afterFilter0.tif");
    // }
	std::cout << "LRcore. ";

    calcLRcore(CC, rawGPUbuf, nx, ny, nz, devProp->maxGridSize[2]);

	
    // c. Determine next iteration image & apply positivity constraint
    // X_kminus1 = X_k;

	std::cout << "Filter2. ";
    filterGPU(CC, nx, ny, nz, rfftplanGPU, rfftplanInvGPU, fftGPUbuf, d_interpOTF,
              true, devProp->maxGridSize[2]);

    updateCurrEstimate(X_k, CC, Y_k, nx, ny, nz, devProp->maxGridSize[2]); //updated current estimate: Y_k * CC plus positivity constraint. "X_k" is updated upon return.

    // G_kminus2 = G_kminus1;
    calcCurrPrevDiff(X_k, Y_k, G_kminus1, nx, ny, nz, devProp->maxGridSize[2]); //G_kminus1 = X_k - Y_k change from RL
	std::cout << "Done. " << std::endl;
  }

  //************************************************************************************
  //****************************RL Iterations complete***********************************
  //************************************************************************************


  // Rotate decon result if requested:
  
  if (rotationMatrix.getSize()) {
	  std::cout << "Rotating...";
	  GPUBuffer d_rotatedResult(nz * nxy * sizeof(float), myGPUdevice, UseOnlyHostMem);

	  GPUBuffer d_rotMatrix(rotationMatrix, myGPUdevice, UseOnlyHostMem);

    rotate_GPU(X_k, nx, ny, nz, d_rotMatrix, d_rotatedResult);
    // Download from device memory back to "raw":
    cutilSafeCall(cudaMemcpy(raw.data(), d_rotatedResult.getPtr(), nz*nxy*sizeof(float),
		cudaMemcpyDefault));
	std::cout << "Done." << std::endl;
  }

  else {
    // Download from device memory back to "raw":
    cutilSafeCall(cudaMemcpy(raw.data(), X_k.getPtr(), nz*nxy*sizeof(float),
		cudaMemcpyDefault));
  }

  if (nIter > 0)
    cutilSafeCall(cudaHostUnregister(raw.data()));

#ifndef NDEBUG
  printf("%f msecs\n", stopwatch.getTime());
#endif

  // result is returned in "raw"
}


unsigned output_ny, output_nz, output_nx;
bool bCrop;
CImg<> complexOTF;
double deskewFactor;
unsigned deskewedXdim;
CPUBuffer rotMatrix;
cufftHandle rfftplanGPU, rfftplanInvGPU;
GPUBuffer d_interpOTF(0, false); // since this is a global for th RL_interface dll, just leave device empty.  It will probably default to GPU 0.

unsigned get_output_nx()
{
  return deskewedXdim; //output_nx;
}

unsigned get_output_ny()
{
  return output_ny;
}
unsigned get_output_nz()
{
  return output_nz;
}

int RL_interface_init(int nx, int ny, int nz, // raw image dimensions
                      float dr, float dz, // raw image pixel sizes
                      float dr_psf, float dz_psf, // PSF image pixel sizes
                      float deskewAngle, // deskew
                      float rotationAngle,
                      int outputWidth,
					  char * OTF_file_name, int myGPUdevice) // device might not work, since d_interpOTF is a global and device is set at compile time.
{
	cudaSetDevice(myGPUdevice);
  // Find the optimal dimensions nearest to the originals to meet CUFFT demands
  bCrop = false;
  output_ny = findOptimalDimension(ny);
  if (output_ny != ny) {
    printf("output ny=%d\n", output_ny);
    bCrop = true;
  }

  output_nz = findOptimalDimension(nz);
  if (output_nz != nz) {
    printf("output nz=%d\n", output_nz);
    bCrop = true;
  }

  // only if no deskewing is happening do we want to change image width here
  output_nx = nx;
  if ( ! (fabs(deskewAngle) > 0.0) ){
    output_nx = findOptimalDimension(nx);
    if (output_nx != nx) {
      printf("new nx=%d\n", output_nx);
      bCrop = true;
    }
  }

  // Load OTF and obtain OTF dimensions and pixel sizes, etc:
  try {
    complexOTF.assign(OTF_file_name);
  }
  catch (CImgIOException &e) {
    std::cerr << e.what() << std::endl; //OTF_file_name << " cannot be opened\n";
    return 0;
  }
  unsigned nr_otf = complexOTF.height();
  unsigned nz_otf = complexOTF.width() / 2;
  float dkr_otf = 1/((nr_otf-1)*2 * dr_psf);
  float dkz_otf = 1/(nz_otf * dz_psf);

  // Obtain deskew factor and new x dimension if deskew is run:
  deskewFactor = 0.;
  deskewedXdim = output_nx;
  if (fabs(deskewAngle) > 0.0) {
    if (deskewAngle <0) deskewAngle += 180.;
    deskewFactor = cos(deskewAngle * M_PI/180.) * dz / dr;
    if (outputWidth ==0)
      deskewedXdim += floor(output_nz * dz * 
                            fabs(cos(deskewAngle * M_PI/180.)) / dr)/4.; // TODO /4.
    else
      deskewedXdim = outputWidth; // use user-provided output width if available

    deskewedXdim = findOptimalDimension(deskewedXdim);
    // update z step size: (this is fine even though dz is a function parameter)
    dz *= sin(deskewAngle * M_PI/180.);
  }

  // Construct rotation matrix:
  if (fabs(rotationAngle) > 0.0) {
    rotMatrix.resize(4*sizeof(float));
    rotationAngle *= M_PI/180;
    float stretch = dr / dz;
    float *p = (float *)rotMatrix.getPtr();
    p[0] = cos(rotationAngle) * stretch;
    p[1] = sin(rotationAngle) * stretch;
    p[2] = -sin(rotationAngle);
    p[3] = cos(rotationAngle);
  }

  // Create reusable cuFFT plans
  cufftResult cuFFTErr = cufftPlan3d(&rfftplanGPU, output_nz, output_ny, deskewedXdim, CUFFT_R2C);
  if (cuFFTErr != CUFFT_SUCCESS) {
    std::cerr << "cufftPlan3d() c2r failed\n";
    return 0;
  }
  cuFFTErr = cufftPlan3d(&rfftplanInvGPU, output_nz, output_ny, deskewedXdim, CUFFT_C2R);
  if (cuFFTErr != CUFFT_SUCCESS) {
    std::cerr << "cufftPlan3d() c2r failed\n";
    return 0;
  }

  // Pass some constants to CUDA device:
  float dkx = 1.0/(dr * deskewedXdim);
  float dky = 1.0/(dr * output_ny);
  float dkz = 1.0/(dz * output_nz);
  float eps = std::numeric_limits<float>::epsilon();
  transferConstants(deskewedXdim, output_ny, output_nz, nr_otf, nz_otf,
                    dkx/dkr_otf, dky/dkr_otf, dkz/dkz_otf,
                    eps, complexOTF.data());

  // make a 3D interpolated OTF array:
  d_interpOTF.resize(output_nz * output_ny * (deskewedXdim+2) * sizeof(float));
  // catch exception here
  makeOTFarray(d_interpOTF, deskewedXdim, output_ny, output_nz);
  return 1;
}

int RL_interface(const unsigned short * const raw_data,
                 int nx, int ny, int nz,
                 float * const result,
                 float background,
                 int nIters,
                 int extraShift,
				 int myGPUdevice
                 )
{
	cudaSetDevice(myGPUdevice);
  CImg<> raw_image(raw_data, nx, ny, nz);

  if (bCrop)
    raw_image.crop(0, 0, 0, 0, output_nx-1, output_ny-1, output_nz-1, 0);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, myGPUdevice); 

  // Finally do calculation including deskewing, decon, rotation:
  CImg<> raw_deskewed;
  RichardsonLucy_GPU(raw_image, background, d_interpOTF, nIters,
                     deskewFactor, deskewedXdim, extraShift, 15, 10, rotMatrix,
					 rfftplanGPU, rfftplanInvGPU, raw_deskewed, &deviceProp, myGPUdevice, false, 1, true, false);

  // Copy deconvolved data, stored in raw_image, to "result" for return:
  memcpy(result, raw_image.data(), raw_image.size() * sizeof(float));

  return 1;
}

void RL_cleanup()
{
  d_interpOTF.resize(0);
  cufftDestroy(rfftplanGPU);
  cufftDestroy(rfftplanInvGPU);
}
