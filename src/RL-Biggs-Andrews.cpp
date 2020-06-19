#include "linearDecon.h"


#include <vector>
#include <algorithm>  // max(), min()
#include <limits>  // epsilon()
#include <stdio.h> // print to GPUmessage





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


void  determine_OTF_dimensions(CImg<> &complexOTF, float dr_psf, float dz_psf,
                              unsigned &nx_otf, unsigned &ny_otf, unsigned &nz_otf,
                              float &dkx_otf, float &dky_otf, float &dkz_otf)
{
  if (complexOTF.depth() > 1) {  // indicating non-RA OTF
    nx_otf = complexOTF.width() / 2;
    ny_otf = complexOTF.height();
    nz_otf = complexOTF.depth();
  }
  else {
    nx_otf = complexOTF.height();
    ny_otf = 1;   // indicator for a rotationally averaged OTF?
    nz_otf = complexOTF.width() / 2;
  }

  dkx_otf = 1/((nx_otf-1)*2 * dr_psf);

  if (ny_otf > 1)
    dky_otf = 1/(ny_otf * dr_psf);
  else
    dky_otf = dkx_otf;

  dkz_otf = 1/(nz_otf * dz_psf);
}


//***************************************************************************************************************
//********************************************* RichardsonLucy_GPU  *********************************************
//***************************************************************************************************************


void RichardsonLucy_GPU(CImg<> & raw, float background,
                        GPUBuffer & d_interpOTF, int nIter,
                        double deskewFactor, int deskewedNx, int extraShift,
                        int napodize, int nZblend,
                        CPUBuffer &rotationMatrix, cufftHandle rfftplanGPU,
                        cufftHandle rfftplanInvGPU, CImg<> & raw_deskewed,
                        cudaDeviceProp *devProp,
                        bool bFlatStartGuess, float my_median,
                        bool bDoRescale,
                        bool bSkewedDecon,
                        float padVal,
                        bool bDupRevStack,
                        bool UseOnlyHostMem,
                        int myGPUdevice)
{
  size_t free; //for GPU memory profiling
  size_t total;//for GPU memory profiling

  // "raw" contains the raw image, also used as the initial guess X_0
  size_t nx = raw.width();
  size_t ny = raw.height();
  size_t nz = raw.depth();

  size_t nxy = nx * ny;
  size_t nxy2 = (nx/2 + 1)*ny; // x=N3, y=N2, z=N1 see: http://docs.nvidia.com/cuda/cufft/#multi-dimensional

#ifndef NDEBUG
#ifdef _WIN32
  StopWatchWin stopwatch;
#else
  StopWatchLinux stopwatch;
#endif
  stopwatch.start();
#endif

  PUSH_RANGE("Alloc some buffers", 1);
  // allocate buffers in GPU device 0
  GPUBuffer X_k(nz * nxy * sizeof(float), myGPUdevice, false); // Estimate after RL iteration
  std::cout << "X_k allocated.          " ;
  cudaMemGetInfo(&free, &total);
  std::cout << std::setw(8) << (X_k.getSize() >> 20) << "MB" << std::setw(8)
            << (free >> 20 ) << "MB free " ;

  std::cout << "Pinning raw.data's Host RAM.  ";
  cutilSafeCall(cudaHostRegister(raw.data(), nz*nxy*sizeof(float),
                                 cudaHostRegisterPortable)); //pin the host RAM
  // transfer host data to GPU
  std::cout << "Copy raw.data to X_k HostToDevice.  ";
  cutilSafeCall(cudaMemcpy(X_k.getPtr(), raw.data(), nz*nxy*sizeof(float), cudaMemcpyDefault));
  std::cout << "Done.  " << std::endl;

#ifndef NDEBUG
  printf("%f msecs\n", stopwatch.getTime());
#endif
  POP_RANGE;

  if (nIter > 0 || raw_deskewed.size()>0 || rotationMatrix.getSize()) {
    apodize_GPU(X_k, nx, ny, nz, napodize);

    //**************************** Background subtraction ***********************************
    // background subtraction (including thresholding by 0):
    // printf("background=%f\n", background);
    backgroundSubtraction_GPU(X_k, nx, ny, nz, background, devProp->maxGridSize[0]);

    //**************************** Bleach correction ***********************************

    // Calculate sum for bleach correction:
    double intensity_overall = meanAboveBackground_GPU(X_k, nx, ny, nz,
                                                       devProp->maxGridSize[0],
                                                       myGPUdevice);

    if (bDoRescale) {
      if (bFirstTime) {
        intensity_overall0 = intensity_overall;
        bFirstTime = false;
      }
      else {
        rescale_GPU(X_k, nx, ny, nz, intensity_overall0/intensity_overall,
                    devProp->maxGridSize[0]);
      }
    }
#ifndef NDEBUG
    printf("intensity_overall=%lf\n", intensity_overall);
#endif

    //**************************** Deskew ***********************************
    if (( !bSkewedDecon || raw_deskewed.size()>0 && nIter == 0)
        && fabs(deskewFactor) > 0.0) { //then deskew raw data along x-axis first:

      GPUBuffer deskewedRaw(nz * ny * deskewedNx * sizeof(float), myGPUdevice, UseOnlyHostMem);
      std::cout << "deskewedRaw allocated.  ";
      cudaMemGetInfo(&free, &total);
      std::cout << std::setw(8) << (deskewedRaw.getSize() >> 20) << "MB"
                << std::setw(8) << (free >> 20) << "MB free" ;

      std::cout << " Deskewing... ";
      deskew_GPU(X_k, nx, ny, nz, deskewFactor, deskewedRaw, deskewedNx, extraShift, padVal);

      // update raw (i.e., X_k) and its dimension variables.
      std::cout << "Copy deskewedRaw back to X_k. ";
      X_k = deskewedRaw;

      nx = deskewedNx;
      nxy = nx * ny;
      nxy2 = (nx / 2 + 1)*ny;

      cutilSafeCall(cudaHostUnregister(raw.data()));
      raw.clear();
      raw.assign(nx, ny, nz, 1);

      if (nIter > 0)
        cutilSafeCall(cudaHostRegister(raw.data(), nz*nxy*sizeof(float),
                                       cudaHostRegisterPortable));

      if (raw_deskewed.size()>0) {
        // save deskewed raw data into "raw_deskewed";
        // if no decon iteration is requested, then return immediately.
        std::cout << "Copy X_k into raw_deskewed. " ;

        cudaError_t myCudaErr = cudaErrorHostMemoryAlreadyRegistered;
        myCudaErr = cudaHostRegister(raw_deskewed.data(), nz*nxy * sizeof(float),
                                     cudaHostRegisterPortable); //pin the destination CImg host RAM
        if (myCudaErr != cudaErrorHostMemoryAlreadyRegistered)
          cutilSafeCall(myCudaErr); // ignore error if this memory has already been registered.

        cutilSafeCall(cudaMemcpy(raw_deskewed.data(), X_k.getPtr(),
                                 nz*nxy*sizeof(float), cudaMemcpyDefault));
        if (nIter == 0)
          return;
      }
      std::cout << "Done." << std::endl;
    } // deskewedRaw's device memory is freed.


      //**************************** Z blend ***********************************

    if (nZblend > 0)
      zBlend_GPU(X_k, nx, ny, nz, nZblend);

    /***** Duplicate reversed stack to minimize ringing in Z ******/
    if (bDupRevStack) {
      GPUBuffer X_k2(nz*2 * nxy * sizeof(float), myGPUdevice, false);
#ifndef NDEBUG
      std::cout << "Copy X_k into X_k2. " ;
#endif
      cutilSafeCall(cudaMemcpy(X_k2.getPtr(), X_k.getPtr(),
                               nz*nxy*sizeof(float),
                               cudaMemcpyDefault));
      std::cout << "Done\n" ;
      duplicateReversedStack_GPU(X_k2, nx, ny, nz);
      nz *= 2;  // double nz till it's time to discard the duplicate
      X_k = X_k2;
    } // if (bDupRevStack)

  } //  if (nIter > 0 || raw_deskewed.size()>0 || rotationMatrix.getSize())



  { // these guys are needed only during RL iterations, we can deallocate them
    // once we are done with iterations. output we need in later stage is only X_k.

    GPUBuffer CC(nz * nxy * sizeof(float), myGPUdevice, UseOnlyHostMem); // RL factor to apply to Y_k to get X_k
    std::cout << "CC allocated.           ";
    cudaMemGetInfo(&free, &total);
    std::cout << std::setw(8) << (CC.getSize() >> 20) << "MB"
      << std::setw(8) << (free >> 20) << "MB free" << std::endl;

    GPUBuffer rawGPUbuf(X_k, myGPUdevice, UseOnlyHostMem);  // make a copy of raw image
    std::cout << "rawGPUbuf allocated.    ";
    cudaMemGetInfo(&free, &total);
    std::cout << std::setw(8) << (rawGPUbuf.getSize() >> 20) << "MB"
      << std::setw(8) << (free >> 20) << "MB free" << std::endl;

    GPUBuffer X_kminus1(nz * nxy * sizeof(float), myGPUdevice, UseOnlyHostMem); // guess at the end of previous RL iteration
    std::cout << "X_kminus1 allocated.    ";
    cudaMemGetInfo(&free, &total);
    std::cout << std::setw(8) << (X_kminus1.getSize() >> 20) << "MB"
      << std::setw(8) << (free >> 20) << "MB free" << std::endl;

    GPUBuffer Y_k(nz * nxy * sizeof(float), myGPUdevice, false); // guess at beginning of RL iteration
    std::cout << "Y_k allocated.          ";
    cudaMemGetInfo(&free, &total);
    std::cout << std::setw(8) << (Y_k.getSize() >> 20) << "MB"
      << std::setw(8) << (free >> 20) << "MB free" << std::endl;

    GPUBuffer G_kminus1(nz * nxy * sizeof(float), myGPUdevice, UseOnlyHostMem); // X_k - Y_k (RL change)
    std::cout << "G_kminus1 allocated.    ";
    cudaMemGetInfo(&free, &total);
    std::cout << std::setw(8) << (G_kminus1.getSize() >> 20) << "MB"
      << std::setw(8) << (free >> 20) << "MB free" << std::endl;

    GPUBuffer G_kminus2(nz * nxy * sizeof(float), myGPUdevice, UseOnlyHostMem); // previous X_k - Y_k (change between prediction and acceleration)
    std::cout << "G_kminus2 allocated.    ";
    cudaMemGetInfo(&free, &total);
    std::cout << std::setw(8) << (G_kminus2.getSize() >> 20) << "MB"
      << std::setw(8) << (free >> 20) << "MB free" << std::endl;

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
    std::cout << std::setw(8) << (fftGPUbuf.getSize() >> 20) << "MB"
      << std::setw(8) << (free >> 20) << "MB free" << std::endl;

#ifndef NDEBUG
    std::cout << "After all buffer alloc: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
#endif
    double lambda = 0; // acceleration factor
    float eps = std::numeric_limits<float>::epsilon(); // a value used inside RL iterations


    //****************************************************************************
    //****************************RL Iterations ***********************************
    //****************************************************************************
    // Before RL starts, make sure if we are doing 2-step 3D FFT, as indicated by a
    // NULL-valued handle "rfftplanInvGPU"
    cufftHandle rfftplan2D = NULL;
    if (rfftplanInvGPU == NULL) {
      // Allocate a 2D cuFFT forward and inverse plans
      cufftResult err = cufftPlan2d(&rfftplan2D, ny, nx, CUFFT_R2C);
      assert(err == CUFFT_SUCCESS);
      err = cufftPlan2d(&rfftplanInvGPU, ny, nx, CUFFT_C2R);
      assert(err == CUFFT_SUCCESS);
    }
    // R-L iteration
    for (int k = 0; k < nIter; k++) {

      std::cout << "Iteration ";
      int OldstdoutWidth = std::cout.width(2);
      std::cout << k;
      std::cout.width(OldstdoutWidth);
      std::cout << ". ";

      char GPUmessage[50];
      sprintf(GPUmessage, "Iter %d", k);
      PUSH_RANGE(GPUmessage, k);

      // a. Make an image predictions for the next iteration
      if (k > 1) {
        lambda = calcAccelFactor(G_kminus1, G_kminus2, nx, ny, nz, eps, myGPUdevice); // (G_km1 dot G_km2) / (G_km2 dot G_km2)
        lambda = std::max(std::min(lambda, 1.), 0.); // stability enforcement

        printf("Lambda = %.2f. ", lambda);

        updatePrediction(Y_k, X_k, X_kminus1, lambda, nx, ny, nz, devProp->maxGridSize[0]); // Y_k = X_k + lambda * (X_k - X_kminus1)
      }

      else if (bFlatStartGuess && k == 0) {
        std::cout << "Median. ";
        CImg<float> FlatStartGuess(raw, "xyzc", my_median); //create a buffer and fill with median value of image.
        std::cout << "Copy Median to Y_k. ";
        cutilSafeCall(cudaHostRegister(FlatStartGuess.data(), nz*nxy * sizeof(float), cudaHostRegisterPortable)); //pin the host RAM; do we really need this?? -lin
        // transfer host data to GPU
        cutilSafeCall(cudaMemcpy(Y_k.getPtr(), FlatStartGuess.data(), nz*nxy * sizeof(float),
                                 cudaMemcpyDefault));
        cutilSafeCall(cudaHostUnregister(FlatStartGuess.data()));
      }
      else {
        std::cout << "Cpy X_k to Y_k.";
        Y_k = X_k; // copy data (not pointer) from X_k to Y_k (= operator has been redefined)
#ifndef NDEBUG
        std::cout << "After Y_k=X_k: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
#endif
      }

      std::cout << "Copy X_k to X_k-1. ";  //copy previous guess to X_kminus1:
      cutilSafeCall(cudaMemcpyAsync(X_kminus1.getPtr(), X_k.getPtr(),
                                    X_k.getSize(), cudaMemcpyDefault));

      if (k > 0) {
        std::cout << "Copy G_k-1 to G_k-2. ";
        cutilSafeCall(cudaMemcpyAsync(G_kminus2.getPtr(), G_kminus1.getPtr(),
                                      G_kminus1.getSize(), cudaMemcpyDefault));
      }

      std::cout << "Filter1. ";
      // b.  Make core for the LR estimation ( raw/reblurred_current_estimation )
      CC = Y_k;
      filterGPU(CC, nx, ny, nz, rfftplanGPU, rfftplanInvGPU, rfftplan2D,
                fftGPUbuf, d_interpOTF, false, devProp->maxGridSize[0]);

      std::cout << "LRcore. ";

      calcLRcore(CC, rawGPUbuf, nx, ny, nz, devProp->maxGridSize[0]);

      // c. Determine next iteration image & apply positivity constraint
      // X_kminus1 = X_k;

      std::cout << "Filter2. ";
      filterGPU(CC, nx, ny, nz, rfftplanGPU, rfftplanInvGPU, rfftplan2D,
                fftGPUbuf, d_interpOTF, true, devProp->maxGridSize[0]);

      // updated current estimate: Y_k * CC plus positivity constraint;
      // "X_k" is updated upon return:
      updateCurrEstimate(X_k, CC, Y_k, nx, ny, nz, devProp->maxGridSize[0]);

      // G_kminus2 = G_kminus1;
      calcCurrPrevDiff(X_k, Y_k, G_kminus1, nx, ny, nz, devProp->maxGridSize[0]); //G_kminus1 = X_k - Y_k change from RL
      std::cout << "Done. " << std::endl;
      POP_RANGE;
    }
    if (rfftplan2D != NULL) { // clean up if 2D FFT plans were allocated
      cufftDestroy(rfftplan2D);
      cufftDestroy(rfftplanInvGPU);
    }
  } // iterations complete. Deallocate GPUbuffers that we don't need.  Just keep X_k
  //************************************************************************************
  //****************************RL Iterations complete**********************************
  //************************************************************************************

  if (bDupRevStack)
    // Change nz back to original and nothing else needs changed (hopefully)
    // even though X_k contains double-sized stack.
    nz /= 2;

  cudaDeviceSynchronize();
#ifndef NDEBUG
  std::cout << "After RL iterations: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
#endif
  
  if (bSkewedDecon && fabs(deskewFactor) > 0.0) { //deskew after decon
    cudaMemGetInfo(&free, &total);
    std::cout << std::setw(8) << free / (1<<20) << "MB free" << std::endl;
    GPUBuffer deskewedAfter(nz * ny * deskewedNx * sizeof(float), myGPUdevice, UseOnlyHostMem);
    std::cout << "deskewedAfter allocated.  ";
    cudaMemGetInfo(&free, &total);
    std::cout << std::setw(8) << deskewedAfter.getSize() / (1<<20) << "MB" << std::setw(8) << free / (1<<20) << "MB free" ;

    std::cout << " Deskewing after deconv... ";
    deskew_GPU(X_k, nx, ny, nz, deskewFactor, deskewedAfter, deskewedNx, extraShift);
    X_k = deskewedAfter;
    nx = deskewedNx;
    nxy = nx * ny;

    cutilSafeCall(cudaHostUnregister(raw.data()));
    raw.clear();
    raw.assign(nx, ny, nz, 1);
    cutilSafeCall(cudaHostRegister(raw.data(), nz*nxy*sizeof(float), cudaHostRegisterPortable));
  }

  // Rotate decon result if requested:

  if (rotationMatrix.getSize()) {
    std::cout << "Rotating...";

    float *p = (float *) rotationMatrix.getPtr();
    // Refer to rotMatrix definition in main():
    int nz_afterRot = nz * p[3] / p[0];
    int nx_afterRot = nx  * p[3] + nz * p[2] * p[2] / p[1];
    GPUBuffer d_rotatedResult(nz_afterRot * nx_afterRot * ny * sizeof(float), myGPUdevice, UseOnlyHostMem);
    GPUBuffer d_rotMatrix(rotationMatrix, myGPUdevice, UseOnlyHostMem);

    rotate_GPU(X_k, nx, ny, nz, d_rotMatrix, d_rotatedResult, nx_afterRot, nz_afterRot);
    if (nIter > 0)
      cutilSafeCall(cudaHostUnregister(raw.data()));
    raw.assign(nx_afterRot, ny, nz_afterRot);

    if (nIter > 0)
      cutilSafeCall(cudaHostRegister(raw.data(), nz_afterRot*nx_afterRot*ny*sizeof(float), cudaHostRegisterPortable));
    // Download from device memory back to "raw":
    cutilSafeCall(cudaMemcpy(raw.data(), d_rotatedResult.getPtr(),
                             nz_afterRot * nx_afterRot * ny * sizeof(float),
                             cudaMemcpyDefault));
    std::cout << "Done." << std::endl;
  }

  else {
    CPUBuffer temp(X_k);
    CImg<> temp1((float *) temp.getPtr(), nx, ny, nz, 1, true);
    raw = temp1;
    // Why the following throws "unspecified launch error" for nx less than certain limit?
    // Download from device memory back to "raw":
    //  cutilSafeCall(cudaMemcpy(raw.data(), X_k.getPtr(), nz*nxy*sizeof(float), cudaMemcpyDefault));
  }

  if (nIter > 0)
    cutilSafeCall(cudaHostUnregister(raw.data()));

  if (raw_deskewed.size())
    cudaHostUnregister(raw_deskewed.data()); // ignore error

#ifndef NDEBUG
  printf("%f msecs\n", stopwatch.getTime());
#endif

  // result is returned in "raw"
}


// ******************************************************
//                Shared library stuff
// ******************************************************



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

  if (rotMatrix.getSize() > 0) {
    float *p = (float *)rotMatrix.getPtr();
    int nx_afterRot = deskewedXdim  * p[3] + output_nz * p[2] * p[2] / p[1];
    return nx_afterRot;
  } else {
    return deskewedXdim;
  }

}

unsigned get_output_ny()
{
  return output_ny;
}
unsigned get_output_nz()
{

  if (rotMatrix.getSize() > 0) {
    float *p = (float *)rotMatrix.getPtr();
    int nz_afterRot = output_nz * p[3] / p[0];
    return nz_afterRot;
  } else {
    return output_nz;
  }


}

int RL_interface_init(int nx, int ny, int nz, // raw image dimensions
                      float dr, float dz, // raw image pixel sizes
                      float dr_psf, float dz_psf, // PSF image pixel sizes
                      float deskewAngle, // deskew
                      float rotationAngle,
                      int outputWidth,
                      bool bSkewedDecon,
                      bool bNoLimitRatio, // limit ratio to 10 in LRcore update?
                      char * OTF_file_name) // device might not work, since d_interpOTF is a global and device is set at compile time.
{

  //cudaSetDevice(myGPUdevice);
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
  // unsigned nr_otf = complexOTF.height();
  // unsigned nz_otf = complexOTF.width() / 2;
  // float dkr_otf = 1/((nr_otf-1)*2 * dr_psf);
  // float dkz_otf = 1/(nz_otf * dz_psf);
  unsigned nx_otf, ny_otf, nz_otf;
  float dkx_otf, dkz_otf, dky_otf;
  if (bSkewedDecon)
    dz_psf *= fabs(sin(deskewAngle * M_PI/180.));
  determine_OTF_dimensions(complexOTF, dr_psf, dz_psf, nx_otf, ny_otf, nz_otf,
                           dkx_otf, dky_otf, dkz_otf);

  GPUBuffer d_rawOTF(0, false);
  d_rawOTF.resize(nx_otf * ny_otf * nz_otf * sizeof(cuFloatComplex));
  cutilSafeCall(cudaMemcpy(d_rawOTF.getPtr(), complexOTF.data(),
                           d_rawOTF.getSize(), cudaMemcpyDefault));


  // Obtain deskew factor and new x dimension if deskew is run:
  deskewFactor = 0.;
  deskewedXdim = output_nx;
  if (fabs(deskewAngle) > 0.0) {
    if (deskewAngle <0) deskewAngle += 180.;
    deskewFactor = cos(deskewAngle * M_PI / 180.) * dz / dr;
    if (outputWidth == 0)
      deskewedXdim +=
          floor(output_nz * dz * fabs(cos(deskewAngle * M_PI / 180.)) / dr); // TODO /4.
    else
      deskewedXdim = outputWidth; // use user-provided output width if available

    deskewedXdim = findOptimalDimension(deskewedXdim);
    // update z step size: (this is fine even though dz is a function parameter)
    dz *= fabs(sin(deskewAngle * M_PI/180.));
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
  transferConstants(deskewedXdim, output_ny, output_nz, nx_otf, ny_otf, nz_otf,
                    dkx/dkx_otf, dky/dky_otf, dkz/dkz_otf, bNoLimitRatio, eps);

  // make a 3D interpolated OTF array:
  if (bSkewedDecon) {
    d_interpOTF.resize(output_nz * output_ny * (output_nx/2+1)* 2 * sizeof(float));
    makeOTFarray(d_rawOTF, d_interpOTF, deskewedXdim, output_ny, output_nz);
  }
  else {
    d_interpOTF.resize(output_nz * output_ny * (deskewedXdim+2) * sizeof(float));
    makeOTFarray(d_rawOTF, d_interpOTF, deskewedXdim, output_ny, output_nz);
  }
  return 1;
}

int RL_interface(const unsigned short * const raw_data,
                 int nx, int ny, int nz,
                 float * result,
                 float * raw_deskewed_result,
                 float background,
                 bool bDoRescale,
                 bool bSaveDeskewedRaw,
                 int nIters,
                 int extraShift,
                 int napodize, int nZblend,
                 float padVal,
                 bool bDupRevStack,
                 bool bSkewedDecon
                 )
{


  CImg<> raw_image(raw_data, nx, ny, nz);

  if (bCrop)
    raw_image.crop(0, 0, 0, 0, output_nx-1, output_ny-1, output_nz-1, 0);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  // Finally do calculation including deskewing, decon, rotation:
  CImg<> raw_deskewed;
  if (bSaveDeskewedRaw && (fabs(deskewFactor) > 0.0) ) {
    raw_deskewed.assign(deskewedXdim, output_ny, output_nz);
  }

  bool bFlatStartGuess = false;
  float my_median = 1;
  RichardsonLucy_GPU(raw_image, background, d_interpOTF, nIters,
                     deskewFactor, deskewedXdim, extraShift, napodize, nZblend, rotMatrix,
                     rfftplanGPU, rfftplanInvGPU, raw_deskewed, &deviceProp,
                     bFlatStartGuess, my_median, bDoRescale, padVal, bDupRevStack,
                     bSkewedDecon, false);

  // Copy deconvolved data, stored in raw_image, to "result" for return:
  memcpy(result, raw_image.data(), raw_image.size() * sizeof(float));

  // optionally grab deskewed data as well:
  if (bSaveDeskewedRaw) {
    memcpy(raw_deskewed_result, raw_deskewed.data(),
            deskewedXdim * output_ny * output_nz * sizeof(float));
  }

  return 1;
}

void RL_cleanup()
{
  intensity_overall0 = 0.;
  bFirstTime = true;
  d_interpOTF.resize(0);
  rotMatrix.resize(0);
  cufftDestroy(rfftplanGPU);
  cufftDestroy(rfftplanInvGPU);
}

void cuda_reset()
{
  cudaDeviceReset();
}


//For deskew only
int Deskew_interface(const float * const raw_data,
                     int nx, int ny, int nz,
                     float dz, float dr, float deskewAngle,
                     float * const result,
                     int outputWidth, int extraShift, float padVal){
  CImg<> raw_image(raw_data, nx, ny, nz);

  double deskewFactor;
  unsigned deskewedXdim;

  // Obtain deskew factor and new x dimension if deskew is run:
  deskewFactor = 0.;
  deskewedXdim = nx;
  if (fabs(deskewAngle) > 0.0) {
    if (deskewAngle <0) deskewAngle += 180.;
    deskewFactor = cos(deskewAngle * M_PI/180.) * dz / dr;
    if (outputWidth ==0)
      deskewedXdim += floor(output_nz * dz * fabs(cos(deskewAngle * M_PI/180.)) / dr)/4.;
    else
      deskewedXdim = outputWidth; // use user-provided output width if available
  }

//function signature
 //deskew_GPU(GPUBuffer &inBuf, int nx, int ny, int nz,
 //                         double deskewFactor, GPUBuffer &outBuf,
 //                         int newNx, int extraShift)



   // allocate buffers in GPU device 0
  GPUBuffer d_rawData(nz * nx * ny * sizeof(float), 0, false);
  GPUBuffer d_deskewedResult(nz * ny * deskewedXdim * sizeof(float), 0, false);
  //CImg<> deskewedResult_host(deskewedXdim, ny, nz);

  // pagelock data memory on host
  cudaHostRegister(raw_image.data(), nz * nx * ny * sizeof(float), cudaHostRegisterPortable);

  // transfer host data to GPU
  cudaMemcpy(d_rawData.getPtr(), raw_image.data(), nz * nx * ny *sizeof(float), cudaMemcpyHostToDevice);

  //perform deskew
  deskew_GPU(d_rawData, nx, ny, nz, deskewFactor, d_deskewedResult, deskewedXdim, extraShift, padVal);

  //transfer result back to host
  cudaMemcpy(result, d_deskewedResult.getPtr(), nz * deskewedXdim * ny*sizeof(float), cudaMemcpyDeviceToHost);

  cudaHostUnregister(raw_image.data());

  return 1;
}


int Affine_interface(const float * const raw_data,
                     int nx, int ny, int nz,
                     float * const result,
                     const float * affMat)
{
  CImg<> raw_image(raw_data, nx, ny, nz);

  GPUBuffer d_affMatrix(16 * sizeof(float), 0, false);
  cudaMemcpy(d_affMatrix.getPtr(), affMat, 16 * sizeof(float), cudaMemcpyHostToDevice);

  // Allocate CUDA array in device memory
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(
      32, 0, 0, 0, cudaChannelFormatKindFloat);

  cudaArray* cuArray;
  cudaExtent extent = make_cudaExtent(nx, ny, nz);
  cudaMalloc3DArray(
      &cuArray,
      &channelDesc,
      extent,
      cudaArrayDefault
  );

  // Copy to device memory some data located at address h_data
  // in host memory
  cudaMemcpy3DParms parms = {0};
  parms.srcPtr = make_cudaPitchedPtr(
      raw_image.data(),
      nx * sizeof(float), nx, ny
  );
  parms.dstArray = cuArray;
  parms.extent = extent;
  parms.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&parms);

  affine_GPU(cuArray, nx, ny, nz, result, d_affMatrix);

  return 1;
}

int Affine_interface_RA(const float * const raw_data,
                     int nx, int ny, int nz,
                     float dx, float dy, float dz,
                     float * const result,
                     const float * affMat)
{
  CImg<> raw_image(raw_data, nx, ny, nz);

  GPUBuffer d_affMatrix(16 * sizeof(float), 0, false);
  cudaMemcpy(d_affMatrix.getPtr(), affMat, 16 * sizeof(float), cudaMemcpyHostToDevice);

  // Allocate CUDA array in device memory
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(
      32, 0, 0, 0, cudaChannelFormatKindFloat);

  cudaArray* cuArray;
  cudaExtent extent = make_cudaExtent(nx, ny, nz);
  cudaMalloc3DArray(
      &cuArray,
      &channelDesc,
      extent,
      cudaArrayDefault
  );

  // Copy to device memory some data located at address h_data
  // in host memory
  cudaMemcpy3DParms parms = {0};
  parms.srcPtr = make_cudaPitchedPtr(
      raw_image.data(),
      nx * sizeof(float), nx, ny
  );
  parms.dstArray = cuArray;
  parms.extent = extent;
  parms.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&parms);

  affine_GPU_RA(cuArray, nx, ny, nz, dx, dy, dz, result, d_affMatrix);

  return 1;
}


// putting this here for now until I can clean up the camcor problem




int camcor_interface_init(int nx, int ny, int nz,
                     const float * const camparam)
{
  CImg<> h_camparam(camparam, nx, ny, 3);
  setupConst(nx, ny, nz);
  setupCamCor(nx, ny, h_camparam.data());
  return 1;
}


int camcor_interface(const unsigned short * const raw_data,
                     int nx, int ny, int nz,
                     unsigned short * const result)
{
  CImg<unsigned short> input(raw_data, nx, ny, nz);
  CImg<unsigned> raw_image(input);
  GPUBuffer d_correctedResult(nx * ny * nz * sizeof(unsigned short), 0, false);
  setupData(nx, ny, nz, raw_image.data());
  camcor_GPU(nx, ny, nz, d_correctedResult);
  //transfer result back to host
  cudaMemcpy(result, d_correctedResult.getPtr(), nx * ny * nz * sizeof(unsigned short), cudaMemcpyDeviceToHost);
  return 1;
}



