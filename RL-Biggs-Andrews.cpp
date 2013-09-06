#include "linearDecon.h"

#include <vector>
#include <algorithm>  // max(), min()
#include <limits>  // epsilon()

#include <helper_timer.h>


void filter(CImg<> &img, float dr, float dz,
            CImg<> &otf, float dkr_otf, float dkz_otf,
            fftwf_plan rfftplan, fftwf_plan rfftplan_inv,
            CImg<> &fft, bool bConj)
{
  int nx = img.width();
  int ny = img.height();
  int nz = img.depth();

  fftwf_execute_dft_r2c(rfftplan, img.data(), (fftwf_complex *) fft.data());

  float dkx = 1./(nx * dr);
  float dky = 1./(ny * dr);
  float dkz = 1./(nz * dz);

  float kxscale = dkx/dkr_otf;
  float kyscale = dky/dkr_otf;
  float kzscale = dkz/dkz_otf;

#pragma omp parallel for
  for (int k=0; k<nz; k++) {
    int kz = ( k>nz/2 ? k-nz : k );
    for (int i=0; i<ny; i++) {
      int ky = ( i > ny/2 ? i-ny : i );
      for (int j=0; j<nx/2+1; j++) {
        int kx = j;
        std::complex<float> otf_val =
          otfinterpolate((std::complex<float>*) otf.data(),
                         kx*kxscale, ky*kyscale,
                         kz*kzscale, otf.width()/2, otf.height());

        std::complex<float> result;
        if (bConj) // doing correlation instead of convolution
          result = std::conj(otf_val) * std::complex<float>(fft(2*j, i, k), fft(2*j+1, i, k));
        else
          result = otf_val * std::complex<float>(fft(2*j, i, k), fft(2*j+1, i, k));

        fft( 2*j,  i, k) = result.real();
        fft(2*j+1, i, k) = result.imag();
      }
    }
  }

  // fft.display();
  fftwf_execute_dft_c2r(rfftplan_inv, (fftwf_complex *) fft.data(), img.data());
  img /= img.size();
}

void RichardsonLucy(CImg<> & raw, float dr, float dz, 
                    CImg<> & otf, float dkr_otf, float dkz_otf, 
                    float rcutoff, int nIter,
                    fftwf_plan rfftplan, fftwf_plan rfftplan_inv, CImg<> &fft)
{
  // "raw" contains the raw image, also used as the initial guess X_0
  CImg<> G_kminus1, G_kminus2;

  float lambda=0;

  CImg<> X_k(raw);
  CImg<> X_kminus1; //(X_k, "xyz");
  CImg<> Y_k; //(X_k, "xyz");
  CImg<> CC;

  float eps = std::numeric_limits<float>::epsilon();

  for (int k = 0; k < nIter; k++) {
    std::cout << "Iteration " << k << std::endl;
    // a. Make an image predictions for the next iteration    
    if (k > 1) {
      lambda = G_kminus1.dot(G_kminus2) / (G_kminus2.dot(G_kminus2) + eps);
      lambda = std::max(std::min(lambda, 1.f), 0.f); // stability enforcement
      std::cout << lambda << std::endl;
      Y_k = X_k + lambda*(X_k - X_kminus1);
      Y_k.max(0.f); // plus positivity constraint
    }
    else 
      Y_k = X_k;
  
    // b.  Make core for the LR estimation ( raw/reblurred_current_estimation )

    CC = Y_k;
    filter(CC, dr, dz, otf, dkr_otf, dkz_otf, rfftplan, rfftplan_inv, fft, false);
    CC.max(eps);
#pragma omp parallel for
    cimg_forXYZ(CC, x, y, z) CC(x, y, z) = raw(x, y, z) / CC(x, y, z);

    // c. Determine next iteration image & apply positivity constraint
    X_kminus1 = X_k;
    filter(CC, dr, dz, otf, dkr_otf, dkz_otf, rfftplan, rfftplan_inv, fft, true);
#pragma omp parallel for
    cimg_forXYZ(CC, x, y, z) { X_k(x, y, z) = Y_k(x, y, z) * CC(x, y, z); 
      X_k(x, y, z) = X_k(x, y, z) > 0 ? X_k(x, y, z) : 0; } // plus positivity constraint

    G_kminus2 = G_kminus1;
    G_kminus1 = X_k - Y_k;
  }
  raw = X_k; // result is returned in "raw"
}


bool notGoodDimension(unsigned num)
// Good dimension is defined as one that can be fatorized into 2s, 3s, 5s, and 7s
// According to CUFFT manual, such dimension would warranty fast FFT
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

unsigned findOptimalDimension(unsigned inSize, int step=-1)
{
  unsigned outSize = inSize;
  while (notGoodDimension(outSize))
    outSize += step;

  return outSize;
}

void RichardsonLucy_GPU(CImg<> & raw, float background, float dr, float dz, 
                        CImg<> & otf, float dkr_otf, float dkz_otf, 
                        float rcutoff, int nIter,
                        CPUBuffer &deskewMatrix, int newXdim,
                        CPUBuffer &rotationMatrix)
{
  // "raw" contains the raw image, also used as the initial guess X_0

  unsigned int nx = raw.width();
  unsigned int ny = raw.height();
  unsigned int nz = raw.depth();
  std::cout << nz << ", " << ny << ", " << nx << std::endl;


  unsigned int nxy = nx * ny;
  unsigned int nxy2 = (nx+2)*ny;

  CPUBuffer raw1plane(nxy*sizeof(float));

#ifdef _WIN32
  StopWatchWin stopwatch;
#else
  StopWatchLinux stopwatch;
#endif

  stopwatch.start();

   // allocate buffers in GPU device 0
  GPUBuffer X_k(nz * nxy * sizeof(float), 0);
  // transfer host data to GPU
  cutilSafeCall(cudaMemcpy(X_k.getPtr(), raw.data(), nz*nxy*sizeof(float),
                           cudaMemcpyHostToDevice));

  // for (unsigned int z=0; z<nz; z++) {
  //   raw1plane.setFrom(raw.data(0, 0, z), 0, nxy*sizeof(float), 0);
  //   raw1plane.set(&X_k, 0, nxy*sizeof(float),
  //                 (z*nxy) * sizeof(float));
  // }

  printf("%f msecs\n", stopwatch.getTime());

  // resize the 3 dimensions to be cuFFT friendly (i.e., factoriz-able into 2^m*3^n*5^o*7^p)

  bool bCrop = false;
  unsigned new_ny = findOptimalDimension(ny);
  if (new_ny != ny) {
    printf("new ny=%d\n", new_ny);
    bCrop = true;
  }

  unsigned new_nz = findOptimalDimension(nz);
  if (new_nz != nz) {
    printf("new nz=%d\n", new_nz);
    bCrop = true;
  }

  // only if no deskewing is happening do we want to change image width here
  unsigned new_nx = nx;
  if (!deskewMatrix.getSize()) {
    new_nx = findOptimalDimension(nx);
    if (new_nx != nx) {
      printf("new nx=%d\n", new_nx);
      bCrop = true;
    }
  }

  if (bCrop) {
    GPUBuffer croppedImg(new_nx * new_ny * new_nz * sizeof(float), 0);
    cropGPU(X_k, nx, ny, nz, new_nx, new_ny, new_nz, croppedImg);

    X_k = croppedImg;

    nx = new_nx;
    ny = new_ny;
    nz = new_nz;
    nxy = nx*ny;
    nxy2 = (nx+2)*ny;
  }

  // background subtraction:
  backgroundSubtraction_GPU(X_k, nx, ny, nz, background);

  // if deskewMatrix's size > 0 then deskew raw data along x-axis first.
  GPUBuffer d_deskewMatrix;
  if (deskewMatrix.getSize()) {
    unsigned deskewedNx = findOptimalDimension(newXdim);
    printf("new nx=%d\n", deskewedNx);
    GPUBuffer deskewedRaw(nz * ny * deskewedNx * sizeof(float), 0);
  
    d_deskewMatrix = deskewMatrix; // should be done only for the first time point

    deskew_GPU(X_k, nx, ny, nz, d_deskewMatrix, deskewedRaw, deskewedNx);
    std::cout<< cudaGetErrorString(cudaGetLastError()) << std::endl;
    // update raw (i.e., X_k) and its dimension variables.
    X_k = deskewedRaw;

    nx = deskewedNx;
    nxy = nx*ny;
    nxy2 = (nx+2)*ny;
  }

  GPUBuffer rawGPUbuf(X_k);  // make a copy of raw image
  GPUBuffer X_kminus1(nz * nxy * sizeof(float), 0);
  GPUBuffer Y_k(nz * nxy * sizeof(float), 0);
  GPUBuffer G_kminus1(nz * nxy * sizeof(float), 0);
  GPUBuffer G_kminus2(nz * nxy * sizeof(float), 0);
  GPUBuffer CC(nz * nxy * sizeof(float), 0);

  // testing using 2D texture for OTF interpolation
//   CImg<> realpart(otf.width()/2, otf.height()), imagpart(realpart);
// #pragma omp parallel for  
//   cimg_forXY(realpart, x, y) {
//     realpart(x, y) = otf(2*x  , y);
//     imagpart(x, y) = otf(2*x+1, y);
//   }

//   prepareOTFtexture(realpart.data(), imagpart.data(), realpart.width(), realpart.height());
  
  // transfer a bunch of constants to device, including OTF array
  float eps = std::numeric_limits<float>::epsilon();
  transferConstants(nx, ny, nz, otf.height(), otf.width()/2,
                    1/(dr*nx)/dkr_otf, 1/(dr*ny)/dkr_otf, 1/(dz*nz)/dkz_otf,
                    eps, otf.data());

  // make a 3D interpolated OTF array:
  GPUBuffer d_interpOTF(nz * nxy2 * sizeof(float), 0);
  makeOTFarray(d_interpOTF, nx, ny, nz);

  // CPUBuffer interpOTF(d_interpOTF); //.getSize());
  // // d_interpOTF.set(&interpOTF, 0, interpOTF.getSize(), 0);

  // CImg<float> otfarr((float *) interpOTF.getPtr(), nz*2, nx/2+1);
  // otfarr.save("interpOTF.tif");

  // return;
  //debugging code ends

  // 

  // Create reusable cuFFT plans
  cufftHandle rfftplanGPU, rfftplanInvGPU;
  cufftResult cuFFTErr = cufftPlan3d(&rfftplanGPU, nz, ny, nx, CUFFT_R2C);
  if (cuFFTErr != CUFFT_SUCCESS) {
    std::cout << "Error code: " << cuFFTErr << std::endl;
    throw std::runtime_error("cufftPlan3d() r2c failed.");
  }
  cuFFTErr = cufftPlan3d(&rfftplanInvGPU, nz, ny, nx, CUFFT_C2R);
  if (cuFFTErr != CUFFT_SUCCESS) {
    std::cout << "Error code: " << cuFFTErr << std::endl;
    throw std::runtime_error("cufftPlan3d() c2r failed.");
  }

  // Allocate GPU buffer for temp FFT result
  GPUBuffer fftGPUbuf(nz * nxy2 * sizeof(float), 0);

  double lambda=0;
  // R-L iteration
  for (int k = 0; k < nIter; k++) {
    std::cout << "Iteration " << k << std::endl;
    // a. Make an image predictions for the next iteration    
    if (k > 1) {
      lambda = calcAccelFactor(G_kminus1, G_kminus2, nx, ny, nz, eps);
      lambda = std::max(std::min(lambda, 1.), 0.); // stability enforcement
      std::cout << lambda << std::endl;
      updatePrediction(Y_k, X_k, X_kminus1, lambda, nx, ny, nz);
    }
    else 
      Y_k = X_k;
  
    // b.  Make core for the LR estimation ( raw/reblurred_current_estimation )

    CC = Y_k;
    filterGPU(CC, nx, ny, nz, rfftplanGPU, rfftplanInvGPU, fftGPUbuf, d_interpOTF, false);
    calcLRcore(CC, rawGPUbuf, nx, ny, nz);

    // c. Determine next iteration image & apply positivity constraint
    X_kminus1 = X_k;
    filterGPU(CC, nx, ny, nz, rfftplanGPU, rfftplanInvGPU, fftGPUbuf, d_interpOTF, true);
    updateCurrEstimate(X_k, CC, Y_k, nx, ny, nz);

    G_kminus2 = G_kminus1;
    calcCurrPrevDiff(X_k, Y_k, G_kminus1, nx, ny, nz);
  }

  raw1plane.resize(nxy * sizeof(float));
  raw.clear();
  raw.assign(nx, ny, nz, 1);

  // Rotate decon result if requested:
  
  if (rotationMatrix.getSize()) {
    GPUBuffer d_rotatedResult(nz * nxy * sizeof(float), 0);

    GPUBuffer d_rotMatrix(rotationMatrix, 0);

    rotate_GPU(X_k, nx, ny, nz, d_rotMatrix, d_rotatedResult);

    for (unsigned int z=0; z<nz; z++) {
      // transfer from device to host one plane at a time
      d_rotatedResult.set(&raw1plane, z*nxy*sizeof(float), (z+1)*nxy*sizeof(float), 0);
      // transfer to "raw"
      raw1plane.setPlainArray(raw.data(0, 0, z), 0, nxy*sizeof(float), 0);
    }
  }

  else {
    // Download from device memory back to "raw":
    for (unsigned int z=0; z<nz; z++) {
      // transfer from device to host one plane at a time
      X_k.set(&raw1plane, z*nxy*sizeof(float), (z+1)*nxy*sizeof(float), 0);
      // transfer to "raw"
      raw1plane.setPlainArray(raw.data(0, 0, z), 0, nxy*sizeof(float), 0);
    }
  }
  printf("%f msecs\n", stopwatch.getTime());
  // result is returned in "raw"
}

