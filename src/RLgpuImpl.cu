#include "cutilSafeCall.h"

#include <CPUBuffer.h>
#include <GPUBuffer.h>
#include <cufft.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <math.h>

// Disable silly warnings on some Microsoft VC++ compilers.
#pragma warning(disable : 4244) // Disregard loss of data from float to int.
#pragma warning(                                                               \
    disable : 4267) // Disregard loss of data from size_t to unsigned int.
#pragma warning(disable : 4305) // Disregard loss of data from double to float.
#endif

__constant__ unsigned const_nx;
__constant__ unsigned const_ny;
__constant__ unsigned const_nz;
__constant__ unsigned const_nxyz;
__constant__ unsigned const_nxotf;
__constant__ unsigned const_nyotf;
__constant__ unsigned const_nzotf;

__constant__ float const_kxscale;
__constant__ float const_kyscale;
__constant__ float const_kzscale;
__constant__ float const_eps;
__constant__ int   const_bNoLimitRatio;

__global__ void filter_kernel(cuFloatComplex *devImg, cuFloatComplex *devOTF,
                              int size);
__global__ void filterConj_kernel(cuFloatComplex *devImg,
                                  cuFloatComplex *devOTF, int size);
__global__ void scale_kernel(float *img, double factor);
__global__ void LRcore_kernel(float *img1, float *img2);
__global__ void currEstimate_kernel(float *img1, float *img2, float *img3);
__global__ void currPrevDiff_kernel(float *img1, float *img2, float *img3);
__global__ void innerProduct_kernel(float *img1, float *img2,
                                    double *intRes1); //, double * intRes2);
__global__ void updatePrediction_kernel(float *Y_k, float *X_k, float *X_km1,
                                        float lambda);
__global__ void summation_kernel(float *img, double *intRes, int n);
__global__ void sumAboveThresh_kernel(float *img, double *intRes,
                                      unsigned *counter, float thresh, int n);
__global__ void apodize_x_kernel(int napodize, int nx, int ny, float *image);
__global__ void apodize_y_kernel(int napodize, int nx, int ny, float *image);
__global__ void zBlend_kernel(int nx, int ny, int nz, int nZblend,
                              float *image);

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
// (Copied from reduction_kernel.cu of CUDA samples)
template <class T> struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// texture<float, cudaTextureType2D, cudaReadModeElementType> texRef1, texRef2;
// cudaArray* d_realpart, *d_imagpart;  // used for OTF texture

__host__ void transferConstants(int nx, int ny, int nz, int nxotf, int nyotf, int nzotf,
                                float kxscale, float kyscale, float kzscale,
                                int bNoLimitRatio, float eps) {
  cutilSafeCall(cudaMemcpyToSymbol(const_nx, &nx, sizeof(int)));
  /* this could fail with "invalid device symbol" if the code is not compiled
  for this device compute capability.
  https://devtalk.nvidia.com/default/topic/474415/cuda-programming-and-performance/copy-to-constant-memory-fails/post/3376488/#3376488
  */
  cutilSafeCall(cudaMemcpyToSymbol(const_ny, &ny, sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(const_nz, &nz, sizeof(int)));
  unsigned int nxyz = nx * ny * nz;
  cutilSafeCall(cudaMemcpyToSymbol(const_nxyz, &nxyz, sizeof(unsigned int)));
  cutilSafeCall(cudaMemcpyToSymbol(const_nxotf, &nxotf, sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(const_nyotf, &nyotf, sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(const_nzotf, &nzotf, sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(const_kxscale, &kxscale, sizeof(float)));
  cutilSafeCall(cudaMemcpyToSymbol(const_kyscale, &kyscale, sizeof(float)));
  cutilSafeCall(cudaMemcpyToSymbol(const_kzscale, &kzscale, sizeof(float)));
  cutilSafeCall(cudaMemcpyToSymbol(const_bNoLimitRatio, &bNoLimitRatio, sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(const_eps, &eps, sizeof(float)));
}

// __host__ void prepareOTFtexture(float * realpart, float * imagpart, int nx,
// int ny)
// {
//   // Allocate CUDA array in device memory
//   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

//   cudaMallocArray(&d_realpart, &channelDesc, nx, ny);
//   cudaMallocArray(&d_imagpart, &channelDesc, nx, ny);

//   // Copy to device memory
//   cudaMemcpyToArray(d_realpart, 0, 0, realpart,
//                     nx * ny * sizeof(float),
//                     cudaMemcpyHostToDevice);
//   cudaMemcpyToArray(d_imagpart, 0, 0, imagpart,
//                     nx * ny * sizeof(float),
//                     cudaMemcpyHostToDevice);

//   // Set texture reference parameters
//   texRef1.addressMode[0] = cudaAddressModeClamp;
//   texRef1.addressMode[1] = cudaAddressModeClamp;
//   texRef1.filterMode = cudaFilterModeLinear;
//   texRef1.normalized = true;
//   texRef2.addressMode[0] = cudaAddressModeClamp;
//   texRef2.addressMode[1] = cudaAddressModeClamp;
//   texRef2.filterMode = cudaFilterModeLinear;
//   texRef2.normalized = true;
//   // Bind the arrays to the texture reference
//   cudaBindTextureToArray(texRef1, d_realpart, channelDesc);
//   cudaBindTextureToArray(texRef2, d_imagpart, channelDesc);
// }

__global__ void bgsubtr_kernel(float *img, size_t size, float background) {
  unsigned ind =
      (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

  if (ind < size) {
    img[ind] -= background;
    img[ind] = img[ind] > 0 ? img[ind] : 0;
  }
}

__host__ void backgroundSubtraction_GPU(GPUBuffer &img, int nx, int ny, int nz,
                                        float background,
                                        unsigned maxGridXdim) {
  unsigned nThreads = 1024;
  unsigned NXblock = ceil(nx * ny * nz / (float)nThreads);
  unsigned NYblock = 1;
  if (NXblock > maxGridXdim) {
    NYblock = NXblock / maxGridXdim;
    NXblock = maxGridXdim;
    // NYblock = NXblock = ceil(sqrt(NXblock));
  }
  dim3 grid(NXblock, NYblock);
  dim3 block(nThreads);

  bgsubtr_kernel<<<grid, block>>>((float *)img.getPtr(), nx * ny * nz,
                                  background);
#ifndef NDEBUG
  std::cout << "backgroundSubtraction_GPU(): "
            << cudaGetErrorString(cudaGetLastError()) << std::endl;
#endif
}

__host__ void filterGPU(GPUBuffer &img, int nx, int ny, int nz,
                        cufftHandle &rfftplan, cufftHandle &rfftplanInv,
                        cufftHandle &rfftplan2D,
                        GPUBuffer &fftBuf, GPUBuffer &otfArray, bool bConj,
                        unsigned maxGridXdim)
// "img" is of dimension (nx, ny, nz) and of float type
// "otf" is of dimension (const_nzotf, const_nrotf) and of complex type
{
  //
  // Rescale KERNEL
  //
  unsigned nThreads = 1024;
  unsigned NXblock = ceil(((float)(nx * ny * nz)) / nThreads);
  unsigned NYblock = 1;
  if (NXblock > maxGridXdim)
    NXblock = NYblock = ceil(sqrt(NXblock));

  dim3 gridDim(NXblock, NYblock);
  scale_kernel<<<gridDim, nThreads>>>((float *)img.getPtr(),
                                      1. / (nx * ny * nz));
#ifndef NDEBUG
  std::cout << "scale_kernel(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;
#endif
  cufftResult cuFFTErr;
  unsigned strideR = ny * nx;
  unsigned strideC = (nx/2 + 1) * ny;
  
  if (rfftplan2D == NULL)
    cuFFTErr = cufftExecR2C(rfftplan, (cufftReal *)img.getPtr(),
                            (cuFloatComplex *)fftBuf.getPtr());
  else {  // implying 2-step 3D FFT
    // First, 2D R2C FFT of all planes:
    for (auto z=0; z<nz; z++) {
      cuFFTErr = cufftExecR2C(rfftplan2D, ((cufftReal *)img.getPtr()) + z*strideR,
                              ((cuFloatComplex *)fftBuf.getPtr()) + z*strideC);
      if (cuFFTErr != CUFFT_SUCCESS) break;
    }
    // Second, a batch of in-place 1D C2C FFT along Z of all X-Y pixels:
    if (cuFFTErr == CUFFT_SUCCESS)
      cuFFTErr = cufftExecC2C(rfftplan, (cuFloatComplex *)fftBuf.getPtr(),
                              (cuFloatComplex *)fftBuf.getPtr(), CUFFT_FORWARD);
  }
  if (cuFFTErr != CUFFT_SUCCESS) {
    std::cerr << "Line:" << __LINE__ << " in function: " << __func__ << std::endl;
    throw std::runtime_error("cufft failed.");
  }
  //
  // KERNEL 1
  //

  unsigned arraySize = nz * ny * (nx / 2 + 1);
  NXblock = ceil(arraySize / (float)nThreads);
  NYblock = 1;
  if (NXblock > maxGridXdim)
    NXblock = NYblock = ceil(sqrt(NXblock));
  dim3 grid(NXblock, NYblock);
  dim3 block(nThreads);

  if (bConj)
    filterConj_kernel<<<grid, block>>>((cuFloatComplex *)fftBuf.getPtr(),
                                       (cuFloatComplex *)otfArray.getPtr(),
                                       arraySize);
  else
    filter_kernel<<<grid, block>>>((cuFloatComplex *)fftBuf.getPtr(),
                                   (cuFloatComplex *)otfArray.getPtr(),
                                   arraySize);
#ifndef NDEBUG
  std::cout << "filter_kernel(): " << cudaGetErrorString(cudaGetLastError())
            << std::endl;
#endif

  if (rfftplan2D == NULL)
    cuFFTErr = cufftExecC2R(rfftplanInv, (cuFloatComplex *)fftBuf.getPtr(),
                            (cufftReal *)img.getPtr());
  else {  // implying 2-step 3D IFFT
    // First, a batch of in-place 1D C2C IFFT along Z of all X-Y pixels:
    cuFFTErr = cufftExecC2C(rfftplan, (cuFloatComplex *) fftBuf.getPtr(),
                            (cuFloatComplex *) fftBuf.getPtr(), CUFFT_INVERSE);
    if (cuFFTErr != CUFFT_SUCCESS) {
      std::cout << "Line:" << __LINE__ << " in function " << __func__ << std::endl;
      throw std::runtime_error("cufft failed.");
    }
    // Second, 2D C2R FFT of all planes:
    for (auto z=0; z>=nz; z++) {
      cuFFTErr = cufftExecC2R(rfftplanInv, ((cuFloatComplex *) fftBuf.getPtr()) + z*strideC,
                              ((cufftReal *) img.getPtr()) + z*strideR);
      if (cuFFTErr != CUFFT_SUCCESS) break;
    }
  }

  if (cuFFTErr != CUFFT_SUCCESS) {
    std::cout << "Line:" << __LINE__ << " in function " << __func__ << std::endl;
    throw std::runtime_error("cufft failed.");
  }
}

// returns otfval at the given kx, ky, kz coordinates, based on the GPU 2D array "const_otf" which was loaded with "transferConstants"
__device__ cuFloatComplex dev_otfinterpolate(cuFloatComplex * d_rawotf,
                                             float kx, float ky, float kz)
/* (kx, ky, kz) is Fourier space coords with origin at kx=ky=kz=0 and going
   betwen -nx(or ny,nz)/2 and +nx(or ny,nz)/2 */
{
  cuFloatComplex otfval = make_cuFloatComplex(0.f, 0.f);
  float kzindex = (kz<0 ? kz+const_nzotf : kz);
  if (const_nyotf == 1) {// rotationally averaged raw OTF
    float krindex = sqrt(kx*kx + ky*ky);

    if (krindex < const_nxotf-1 && kzindex < const_nzotf) {
      int irindex, izindex, indices[2][2];
      float ar, az;

      irindex = floor(krindex);
      izindex = floor(kzindex);

      ar = krindex - irindex;
      az = kzindex - izindex;  // az is always 0 for 2D case, and it'll just become a 1D interp

      if (izindex == const_nzotf-1) {
        indices[0][0] = irindex*const_nzotf+izindex;
        indices[0][1] = irindex*const_nzotf;
        indices[1][0] = (irindex+1)*const_nzotf+izindex;
        indices[1][1] = (irindex+1)*const_nzotf;
      }
      else {
        indices[0][0] = irindex*const_nzotf+izindex;
        indices[0][1] = irindex*const_nzotf+(izindex+1);
        indices[1][0] = (irindex+1)*const_nzotf+izindex;
        indices[1][1] = (irindex+1)*const_nzotf+(izindex+1);
      }
      otfval.x = (1-ar)*(d_rawotf[indices[0][0]].x*(1-az) + d_rawotf[indices[0][1]].x*az) +
        ar*(d_rawotf[indices[1][0]].x*(1-az) + d_rawotf[indices[1][1]].x*az);
      otfval.y = (1-ar)*(d_rawotf[indices[0][0]].y*(1-az) + d_rawotf[indices[0][1]].y*az) +
        ar*(d_rawotf[indices[1][0]].y*(1-az) + d_rawotf[indices[1][1]].y*az);
    }
  }
  else {  // non-RA raw OTF
    float kxindex = kx; // because of half kx dimension; and no need for conj concern; check!
    float kyindex = (ky < 0? ky + const_nyotf : ky);
    if (kxindex < const_nxotf-1 && kyindex < const_nyotf && kzindex < const_nzotf) {
      int ixindex, iyindex, izindex, indices[2][2][2];
      float ax, ay, az;

      ixindex = floor(kxindex);
      iyindex = floor(kyindex);
      izindex = floor(kzindex);

      int iyindex_plus_1 = (iyindex+1) % const_nyotf;
      int izindex_plus_1 = (izindex+1) % const_nzotf;
      ax = kxindex - ixindex;
      ay = kyindex - iyindex;
      az = kzindex - izindex;
      int nxyotf = const_nxotf * const_nyotf;

      // Find the 8 vertices surrounding the point (kxindex, kyindex, kzindex)
      indices[0][0][0] = izindex*nxyotf        + iyindex*const_nxotf        + ixindex;
      indices[0][0][1] = izindex*nxyotf        + iyindex*const_nxotf        + (ixindex+1);
      indices[0][1][0] = izindex*nxyotf        + iyindex_plus_1*const_nxotf + ixindex;
      indices[0][1][1] = izindex*nxyotf        + iyindex_plus_1*const_nxotf + (ixindex+1);
      indices[1][0][0] = izindex_plus_1*nxyotf + iyindex*const_nxotf        + ixindex;
      indices[1][0][1] = izindex_plus_1*nxyotf + iyindex*const_nxotf        + (ixindex+1);
      indices[1][1][0] = izindex_plus_1*nxyotf + iyindex_plus_1*const_nxotf + ixindex;
      indices[1][1][1] = izindex_plus_1*nxyotf + iyindex_plus_1*const_nxotf + (ixindex+1);

      otfval.x = (1-az)*(d_rawotf[indices[0][0][0]].x*(1-ay)*(1-ax) +
                         d_rawotf[indices[0][0][1]].x*(1-ay)*ax +
                         d_rawotf[indices[0][1][0]].x*ay*(1-ax) +
                         d_rawotf[indices[0][1][1]].x*ay*ax)    +
        az*(d_rawotf[indices[1][0][0]].x*(1-ay)*(1-ax) +
            d_rawotf[indices[1][0][1]].x*(1-ay)*ax +
            d_rawotf[indices[1][1][0]].x*ay*(1-ax) +
            d_rawotf[indices[1][1][1]].x*ay*ax);
      otfval.y = (1-az)*(d_rawotf[indices[0][0][0]].y*(1-ay)*(1-ax) +
                         d_rawotf[indices[0][0][1]].y*(1-ay)*ax +
                         d_rawotf[indices[0][1][0]].y*ay*(1-ax) +
                         d_rawotf[indices[0][1][1]].y*ay*ax)    +
        az*(d_rawotf[indices[1][0][0]].y*(1-ay)*(1-ax) +
            d_rawotf[indices[1][0][1]].y*(1-ay)*ax +
            d_rawotf[indices[1][1][0]].y*ay*(1-ax) +
            d_rawotf[indices[1][1][1]].y*ay*ax);
    }
  }
  // This could be rewritten using Textures for the interpolation?
  // float krindex = sqrt(kx*kx + ky*ky) / const_nrotf;
  // float kzindex = (kz<0 ? kz+const_nzotf : kz) / const_nzotf;

  // cuFloatComplex otfval;

  // otfval.x = tex2D(texRef1, kzindex, krindex);
  // otfval.y = tex2D(texRef2, kzindex, krindex);
  return otfval;
}

__global__ void filter_kernel(cuFloatComplex *devImg, cuFloatComplex *devOTF,
                              int size)
{
  unsigned ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < size) {
    cuFloatComplex otf_val = devOTF[ind];
    devImg[ind] = cuCmulf(otf_val, devImg[ind]);
  }
}

__global__ void filterConj_kernel(cuFloatComplex *devImg,
                                  cuFloatComplex *devOTF, int size)
{
  unsigned ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < size) {
    cuFloatComplex otf_val = devOTF[ind];
    otf_val.y *= -1;
    devImg[ind] = cuCmulf(otf_val, devImg[ind]);
  }
}

__global__ void makeOTFarray_kernel(cuFloatComplex *src, cuFloatComplex *result)
{
  unsigned kx = blockIdx.x * blockDim.x + threadIdx.x;
  // x>>1 is equivalent to x/2 when x is integer
  int ky = blockIdx.y > const_ny >> 1 ? blockIdx.y - const_ny : blockIdx.y;
  int kz = blockIdx.z > const_nz >> 1 ? blockIdx.z - const_nz : blockIdx.z;

  int half_nx = (const_nx>>1) + 1;
  if (kx < half_nx) {
    cuFloatComplex otf_val = dev_otfinterpolate(src, kx*const_kxscale, ky*const_kyscale, kz*const_kzscale);
    unsigned ind = blockIdx.z * half_nx * const_ny  + blockIdx.y * half_nx + kx;
    result[ind].x = otf_val.x;
    result[ind].y = otf_val.y;
  }
}

__host__ void makeOTFarray(GPUBuffer &raw_otfarray, GPUBuffer &otfarray, int nx, int ny, int nz)
{
  unsigned nThreads = 128;
  dim3 block(nThreads, 1, 1);
  unsigned blockNx = ceil((nx / 2 + 1) / (float)nThreads);
  dim3 grid(blockNx, ny, nz);

  makeOTFarray_kernel<<<grid, block>>>((cuFloatComplex *) raw_otfarray.getPtr(),
                                       (cuFloatComplex *)otfarray.getPtr());
#ifndef NDEBUG
  std::cout << "makeOTFarray(): " << cudaGetErrorString(cudaGetLastError())
            << std::endl;
#endif
}

__global__ void scale_kernel(float *img, double factor)
{
  unsigned ind =
    (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  if (ind < const_nxyz)
    img[ind] *= factor;
}

__host__ void calcLRcore(GPUBuffer &reblurred, GPUBuffer &raw, int nx, int ny,
                         int nz, unsigned maxGridXdim)
// calculate raw image divided by reblurred, a key step in R-L;
// Both input, "reblurred" and "raw", are of dimension (nx, ny, nz) and of
// floating type; "reblurred" is updated upon return.
{
  unsigned nThreads = 1024;
  unsigned NXBlocks = ceil(((float)(nx * ny * nz)) / nThreads);
  unsigned NYBlocks = 1;
  if (NXBlocks > maxGridXdim)
    NYBlocks = NXBlocks = ceil(sqrt(NXBlocks));

  dim3 grid(NXBlocks, NYBlocks);
  dim3 block(nThreads);

  LRcore_kernel<<<grid, block>>>((float *)reblurred.getPtr(),
                                 (float *)raw.getPtr());
#ifndef NDEBUG
  std::cout << "calcLRcore(): " << cudaGetErrorString(cudaGetLastError())
            << std::endl;
#endif
}

__global__ void LRcore_kernel(float *img1, float *img2)
//! Calculate img2/img1; results returned in img1
{
  unsigned ind =
    (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

  if (ind < const_nxyz) {
    img1[ind] = fabs(img1[ind]) > 0 ? img1[ind] : const_eps;
    // img1[ind] = img1[ind] > const_eps ? img1[ind] :
    //   (img1[ind] > 0 ? const_eps : img2[ind]);
    img1[ind] = img2[ind] / img1[ind] + const_eps;
    // The following thresholding is necessary for occasional very high
    // DR data and incorrectly high background value specified (-b flag).
    if (!const_bNoLimitRatio) {
      if (img1[ind] > 10)
        img1[ind] = 10;
      if (img1[ind] < -10)
        img1[ind] = -10;
    }
  }
}

__host__ void updateCurrEstimate(GPUBuffer &X_k, GPUBuffer &CC, GPUBuffer &Y_k,
                                 int nx, int ny, int nz, unsigned maxGridXdim)
// calculate updated current estimate: Y_k * CC plus positivity constraint
// All inputs are of dimension (nx+2, ny, nz) and of floating type;
// "X_k" is updated upon return.
{
  unsigned nThreads = 1024;
  unsigned NXBlocks = ceil(((float)(nx * ny * nz)) / nThreads);
  unsigned NYBlocks = 1;
  if (NXBlocks > maxGridXdim)
    NYBlocks = NXBlocks = ceil(sqrt(NXBlocks));

  dim3 grid(NXBlocks, NYBlocks);
  dim3 block(nThreads);

  currEstimate_kernel<<<grid, block>>>((float *)X_k.getPtr(), (float *)CC.getPtr(),
                                       (float *)Y_k.getPtr());
#ifndef NDEBUG
  std::cout << "updateCurrEstimate(): "
            << cudaGetErrorString(cudaGetLastError()) << std::endl;
#endif
}

__global__ void currEstimate_kernel(float *img1, float *img2, float *img3)
{
  unsigned ind = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

  if (ind < const_nxyz) {
    img1[ind] = img2[ind] * img3[ind];
    img1[ind] = img1[ind] > 0 ? img1[ind] : 0;
  }
}

__host__ void calcCurrPrevDiff(GPUBuffer &X_k, GPUBuffer &Y_k,
                               GPUBuffer &G_kminus1, int nx, int ny, int nz,
                               unsigned maxGridXdim)
// calculate X_k - Y_k and assign the result to G_kminus1;
// All inputs are of dimension (nx+2, ny, nz) and of floating type;
// "X_k" is updated upon return.
{
  unsigned nThreads = 1024;

  unsigned NXBlocks = ceil(((float)(nx * ny * nz)) / nThreads);
  unsigned NYBlocks = 1;
  if (NXBlocks > maxGridXdim)
    NYBlocks = NXBlocks = ceil(sqrt(NXBlocks));

  dim3 grid(NXBlocks, NYBlocks);
  dim3 block(nThreads);

  currPrevDiff_kernel<<<grid, block>>>((float *)X_k.getPtr(),
                                       (float *)Y_k.getPtr(),
                                       (float *)G_kminus1.getPtr());
#ifndef NDEBUG
  std::cout << "calcCurrPrevDiff(): " << cudaGetErrorString(cudaGetLastError())
            << std::endl;
#endif
}

__global__ void currPrevDiff_kernel(float *img1, float *img2, float *img3)
{
  // compute x, y, z indices based on block and thread indices
  unsigned ind =
      (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

  if (ind < const_nxyz)
    img3[ind] = img1[ind] - img2[ind];
}

__host__ double calcAccelFactor(GPUBuffer &G_km1, GPUBuffer &G_km2, int nx,
                                int ny, int nz, float eps, int myGPUdevice)
// (G_km1 dot G_km2) / (G_km2 dot G_km2)
// All inputs are of dimension (nx, ny, nz) and of floating type;
{
  unsigned nThreads = 1024; // Maximum number of threads per block for C2070,
                            // M2090, or Quadro 4000
  unsigned nBlocks = ceil(((float)(nx * ny * nz)) / nThreads / 2);

  // Used for holding partial reduction results; one for each thread block:
  GPUBuffer devBuf1(nBlocks * sizeof(double) * 2, myGPUdevice, false);
  // First nBlocks: numerator; second nBlocks: denominator

  unsigned smemSize = nThreads * sizeof(double) * 2;
  innerProduct_kernel<<<nBlocks, nThreads, smemSize>>>(
      (float *)G_km1.getPtr(), (float *)G_km2.getPtr(),
      (double *)devBuf1.getPtr());
  double numerator = 0, denom = 0;

  CPUBuffer h_numer_denom(devBuf1); // This copy is going to be sloooow.

  double *ptr = (double *)h_numer_denom.getPtr();
  for (unsigned i = 0; i < nBlocks; i++) {
    numerator += *ptr;
    denom += *(ptr + nBlocks);
    ptr++;
  }

  /*

// Use Thrust to do summation on GPU...but it's the same speed.  sigh.

// first we need to convert devBuf1 to thrust:
https://stackoverflow.com/a/10972841

  // obtain raw pointer to device memory
  //int * raw_ptr;
  //cudaMalloc((void **)&raw_ptr, N * sizeof(int));

  // wrap raw pointer with a device_ptr
  thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast((double *)
devBuf1.getPtr());

  // use device_ptr in Thrust algorithms
  double sum = 0;
  sum = thrust::reduce(dev_ptr, dev_ptr + nBlocks); //
https://thrust.github.io/doc/group__reductions_ga69434d74f2e6117040fb38d1a28016c2.html#ga69434d74f2e6117040fb38d1a28016c2
  numerator = sum;
  // std::cout << "numerator: " << numerator;

  sum = thrust::reduce(dev_ptr + nBlocks, dev_ptr + nBlocks + nBlocks);
  denom = sum;
  // std::cout << "  denom: " << denom;
  */

  double accelfactor = numerator / (denom + eps);
  // std::cout << "  accelfactor: " << accelfactor << std::endl;

  return accelfactor;
}

__global__ void innerProduct_kernel(float *img1, float *img2, double *intRes1)
// Using reduction to implement two inner products (img1.dot.img2 and
// img2.dot.img2) Copied from CUDA "reduction" sample code reduce4()
{
  double *sdata = SharedMemory<double>();
  // shared memory; first half for img1.dot.img2;
  // second half for img2.dot.img2

  unsigned tid = threadIdx.x;
  unsigned ind = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  double mySum1 = 0, mySum2 = 0;
  if (ind < const_nxyz) {
    mySum1 = img1[ind] * img2[ind];
    mySum2 = img2[ind] * img2[ind];
  }

  unsigned indPlusBlockDim = ind + blockDim.x;
  if (indPlusBlockDim < const_nxyz) {
    mySum1 += img1[indPlusBlockDim] * img2[indPlusBlockDim];
    mySum2 += img2[indPlusBlockDim] * img2[indPlusBlockDim];
  }

  sdata[tid] = mySum1;
  sdata[tid + blockDim.x] = mySum2;
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
      sdata[tid + blockDim.x] += sdata[tid + s + blockDim.x];
    }

    __syncthreads();
  }

  if (tid < 32) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile double *smem1 = sdata;

    // Assuming blockSize is > 64:
    smem1[tid] += smem1[tid + 32];
    smem1[tid] += smem1[tid + 16];
    smem1[tid] += smem1[tid + 8];
    smem1[tid] += smem1[tid + 4];
    smem1[tid] += smem1[tid + 2];
    smem1[tid] += smem1[tid + 1];
    smem1[tid + blockDim.x] += smem1[tid + 32 + blockDim.x];
    smem1[tid + blockDim.x] += smem1[tid + 16 + blockDim.x];
    smem1[tid + blockDim.x] += smem1[tid + 8 + blockDim.x];
    smem1[tid + blockDim.x] += smem1[tid + 4 + blockDim.x];
    smem1[tid + blockDim.x] += smem1[tid + 2 + blockDim.x];
    smem1[tid + blockDim.x] += smem1[tid + 1 + blockDim.x];
  }
  // write result for this block to global mem
  if (tid == 0) {
    intRes1[blockIdx.x] = sdata[0];
    intRes1[blockIdx.x + gridDim.x] = sdata[blockDim.x];
  }
}

__host__ void updatePrediction(GPUBuffer &Y_k, GPUBuffer &X_k,
                               GPUBuffer &X_kminus1, double lambda, int nx,
                               int ny, int nz, unsigned maxGridXdim)
{
  // Y_k = X_k + lambda * (X_k - X_kminus1)
  unsigned nThreads = 1024; // Maximum number of threads per block for C2070,
                            // M20990, or Quadro 4000
  unsigned NXblock = ceil(nx * ny * nz / (float)nThreads);
  unsigned NYblock = 1;
  if (NXblock > maxGridXdim)
    NYblock = NXblock = ceil(sqrt(NXblock));

  dim3 grid(NXblock, NYblock);
  dim3 block(nThreads);

  updatePrediction_kernel<<<grid, block>>>((float *)Y_k.getPtr(),
                                           (float *)X_k.getPtr(),
                                           (float *)X_kminus1.getPtr(), lambda);
#ifndef NDEBUG
  std::cout << "updatePrediction(): " << cudaGetErrorString(cudaGetLastError())
            << std::endl;
#endif
}

__global__ void updatePrediction_kernel(float *Y_k, float *X_k, float *X_km1,
                                        float lambda) {
  unsigned ind =
      (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  if (ind < const_nxyz) {
    Y_k[ind] = X_k[ind] + lambda * (X_k[ind] - X_km1[ind]);
    Y_k[ind] = (Y_k[ind] > 0) ? Y_k[ind] : 0;
  }
}

__host__ double meanAboveBackground_GPU(GPUBuffer &img, int nx, int ny, int nz,
                                        unsigned maxGridXdim, int myGPUdevice)
{
  unsigned nThreads = 1024;
  unsigned nXblocks = ceil(nx * ny * nz / (float)nThreads / 2);
  unsigned nYblocks = 1;
  if (nXblocks > maxGridXdim)
    nYblocks = nXblocks = ceil(sqrt(nXblocks));

  unsigned smemSize = nThreads * sizeof(double);

  // used for holding intermediate reduction results; one for each thread block
  GPUBuffer d_intres(nYblocks * nXblocks * sizeof(double), myGPUdevice, false);

  summation_kernel<<<dim3(nXblocks, nYblocks), nThreads, smemSize>>>(
      (float *)img.getPtr(), (double *)d_intres.getPtr(), nx * ny * nz);
  // download intermediate results to host:
  CPUBuffer intRes(d_intres);
  double sum = 0;
  double *p = (double *)intRes.getPtr();
  for (unsigned i = 0; i < nXblocks * nYblocks; i++)
    sum += *p++;

  float mean = sum / (nx * ny * nz);

  GPUBuffer d_counter(nXblocks * nYblocks * sizeof(unsigned), myGPUdevice,
                      false);
  smemSize = nThreads * (sizeof(double) + sizeof(unsigned));
  sumAboveThresh_kernel<<<dim3(nXblocks, nYblocks), nThreads, smemSize>>>(
      (float *)img.getPtr(), (double *)d_intres.getPtr(),
      (unsigned *)d_counter.getPtr(), mean, nx * ny * nz);

  // download intermediate results to host:
  CPUBuffer counter(d_counter);
  intRes = d_intres;
  sum = 0;
  unsigned count = 0;
  p = (double *)intRes.getPtr();
  unsigned *pc = (unsigned *)counter.getPtr();
  for (unsigned i = 0; i < nXblocks * nYblocks; i++) {
    sum += *p++;
    count += *pc++;
  }

#ifndef NDEBUG
  printf("mean=%f, sum=%lf, count=%d\n", mean, sum, count);
#endif
  return sum / count;
}

__global__ void summation_kernel(float *img, double *intRes, int n)
// Copied from CUDA "reduction" sample code reduce4()
{
  double *sdata = SharedMemory<double>();

  unsigned tid = threadIdx.x;
  unsigned ind =
      (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * 2 + threadIdx.x;

  double mySum = (ind < n) ? img[ind] : 0;

  if (ind + blockDim.x < n)
    mySum += img[ind + blockDim.x];

  sdata[tid] = mySum;
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile double *smem = sdata;

    // Assuming blockSize is > 64:
    smem[tid] += smem[(tid + 32)];
    smem[tid] += smem[(tid + 16)];
    smem[tid] += smem[(tid + 8)];
    smem[tid] += smem[(tid + 4)];
    smem[tid] += smem[(tid + 2)];
    smem[tid] += smem[(tid + 1)];
  }
  // write result for this block to global mem
  if (tid == 0)
    intRes[blockIdx.y*gridDim.x+blockIdx.x] = sdata[0];
}

__global__ void sumAboveThresh_kernel(float *img, double *intRes,
                                      unsigned *counter, float thresh, int n)
// Adapted from CUDA "reduction" sample code reduce4()
{
  // Size of shared memory allocated is nThreads * (sizeof(double) +
  // sizeof(unsigned)) The first nThreads * sizeof(double) bytes are used for
  // image intensity sum; the next nThreads * sizeof(unsigned) bytes are for
  // counting pixels whose intensity is > thresh
  double *sdata = SharedMemory<double>();
  unsigned *count = (unsigned *)(sdata + blockDim.x);

  unsigned tid = threadIdx.x;
  unsigned ind =
      (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * 2 + threadIdx.x;

  double mySum = 0;
  unsigned myCount = 0;
  if (ind < n && img[ind] > thresh) {
    mySum = img[ind];
    myCount++;
  }

  unsigned ind2 = ind + blockDim.x;
  if (ind2 < n && img[ind2] > thresh) {
    mySum += img[ind2];
    myCount++;
  }

  sdata[tid] = mySum;
  count[tid] = myCount;
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
      count[tid] += count[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    volatile double *smem = sdata;
    volatile unsigned *cmem = count;

    smem[tid] += smem[(tid + 32)];
    smem[tid] += smem[(tid + 16)];
    smem[tid] += smem[(tid + 8)];
    smem[tid] += smem[(tid + 4)];
    smem[tid] += smem[(tid + 2)];
    smem[tid] += smem[(tid + 1)];
    cmem[tid] += cmem[(tid + 32)];
    cmem[tid] += cmem[(tid + 16)];
    cmem[tid] += cmem[(tid + 8)];
    cmem[tid] += cmem[(tid + 4)];
    cmem[tid] += cmem[(tid + 2)];
    cmem[tid] += cmem[(tid + 1)];
  }
  // write result for this block to global mem
  if (tid == 0) {
    intRes [blockIdx.y*gridDim.x+blockIdx.x] = sdata[0];
    counter[blockIdx.y*gridDim.x+blockIdx.x] = count[0];
  }
}

__host__ void rescale_GPU(GPUBuffer &img, int nx, int ny, int nz, float scale,
                          unsigned maxGridXdim) {
  unsigned nThreads = 1024;
  unsigned NXBlocks = ceil(nx * ny * nz / (float)nThreads);
  unsigned NYBlocks = 1;
  if (NXBlocks > maxGridXdim)
    NYBlocks = NXBlocks = ceil(sqrt(NXBlocks));

  scale_kernel<<<dim3(NXBlocks, NYBlocks), nThreads>>>((float *)img.getPtr(),
                                                       scale);
#ifndef NDEBUG
  std::cout << "rescale_GPU(): " << cudaGetErrorString(cudaGetLastError())
            << std::endl;
#endif
}

__host__ void apodize_GPU(GPUBuffer &image, int nx, int ny, int nz,
                          int napodize)
{
  unsigned blockSize = 64;
  dim3 grid;
  grid.x = ceil((float)nx / blockSize);
  grid.y = nz;
  grid.z = 1;

  apodize_x_kernel<<<grid, blockSize>>>(napodize, nx, ny,
                                        ((float *)image.getPtr()));

  grid.x = ceil((float)ny / blockSize);
  apodize_y_kernel<<<grid, blockSize>>>(napodize, nx, ny,
                                        ((float *)image.getPtr()));
#ifndef NDEBUG
  std::cout << __func__ <<"(): " << cudaGetErrorString(cudaGetLastError())
            << std::endl;
#endif
}

__global__ void apodize_x_kernel(int napodize, int nx, int ny, float *image) {
  unsigned k = blockDim.x * blockIdx.x + threadIdx.x;
  if (k < nx) {
    unsigned section_offset = blockIdx.y * nx * ny;
    float diff = (image[section_offset + (ny - 1) * nx + k] -
                  image[section_offset + k]) /
                 2.0;
    for (int l = 0; l < napodize; ++l) {
      float fact =
          diff * (1.0 - sin((((float)l + 0.5) / (float)napodize) * M_PI * 0.5));
      image[section_offset + l * nx + k] += fact;
      image[section_offset + (ny - 1 - l) * nx + k] -= fact;
    }
  }
}

__global__ void apodize_y_kernel(int napodize, int nx, int ny, float *image)
{
  int l = blockDim.x * blockIdx.x + threadIdx.x;
  if (l < ny) {
    unsigned section_offset = blockIdx.y * nx * ny;
    float diff = (image[section_offset + l * nx + nx - 1] -
                  image[section_offset + l * nx]) /
                 2.0;
    for (int k = 0; k < napodize; ++k) {
      float fact =
          diff * (1.0 - sin(((k + 0.5) / (float)napodize) * M_PI * 0.5));
      image[section_offset + l * nx + k] += fact;
      image[section_offset + l * nx + nx - 1 - k] -= fact;
    }
  }
}

__host__ void zBlend_GPU(GPUBuffer &image, int nx, int ny, int nz,
                         int nZblend)
{
  dim3 block(32, 32);
  dim3 grid;
  grid.x = ceil((float)nx / 32);
  grid.y = ceil((float)ny / 32);
  grid.z = nZblend;

  zBlend_kernel<<<grid, block>>>(nx, ny, nz, nZblend,
                                 ((float *)image.getPtr()));
}

__global__ void zBlend_kernel(int nx, int ny, int nz, int nZblend,
                              float *image)
{
  unsigned xidx = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned yidx = blockDim.y * blockIdx.y + threadIdx.y;

  unsigned zidx = blockDim.z;

  if (xidx < nx && yidx < ny) {
    unsigned nxy = nx * ny;
    unsigned row_offset = yidx * nx;
    float diff =
        image[(nz - 1) * nxy + row_offset + xidx] - image[row_offset + xidx];
    float fact =
        diff * (1.0 - sin(((zidx + 0.5) / (float)nZblend) * M_PI * 0.5));
    image[zidx * nxy + row_offset + xidx] += fact;
    image[(nz - zidx - 1) * nxy + row_offset + xidx] -= fact;
  }
}
