#include "cutilSafeCall.h"

#include <CPUBuffer.h>
#include <GPUBuffer.h>
#include <cufft.h>

__constant__ int const_nx;
__constant__ int const_ny;
__constant__ int const_nz;
__constant__ unsigned const_nxyz;
__constant__ int const_nrotf;
__constant__ int const_nzotf;

__constant__ float const_kxscale;
__constant__ float const_kyscale;
__constant__ float const_kzscale;
__constant__ float const_eps;
__constant__ cuFloatComplex const_otf[7680]; // 60 kB should be enough for an OTF array??

__global__ void filter_kernel(cuFloatComplex *devImg, cuFloatComplex *devOTF, int size, bool bConj);
__global__ void scale_kernel(float * img, double factor);
__global__ void LRcore_kernel(float * img1, float * img2);
__global__ void currEstimate_kernel(float * img1, float * img2, float * img3);
__global__ void currPrevDiff_kernel(float * img1, float * img2, float * img3);
__global__ void innerProduct_kernel(float * img1, float * img2,
                                    double * intRes1, double * intRes2);
__global__ void updatePrediction_kernel(float * Y_k, float * X_k, float *X_km1, float lambda);

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
// (Copied from reduction_kernel.cu of CUDA samples)
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};


texture<float, cudaTextureType2D, cudaReadModeElementType> texRef1, texRef2;
cudaArray* d_realpart, *d_imagpart;  // used for OTF texture


__host__ void transferConstants(int nx, int ny, int nz, int nrotf, int nzotf,
                                float kxscale, float kyscale, float kzscale,
                                float eps, float * h_otf)
{
  cutilSafeCall(cudaMemcpyToSymbol(const_nx, &nx, sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(const_ny, &ny, sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(const_nz, &nz, sizeof(int)));
  unsigned int nxyz = nx*ny*nz;
  cutilSafeCall(cudaMemcpyToSymbol(const_nxyz, &nxyz, sizeof(unsigned int)));
  cutilSafeCall(cudaMemcpyToSymbol(const_nrotf, &nrotf, sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(const_nzotf, &nzotf, sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(const_kxscale, &kxscale, sizeof(float)));
  cutilSafeCall(cudaMemcpyToSymbol(const_kyscale, &kyscale, sizeof(float)));
  cutilSafeCall(cudaMemcpyToSymbol(const_kzscale, &kzscale, sizeof(float)));
  cutilSafeCall(cudaMemcpyToSymbol(const_eps, &eps, sizeof(float)));
  cutilSafeCall(cudaMemcpyToSymbol(const_otf, h_otf, nrotf*nzotf*2*sizeof(float)));
}

__host__ void prepareOTFtexture(float * realpart, float * imagpart, int nx, int ny)
{
  // Allocate CUDA array in device memory
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

  cudaMallocArray(&d_realpart, &channelDesc, nx, ny);
  cudaMallocArray(&d_imagpart, &channelDesc, nx, ny);

  // Copy to device memory
  cudaMemcpyToArray(d_realpart, 0, 0, realpart,
                    nx * ny * sizeof(float),
                    cudaMemcpyHostToDevice);
  cudaMemcpyToArray(d_imagpart, 0, 0, imagpart,
                    nx * ny * sizeof(float),
                    cudaMemcpyHostToDevice);

  // Set texture reference parameters
  texRef1.addressMode[0] = cudaAddressModeClamp;
  texRef1.addressMode[1] = cudaAddressModeClamp;
  texRef1.filterMode = cudaFilterModeLinear;
  texRef1.normalized = true;
  texRef2.addressMode[0] = cudaAddressModeClamp;
  texRef2.addressMode[1] = cudaAddressModeClamp;
  texRef2.filterMode = cudaFilterModeLinear;
  texRef2.normalized = true;
  // Bind the arrays to the texture reference
  cudaBindTextureToArray(texRef1, d_realpart, channelDesc);
  cudaBindTextureToArray(texRef2, d_imagpart, channelDesc);
}

__global__ void bgsubtr_kernel(float * img, int size, float background)
{
  int ind = blockIdx.x * blockDim.x + threadIdx.x;

  if (ind < size) {
    img[ind] -= background;
    img[ind] = img[ind] > 0 ? img[ind] : 0;
  }
}

__host__ void backgroundSubtraction_GPU(GPUBuffer &img, int nx, int ny, int nz, float background)
{
  int nThreads = 1024;
  int NXblock = (int) ceil( nx*ny*nz /(float) nThreads );
  dim3 grid(NXblock, 1, 1);
  dim3 block(nThreads, 1, 1);

  bgsubtr_kernel<<<grid, block>>>((float *) img.getPtr(), nx*ny*nz, background);
}

__host__ void filterGPU(GPUBuffer &img, int nx, int ny, int nz,
                        // GPUBuffer &otf,
                        cufftHandle & rfftplan, cufftHandle & rfftplanInv,
                        GPUBuffer &fftBuf,
                        GPUBuffer &otfArray, bool bConj)
// "img" is of dimension (nx, ny, nz) and of float type
// "otf" is of dimension (const_nzotf, const_nrotf) and of complex type
{
  cufftResult cuFFTErr = cufftExecR2C(rfftplan, (cufftReal *) img.getPtr(),
                                      (cuFloatComplex *) fftBuf.getPtr());

  if (cuFFTErr != CUFFT_SUCCESS) {
    std::cout << "Line:" << __LINE__ << std::endl;
    throw std::runtime_error("cufft failed.");
  }
  //
  // KERNEL 1
  //
  int nThreads = 1024;
  int arraySize = nz * ny * (nx/2+1);
  int NXblock = (int) ceil( arraySize / (float) nThreads );
  dim3 grid(NXblock);
  dim3 block(nThreads);

  filter_kernel<<<grid, block>>>((cuFloatComplex*) fftBuf.getPtr(),
                                 (cuFloatComplex*) otfArray.getPtr(),
                                 arraySize, bConj);

  cuFFTErr = cufftExecC2R(rfftplanInv, (cuFloatComplex*)fftBuf.getPtr(), (cufftReal *) img.getPtr());

  if (cuFFTErr != CUFFT_SUCCESS) {
    std::cout << "Line:" << __LINE__ ;
    throw std::runtime_error("cufft failed.");
  }

  //
  // Rescale KERNEL
  //
  nThreads = 1024;
  NXblock = (int) ceil( ((float)(nx*ny*nz)) / nThreads );
  scale_kernel<<<NXblock, nThreads>>>((float *) img.getPtr(), 1./(nx*ny*nz));
}

__device__ cuFloatComplex dev_otfinterpolate(// cuFloatComplex * d_otf, 
                                             float kx, float ky, float kz)
  /* (kx, ky, kz) is Fourier space coords with origin at kx=ky=kz=0 and going  betwen -nx(or ny,nz)/2 and +nx(or ny,nz)/2 */
{
  float krindex = sqrt(kx*kx + ky*ky);
  float kzindex = (kz<0 ? kz+const_nzotf : kz);

  cuFloatComplex otfval = make_cuFloatComplex(0.f, 0.f);

  if (krindex < const_nrotf-1 && kzindex < const_nzotf) {
  // This should be rewritten using Textures for the interpolation. It will be much easier and faster!
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
    otfval.x = (1-ar)*(const_otf[indices[0][0]].x*(1-az) + const_otf[indices[0][1]].x*az) +
      ar*(const_otf[indices[1][0]].x*(1-az) + const_otf[indices[1][1]].x*az);
    otfval.y = (1-ar)*(const_otf[indices[0][0]].y*(1-az) + const_otf[indices[0][1]].y*az) +
      ar*(const_otf[indices[1][0]].y*(1-az) + const_otf[indices[1][1]].y*az);
  }

  // float krindex = sqrt(kx*kx + ky*ky) / const_nrotf;
  // float kzindex = (kz<0 ? kz+const_nzotf : kz) / const_nzotf;

  // cuFloatComplex otfval;

  // otfval.x = tex2D(texRef1, kzindex, krindex);
  // otfval.y = tex2D(texRef2, kzindex, krindex);
  return otfval;
}

__global__ void filter_kernel(cuFloatComplex *devImg, cuFloatComplex *devOTF, int size, bool bConj)
{
  int ind = blockIdx.x * blockDim.x + threadIdx.x;

  if ( ind < size ) {
    cuFloatComplex otf_val = devOTF[ind];
    if (bConj)
      otf_val.y *= -1;
    devImg[ind] = cuCmulf(otf_val, devImg[ind]);
  }
}


__global__ void makeOTFarray_kernel(cuFloatComplex *result)
{
  int kx = blockIdx.x * blockDim.x + threadIdx.x;
  int ky = blockIdx.y > const_ny/2 ? blockIdx.y - const_ny : blockIdx.y;
  int kz = blockIdx.z > const_nz/2 ? blockIdx.z - const_nz : blockIdx.z;

  if (kx < const_nx/2+1) {
    cuFloatComplex otf_val = dev_otfinterpolate(kx*const_kxscale, ky*const_kyscale, kz*const_kzscale);
    unsigned ind = blockIdx.z * (const_nx/2+1) * const_ny  + blockIdx.y * (const_nx/2+1) + kx;
    result[ind].x = otf_val.x;
    result[ind].y = otf_val.y;
  }
}

__host__ void makeOTFarray(GPUBuffer &otfarray, int nx, int ny, int nz)
{
  unsigned nThreads=128;
  dim3 block(nThreads, 1, 1);
  unsigned blockNx = (int) ceil( (nx/2+1) / (float) nThreads );
  dim3 grid(blockNx, ny, nz);

  makeOTFarray_kernel<<<grid, block>>>( (cuFloatComplex *) otfarray.getPtr());
  std::cout<< cudaGetErrorString(cudaGetLastError()) << std::endl;
}

__global__ void scale_kernel(float * img, double factor)
{
  unsigned ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < const_nxyz)
    img[ind] *= factor;
}


__host__ void calcLRcore(GPUBuffer &reblurred, GPUBuffer &raw, int nx, int ny, int nz)
// calculate raw image divided by reblurred, a key step in R-L;
// Both input, "reblurred" and "raw", are of dimension (nx, ny, nz) and of floating type;
// "reblurred" is updated upon return.
{
  int nThreads = 1024;
  int NXblock = (int) ceil( ((float) (nx*ny*nz)) /nThreads );
  dim3 grid(NXblock, 1, 1);
  dim3 block(nThreads, 1, 1);

  LRcore_kernel<<<grid, block>>>((float *) reblurred.getPtr(), (float *) raw.getPtr());
}

__global__ void LRcore_kernel(float * img1, float * img2)
// Calculate img2/img1; results returned in img1
{
  int ind = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (ind < const_nxyz) {
    img1[ind] = img1[ind] > const_eps ? img1[ind] : const_eps;
    img1[ind] = img2[ind] / img1[ind];
  }
}

__host__ void updateCurrEstimate(GPUBuffer &X_k, GPUBuffer &CC, GPUBuffer &Y_k,
                                 int nx, int ny, int nz)
// calculate updated current estimate: Y_k * CC plus positivity constraint
// All inputs are of dimension (nx+2, ny, nz) and of floating type;
// "X_k" is updated upon return.
{
  int nThreads = 1024;
  int NXblock = (int) ceil( ((float) (nx*ny*nz)) / nThreads );
  dim3 grid(NXblock, 1, 1);
  dim3 block(nThreads, 1, 1);

  currEstimate_kernel<<<grid, block>>>((float *) X_k.getPtr(),
                                       (float *) CC.getPtr(),
                                       (float *) Y_k.getPtr());
}

__global__ void currEstimate_kernel(float * img1, float * img2, float * img3)
{
  int ind = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (ind < const_nxyz) {
    img1[ind] = img2[ind] * img3[ind];
    img1[ind] = img1[ind] > 0 ? img1[ind] : 0;
  }
}

__host__ void calcCurrPrevDiff(GPUBuffer &X_k, GPUBuffer &Y_k, GPUBuffer &G_kminus1,
                               int nx, int ny, int nz)
// calculate X_k - Y_k and assign the result to G_kminus1;
// All inputs are of dimension (nx+2, ny, nz) and of floating type;
// "X_k" is updated upon return.
{
  int nThreads = 1024; //128;
  int NXblock = (int) ceil( ((float) (nx*ny*nz)) / nThreads );
  dim3 grid(NXblock, 1, 1);
  dim3 block(nThreads, 1, 1);

  currPrevDiff_kernel<<<grid, block>>>((float *) X_k.getPtr(),
                                       (float *) Y_k.getPtr(),
                                       (float *) G_kminus1.getPtr());
}

__global__ void currPrevDiff_kernel(float * img1, float * img2, float * img3)
{
  // compute x, y, z indices based on block and thread indices
  int ind = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (ind < const_nxyz)
    img3[ind] = img1[ind] - img2[ind];
}

__host__ double calcAccelFactor(GPUBuffer &G_km1, GPUBuffer &G_km2,
                                int nx, int ny, int nz, float eps)
// (G_km1 dot G_km2) / (G_km2 dot G_km2)
// All inputs are of dimension (nx, ny, nz) and of floating type;
{
  int nThreads = 1024; // Maximum number of threads per block for C2070, M20990, or Quadro 4000
  int nBlocks = (int) ceil( ((float) (nx*ny*nz)) / nThreads/2 );

  // used for holding partial reduction results; one for each thread block
  GPUBuffer devBuf1(nBlocks * sizeof(double), 0);
  GPUBuffer devBuf2(nBlocks * sizeof(double), 0);

  unsigned smemSize = nThreads * sizeof(double) * 2;
  innerProduct_kernel<<<nBlocks, nThreads, smemSize>>>((float *) G_km1.getPtr(),
                                                       (float *) G_km2.getPtr(),
                                                       (double *) devBuf1.getPtr(),
                                                       (double *) devBuf2.getPtr());

  CPUBuffer h_Numerator(devBuf1);
  CPUBuffer h_Denominator (devBuf2);

  double numerator=0, denom=0;
  for (int i=0; i<nBlocks; i++) {
    numerator += *((double *) h_Numerator.getPtr() + i);
    denom   += *((double *) h_Denominator.getPtr() + i);
  }

  return numerator / (denom + eps);
}

__global__ void innerProduct_kernel(float * img1, float * img2,
                                    double * intRes1, double * intRes2)
// Using reduction to implement two inner products (img1.dot.img2 and img2.dot.img2)
// Copied from CUDA "reduction" sample code reduce4()
{
  double *sdata1 = SharedMemory<double>();
  // shared memory; even-numbered indices for img1.dot.img2;
  // odd-numbered indices for img2.dot.img2

  unsigned tid = threadIdx.x;
  unsigned ind = blockIdx.x * blockDim.x*2 + threadIdx.x;

  double mySum1=0, mySum2=0;
  if (ind< const_nxyz) {
    mySum1 = img1[ind] * img2[ind];
    mySum2 = img2[ind] * img2[ind];
  }

  unsigned indPlusBlockDim = ind + blockDim.x;
  if (indPlusBlockDim < const_nxyz) {
    mySum1 += img1[indPlusBlockDim] * img2[indPlusBlockDim];
    mySum2 += img2[indPlusBlockDim] * img2[indPlusBlockDim];
  }

  sdata1[2*tid] = mySum1;
  sdata1[2*tid + 1] = mySum2;
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
    if (tid < s) {
      sdata1[2*tid] += sdata1[2*(tid + s)];
      sdata1[2*tid +1] += sdata1[2*(tid + s) +1];
    }

    __syncthreads();
  }

  if (tid < 32) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile double *smem1 = sdata1;
    // volatile double *smem2 = sdata2;

    smem1[2*tid] += smem1[2*(tid + 32)];
    smem1[2*tid] += smem1[2*(tid + 16)];
    smem1[2*tid] += smem1[2*(tid +  8)];
    smem1[2*tid] += smem1[2*(tid +  4)];
    smem1[2*tid] += smem1[2*(tid +  2)];
    smem1[2*tid] += smem1[2*(tid +  1)];
    smem1[2*tid+1] += smem1[2*(tid + 32)+1];
    smem1[2*tid+1] += smem1[2*(tid + 16)+1];
    smem1[2*tid+1] += smem1[2*(tid +  8)+1];
    smem1[2*tid+1] += smem1[2*(tid +  4)+1];
    smem1[2*tid+1] += smem1[2*(tid +  2)+1];
    smem1[2*tid+1] += smem1[2*(tid +  1)+1];
  }
  // write result for this block to global mem
  if (tid == 0) {
    intRes1[blockIdx.x] = sdata1[0];
    intRes2[blockIdx.x] = sdata1[1];
  }
}

__host__ void updatePrediction(GPUBuffer &Y_k, GPUBuffer &X_k, GPUBuffer &X_kminus1,
                               double lambda, int nx, int ny, int nz)
{
  // Y_k = X_k + lambda * (X_k - X_kminus1)
  int nxyz = nx*ny*nz;
  int nThreads = 1024; // Maximum number of threads per block for C2070, M20990, or Quadro 4000
  int nBlocks = (int) ceil( ((float) nxyz) / nThreads );

  updatePrediction_kernel<<<nBlocks, nThreads>>>((float *) Y_k.getPtr(),
                                                 (float *) X_k.getPtr(),
                                                 (float *) X_kminus1.getPtr(),
                                                 lambda);
}

__global__ void updatePrediction_kernel(float * Y_k, float * X_k, float *X_km1, float lambda)
{
  unsigned ind = blockIdx.x * blockDim.x + threadIdx.x;
  if (ind < const_nxyz) {
    Y_k[ind] = X_k[ind] + lambda * (X_k[ind] - X_km1[ind]);
    Y_k[ind] = (Y_k[ind] > 0) ? Y_k[ind] : 0;
  }
}
