#include "linearDecon.h"


//#include <vector>
//#include <algorithm>  // max(), min()
//#include <limits>  // epsilon()
#include "../cutilSafeCall.h"


#ifdef _WIN32
#ifdef LLSPY_IMPORT
  #define LLSPY_API __declspec( dllimport )
#else
  #define LLSPY_API __declspec( dllexport )
#endif
#else
  #define LLSPY_API
#endif


// ***************************************************************
//                   SHARED LIBRARY CALLS                        *
// ***************************************************************


extern "C" {

  LLSPY_API int Deskew_interface(const float * const raw_data,
                       int nx, int ny, int nz,
                       float dz, float dr, float deskewAngle,
                       float * const result,
                       int outputWidth, int extraShift, float padVal = 0);
    
  LLSPY_API int Affine_interface(const float * const raw_data,
                       int nx, int ny, int nz,
                       float * const result,
                       const float * affMat);

  LLSPY_API int Affine_interface_RA(const float * const raw_data,
                       int nx, int ny, int nz,
                       float dx, float dy, float dz,
                       float * const result,
                       const float * affMat);

  // LLSPY_API int camcor_interface_init(int nx, int ny, int nz,
  //                      const float * const camparam);

  // LLSPY_API int camcor_interface(const unsigned short * const raw_data,
  //                      int nx, int ny, int nz,
  //                      unsigned short * const result);

  LLSPY_API void cuda_reset();

}



unsigned output_nz;
double deskewFactor;
unsigned deskewedXdim;



void cuda_reset()
{
  cudaDeviceReset();
}

// int camcor_interface_init(int nx, int ny, int nz,
//                      const float * const camparam)
// {
//   CImg<> h_camparam(camparam, nx, ny, 3);
//   setupConst(nx, ny, nz);
//   setupCamCor(nx, ny, h_camparam.data());
//   return 1;
// }


// int camcor_interface(const unsigned short * const raw_data,
//                      int nx, int ny, int nz,
//                      unsigned short * const result)
// {
//   CImg<unsigned short> input(raw_data, nx, ny, nz);
//   CImg<unsigned> raw_image(input);
//   GPUBuffer d_correctedResult(nx * ny * nz * sizeof(unsigned short), 0, false);
//   setupData(nx, ny, nz, raw_image.data());
//   camcor_GPU(nx, ny, nz, d_correctedResult);
//   //transfer result back to host
//   cudaMemcpy(result, d_correctedResult.getPtr(), nx * ny * nz * sizeof(unsigned short), cudaMemcpyDeviceToHost);
//   return 1;
// }



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

  GPUBuffer d_affMatrix(16 * sizeof(float), 0);
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

  GPUBuffer d_affMatrix(16 * sizeof(float), 0);
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