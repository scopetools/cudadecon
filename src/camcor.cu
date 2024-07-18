#include <GPUBuffer.h>

#include "camcor_context.h"
#include "cutilSafeCall.h"

__host__ void setupCamCor(CamcorContext* context, float* h_camparam) {
  if (!context) return;  // check for null pointer

  int nx = context->nx;
  int ny = context->ny;

  // Allocate CUDA array in device memory
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaArray* d_camparamArray;
  cudaExtent extent = make_cudaExtent(nx, ny, 3);
  cudaMalloc3DArray(&d_camparamArray, &channelDesc, extent);

  // Copy host camparams to device memory
  cudaMemcpy3DParms parms = {0};
  parms.srcPtr = make_cudaPitchedPtr(h_camparam, nx * sizeof(float), nx, ny);
  parms.dstArray = d_camparamArray;
  parms.extent = extent;
  parms.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&parms);

  // Create texture object
  cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = d_camparamArray;

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeBorder;
  texDesc.addressMode[1] = cudaAddressModeBorder;
  texDesc.addressMode[2] = cudaAddressModeBorder;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  cudaCreateTextureObject(&context->camparam_texObj, &resDesc, &texDesc, NULL);
}

__host__ void setupData(CamcorContext* context, unsigned* h_data) {
  int nx = context->nx;
  int ny = context->ny;
  int nz = context->nz;

  // Allocate CUDA array in device memory
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned>();
  cudaArray* d_dataArray;
  cudaExtent extent = make_cudaExtent(nx, ny, nz);
  cudaMalloc3DArray(&d_dataArray, &channelDesc, extent);

  // Copy host data to device memory
  cudaMemcpy3DParms parms = {0};
  parms.srcPtr = make_cudaPitchedPtr(h_data, nx * sizeof(unsigned), nx, ny);
  parms.dstArray = d_dataArray;
  parms.extent = extent;
  parms.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&parms);

  // Define resource descriptor
  cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = d_dataArray;

  // Define texture descriptor
  cudaTextureDesc texDesc = {};
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  texDesc.addressMode[2] = cudaAddressModeWrap;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  // Create texture object
  cudaCreateTextureObject(&context->data_texObj, &resDesc, &texDesc, nullptr);
}

__global__ void camcor_kernel(unsigned nx, unsigned ny, unsigned nz,
                              cudaTextureObject_t camparam_texObj, cudaTextureObject_t data_texObj,
                              unsigned short* output) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= nx || y >= ny || z >= nz) {
    return;
  }

  // normalize coordinates
  float u = (x + 0.5) / (float)nx;
  float v = (y + 0.5) / (float)ny;
  float w = (z + 0.5) / (float)nz;
  float wp = (z - 0.5) / (float)nz;

  float a = tex3D<float>(camparam_texObj, x, y, 0);
  float b = tex3D<float>(camparam_texObj, x, y, 1);
  float offset = tex3D<float>(camparam_texObj, x, y, 2);

  unsigned voxel = tex3D<unsigned int>(data_texObj, u, v, w);
  unsigned previousvoxel = tex3D<unsigned int>(data_texObj, u, v, wp);

  unsigned int i = z * ny * nx + y * nx + x;
  output[i] =
      (unsigned short)voxel - offset - 0.9f * a * (1 - expf(-b * (previousvoxel - offset)));
  output[i] = output[i] > 0 ? output[i] : 0;
}

__host__ void camcor_GPU(CamcorContext* context, GPUBuffer& outBuf) {
  unsigned nx = context->nx;
  unsigned ny = context->ny;
  unsigned nz = context->nz;

  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x, (ny + dimBlock.y - 1) / dimBlock.y,
               (nz + dimBlock.z - 1) / dimBlock.z);

  camcor_kernel<<<dimGrid, dimBlock>>>(nx, ny, nz, context->camparam_texObj, context->data_texObj,
                                       (unsigned short*)outBuf.getPtr());
  // CudaCheckError();
}
