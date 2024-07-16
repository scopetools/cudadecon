#include <GPUBuffer.h>
#include "cutilSafeCall.h"


__constant__ unsigned const_nx;
__constant__ unsigned const_ny;
__constant__ unsigned const_nz;
__constant__ unsigned const_nxy;
__constant__ unsigned const_nxyz;


texture<float, cudaTextureType3D, cudaReadModeElementType> camparam_tex;
texture<unsigned, cudaTextureType3D, cudaReadModeElementType> data_tex;

__host__ void setupConst(int nx, int ny, int nz)
{
  cutilSafeCall(cudaMemcpyToSymbol(const_nx, &nx, sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(const_ny, &ny, sizeof(int)));
  cutilSafeCall(cudaMemcpyToSymbol(const_nz, &nz, sizeof(int)));
  unsigned int nxy = nx*ny;
  cutilSafeCall(cudaMemcpyToSymbol(const_nxy, &nxy, sizeof(unsigned int)));
  unsigned int nxyz = nx*ny*nz;
  cutilSafeCall(cudaMemcpyToSymbol(const_nxyz, &nxyz, sizeof(unsigned int)));
}

__host__ void setupCamCor(int nx, int ny, float * h_caparam)
{

  // Allocate CUDA array in device memory
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(
      32, 0, 0, 0, cudaChannelFormatKindFloat);

  cudaArray* d_camparamArray;
  cudaExtent extent = make_cudaExtent(nx, ny, 3);
  cudaMalloc3DArray(
      &d_camparamArray,
      &channelDesc,
      extent,
      cudaArrayDefault
  );

  // Copy host camparams to device memory
  cudaMemcpy3DParms parms = {0};
  parms.srcPtr = make_cudaPitchedPtr(
      h_caparam,
      nx * sizeof(float), nx, ny
  );
  parms.dstArray = d_camparamArray;
  parms.extent = extent;
  parms.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&parms);

  // Set texture reference parameters
  camparam_tex.addressMode[0] = cudaAddressModeBorder;
  camparam_tex.addressMode[1] = cudaAddressModeBorder;
  camparam_tex.addressMode[2] = cudaAddressModeBorder;
  camparam_tex.filterMode = cudaFilterModePoint;
  camparam_tex.normalized = false;

  // Bind the array to the texture reference
  cudaBindTextureToArray(camparam_tex, d_camparamArray, channelDesc);
}

__host__ void setupData(int nx, int ny, int nz, unsigned * h_data)
{

  cudaArray* d_dataArray;

  // Allocate CUDA array in device memory
  //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(
  //    32, 0, 0, 0, cudaChannelFormatKindUnsigned);

  int e = (int)sizeof(unsigned) * 8;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(e, 0, 0, 0, cudaChannelFormatKindUnsigned);

  cudaExtent extent = make_cudaExtent(nx, ny, nz);
  cudaMalloc3DArray(
      &d_dataArray,
      &channelDesc,
      extent,
      cudaArrayDefault
  );

  // Copy host camparams to device memory
  cudaMemcpy3DParms parms = {0};
  parms.srcPtr = make_cudaPitchedPtr(
      h_data,
      nx * sizeof(unsigned), nx, ny
  );
  parms.dstArray = d_dataArray;
  parms.extent = extent;
  parms.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&parms);

  // Set texture reference parameters
  data_tex.addressMode[0] = cudaAddressModeWrap;
  data_tex.addressMode[1] = cudaAddressModeWrap;
  data_tex.addressMode[2] = cudaAddressModeWrap;
  data_tex.filterMode = cudaFilterModePoint;
  data_tex.normalized = true;

  // Bind the array to the texture reference
  cudaBindTextureToArray(data_tex, d_dataArray, channelDesc);
}


__global__ void camcor_kernel(unsigned short *output)
{

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= const_nx || y >= const_ny || z >= const_nz) {
        return;
    }

    //normalize coordinates
    float u = (x + 0.5) / (float) const_nx;
    float v = (y + 0.5) / (float) const_ny;
    float w = (z + 0.5) / (float) const_nz;
    float wp = (z - 0.5) / (float) const_nz;

    float a = tex3D(camparam_tex, x, y, 0);
    float b = tex3D(camparam_tex, x, y, 1);
    float offset = tex3D(camparam_tex, x, y, 2);

    unsigned voxel = tex3D(data_tex, u, v, w);
    unsigned previousvoxel = tex3D(data_tex, u, v, wp);

    //if (x == 1 and y == 1 and z == 1) {
    //     printf("x: %d, y: %d, z: %d \n ", x, y, z);
    //     printf("u: %f, v: %f, w: %f, wp: %f \n ", u, v, w, wp);
    //     printf("a: %f, b: %f, o: %f \n ", a, b, offset);
    //    printf("voxel: %d, \n ", voxel);
    //     printf("previousvoxel: %d \n ", previousvoxel);
    // }

    unsigned int i = z * const_ny * const_nx + y * const_nx + x;
    output[i] = (unsigned short) voxel - offset - 0.9f * a * (1 - expf(-b * (previousvoxel - offset)));
    output[i] = output[i] > 0 ? output[i] : 0;

}


__host__ void camcor_GPU(int nx, int ny, int nz, GPUBuffer &outBuf){

    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x,
                 (ny + dimBlock.y - 1) / dimBlock.y,
                 (nz + dimBlock.z - 1) / dimBlock.z);

    camcor_kernel<<<dimGrid, dimBlock>>>((unsigned short *) outBuf.getPtr());
    //CudaCheckError();
}
