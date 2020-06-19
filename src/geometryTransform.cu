#include <GPUBuffer.h>


__global__ void deskew_kernel(float *in, int nx, int ny, int nz,
                              float *out, int nxOut, int extraShift,
                              double deskewFactor, float padVal)
{
  unsigned xout = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned yout = blockIdx.y;
  unsigned zout = blockIdx.z;

  if (xout < nxOut) {
    float xin = (xout - nxOut/2.+extraShift) - deskewFactor*(zout-nz/2.) + nx/2.;

    unsigned indout = zout * nxOut * ny + yout * nxOut + xout;
    if (xin >= 0 && xin < nx-1) {

      // 09-03-2013 Very important lesson learned:
      // the (unsigned int) casting has be placed right there because
      // otherwise, the entire express would evaluate as floating point and
      // there're only 24-bit mantissa, so any odd index that's > 16777216 would
      // inaccurately rounded up. int or unsigned does not have the 24-bit limit.
      unsigned indin = zout * nx * ny + yout * nx + (unsigned int) floor(xin);

      float offset = xin - floor(xin);
      out[indout] = (1-offset) * in[indin] + offset * in[indin+1]; // linear interpolation done within each slice (i.e. along x)
    }
    else
      out[indout] = padVal;
  }
}

__host__ void deskew_GPU(GPUBuffer &inBuf, int nx, int ny, int nz,
                         double deskewFactor, GPUBuffer &outBuf,
                         int newNx, int extraShift, float padVal)
{
  dim3 block(128, 1, 1);
  unsigned nxBlocks = (unsigned ) ceil(newNx / (float) block.x);
  dim3 grid(nxBlocks, ny, nz);

  deskew_kernel<<<grid, block>>>((float *) inBuf.getPtr(),
                                 nx, ny, nz, 
                                 (float *) outBuf.getPtr(), newNx,
                                 extraShift, deskewFactor, padVal);
#ifndef NDEBUG
  std::cout<< "deskew_GPU(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;
#endif
}

__global__ void rotate_kernel(float *in, int nx_in, int ny, int nz_in,
                              float *out, int nx_out, int nz_out,
                              float *rotMat)
{
  unsigned xout = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned yout = blockIdx.y;
  unsigned zout = blockIdx.z;

  if (xout < nx_out) {
    float xout_centered, zout_centered;
    xout_centered = xout - nx_out/2.;
    zout_centered = zout - nz_out/2.;

    unsigned nxy_in = nx_in * ny;
    unsigned nxy_out = nx_out * ny;
    unsigned yind_out = yout * nx_out;
    unsigned yind_in = yout * nx_in;

    float zin = rotMat[0] * zout_centered + rotMat[1] * xout_centered + nz_in/2.;
    float xin = rotMat[2] * zout_centered + rotMat[3] * xout_centered + nx_in/2.;

    unsigned indout = (nz_out-1-zout) * nxy_out + yind_out + xout; // flip z indices

    if (xin >= 0 && xin < nx_in-1 && zin >=0 && zin < nz_in-1) {

      unsigned indin00 = (unsigned) floor(zin) * nxy_in + yind_in + (unsigned) floor(xin);
      unsigned indin01 = indin00 + 1;
      unsigned indin10 = indin00 + nxy_in;
      unsigned indin11 = indin10 + 1;

      float xoffset = xin - floor(xin);
      float zoffset = zin - floor(zin);
      out[indout] = (1-zoffset) * ( (1-xoffset) * in[indin00] + xoffset * in[indin01]) + 
        zoffset * ((1-xoffset) * in[indin10] + xoffset * in[indin11]);
    }
    else
      out[indout] = 0.f;
  }
}

__host__ void rotate_GPU(GPUBuffer &inBuf, int nx, int ny, int nz,
                         GPUBuffer &rotMatrix, GPUBuffer &outBuf,
                         int nx_out, int nz_out)
{
  dim3 block(128, 1, 1);
  unsigned nxBlocks = (unsigned ) ceil(nx_out / (float) block.x);
  dim3 grid(nxBlocks, ny, nz_out);

  rotate_kernel<<<grid, block>>>((float *) inBuf.getPtr(),
                                 nx, ny, nz,
                                 (float *) outBuf.getPtr(),
                                 nx_out, nz_out,
                                 (float *) rotMatrix.getPtr());
#ifndef NDEBUG
  std::cout<< "rotate_GPU(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;
#endif
}

__global__ void crop_kernel(float *in, int nx, int ny, int nz,
                            int new_nx, int new_ny, int new_nz,
                            float *out)
{
  unsigned xout = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned yout = blockIdx.y;
  unsigned zout = blockIdx.z;

  if (xout < new_nx) { 
    // Assumption: new dimensions are <= old ones
    unsigned xin = xout + nx - new_nx;
    unsigned yin = yout + ny - new_ny;
    unsigned zin = zout + nz - new_nz;
    unsigned indout = zout * new_nx * new_ny + yout * new_nx + xout;
    unsigned indin = zin * nx * ny + yin * nx + xin;
    out[indout] = in[indin];
  }
}


__host__ void cropGPU(GPUBuffer &inBuf, int nx, int ny, int nz,
                      int new_nx, int new_ny, int new_nz,
                      GPUBuffer &outBuf)
{

  dim3 block(128, 1, 1);
  unsigned nxBlocks = (unsigned ) ceil(new_nx / (float) block.x);
  dim3 grid(nxBlocks, new_ny, new_nz);

  crop_kernel<<<grid, block>>>((float *) inBuf.getPtr(),
                               nx, ny, nz,
                               new_nx, new_ny, new_nz,
                               (float *) outBuf.getPtr());

#ifndef NDEBUG
  std::cout<< "cropGPU(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;
#endif
}

// ******************************************************************//
// Duplicate the first Z half of the "in" stack, in reverse-Z order,
// into the 2nd Z half of it; essentially faking continuous structure
// in Z to reduce Z ringing from FFT
// ******************************************************************//
__global__ void dupRevStack_kernel(float *in, unsigned nx, unsigned nxy, unsigned nz)
{
  unsigned xin = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned yin = blockIdx.y;
  unsigned zin = blockIdx.z;

  if (xin < nx) {
    unsigned zout = (nz<<1) - zin - 1; // + and - take precedence over << and >>!!
    unsigned indout = zout * nxy + yin * nx + xin;
    unsigned indin  =  zin * nxy + yin * nx + xin;
    in[indout] = in[indin];
  }
}

__host__ void duplicateReversedStack_GPU(GPUBuffer &zExpanded, int nx, int ny, int nz)
{
  dim3 block(128, 1, 1);
  unsigned nxBlocks = (unsigned ) ceil(nx / (float) block.x);
  dim3 grid(nxBlocks, ny, nz);

  dupRevStack_kernel<<<grid, block>>>((float *) zExpanded.getPtr(),
                                      nx, nx*ny, nz);
#ifndef NDEBUG
  std::cout<< "duplicateReversedStack_GPU(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;
#endif
}


texture<float, cudaTextureType3D, cudaReadModeElementType> texRef;


// Simple transformation kernel
__global__ void transformKernel(float *output,
                                int nx, int ny, int nz,
                                float *mat)
{

  // Calculate texture coordinates
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= nx || y >= ny || z >= nz) {
    return;
  }

  // for normalized coordinates
  //float u = x / (float)nx;
  //float v = y / (float)ny;
  //float w = z / (float)nz;

  float u = x;
  float v = y;
  float w = z;

  float tu = mat[0]*u + mat[1]*v + mat[2] *w +  mat[3] + 0.5f;
  float tv = mat[4]*u + mat[5]*v + mat[6] *w +  mat[7] + 0.5f;
  float tw = mat[8]*u + mat[9]*v + mat[10]*w + mat[11] + 0.5f;

  // Read from texture and write to global memory
  int idx = z * (nx*ny) + y * nx + x;
  output[idx] = tex3D(texRef, tu, tv, tw);
}

// Simple transformation kernel
__global__ void transformKernelRA(float *output,
                                  int nx, int ny, int nz,
                                  float dx, float dy, float dz,
                                  float *mat)
{

  // Calculate texture coordinates
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= nx || y >= ny || z >= nz) {
    return;
  }

  float u = x;
  float v = y;
  float w = z;

  // intrinsic coords to world
  u = 0.5 + (u - 0.5) * dx;
  v = 0.5 + (v - 0.5) * dy;
  w = 0.5 + (w - 0.5) * dz;

  // transform coordinates in world coordinate frame
  float tu = mat[0]*u + mat[1]*v + mat[2] *w +  mat[3];
  float tv = mat[4]*u + mat[5]*v + mat[6] *w +  mat[7];
  float tw = mat[8]*u + mat[9]*v + mat[10]*w + mat[11];

  // world coords to intrinsic
  tu = 0.5 + (tu - 0.5) / dx;
  tv = 0.5 + (tv - 0.5) / dy;
  tw = 0.5 + (tw - 0.5) / dz;

  // Read from texture and write to global memory
  int idx = z * (nx*ny) + y * nx + x;
  output[idx] = tex3D(texRef, tu, tv, tw);
}


// host data
__host__ void affine_GPU(cudaArray *cuArray, int nx, int ny, int nz,
                         float * result, GPUBuffer &affMat)
{

  // Allocate CUDA array in device memory
  cudaChannelFormatDesc channelDesc =
    cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

  // Set texture reference parameters
  texRef.addressMode[0] = cudaAddressModeBorder;
  texRef.addressMode[1] = cudaAddressModeBorder;
  texRef.addressMode[2] = cudaAddressModeBorder;
  texRef.filterMode = cudaFilterModeLinear;
  texRef.normalized = false;

  // Bind the array to the texture reference
  cudaBindTextureToArray(texRef, cuArray, channelDesc);

  // Allocate result of transformation in device memory
  float* output;
  cudaMalloc(&output, nx * ny * nz * sizeof(float));

  // Invoke kernel dim3
  dim3 dimBlock(16,16,4);
  dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x,
               (ny + dimBlock.y - 1) / dimBlock.y,
               (nz + dimBlock.z - 1) / dimBlock.z);

  transformKernel<<<dimGrid, dimBlock>>>(output, nx, ny, nz, (float *) affMat.getPtr());
  CudaCheckError();

  //transfer result back to host
  cudaMemcpy(result, output, nz * nx * ny * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFreeArray(cuArray);
  cudaFree(output);
}

// host data
__host__ void affine_GPU_RA(cudaArray *cuArray, int nx, int ny, int nz,
                         float dx, float dy, float dz,
                         float * result, GPUBuffer &affMat)
{

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    // Set texture reference parameters
    texRef.addressMode[0] = cudaAddressModeBorder;
    texRef.addressMode[1] = cudaAddressModeBorder;
    texRef.addressMode[2] = cudaAddressModeBorder;
    texRef.filterMode = cudaFilterModeLinear;
    texRef.normalized = false;

    // Bind the array to the texture reference
    cudaBindTextureToArray(texRef, cuArray, channelDesc);

    // Allocate result of transformation in device memory
    float* output;
    cudaMalloc(&output, nx * ny * nz * sizeof(float));

    // Invoke kernel dim3
    dim3 dimBlock(16,16,4);
    dim3 dimGrid((nx + dimBlock.x - 1) / dimBlock.x,
                 (ny + dimBlock.y - 1) / dimBlock.y,
                 (nz + dimBlock.z - 1) / dimBlock.z);

    transformKernelRA<<<dimGrid, dimBlock>>>(output, nx, ny, nz, dx, dy, dz, (float *) affMat.getPtr());
    CudaCheckError();

    //transfer result back to host
    cudaMemcpy(result, output, nz * nx * ny * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFreeArray(cuArray);
    cudaFree(output);
}
