#include <GPUBuffer.h>
#include <helper_cuda.h> //checkCudaErrors()


__global__ void deskew_kernel(float *in, int nx, int ny, int nz,
                              float *out, int nxOut,
                              float *Tvector)
{
  unsigned xout = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned yout = blockIdx.y;
  unsigned zout = blockIdx.z;

  if (xout < nxOut) {
    float xin = (xout - nxOut/2.) - Tvector[0]*(zout-nz/2.) + nx/2.; // - Tvector[1];

    unsigned indout = zout * nxOut * ny + yout * nxOut + xout;
    if (xin >= 0 && xin < nx-1) {

      // 09-03-2013 Very important lesson learned:
      // the (unsigned int) casting has be placed right there because
      // otherwise, the entire express would evaluate as floating point and
      // there're only 24-bit mantissa, so any odd index that's > 16777216 would
      // inaccurately rounded up. int or unsigned does not have the 24-bit limit.
      unsigned indin = zout * nx * ny + yout * nx + (unsigned int) floor(xin);

      float offset = xin - floor(xin);
      out[indout] = (1-offset) * in[indin] + offset * in[indin+1];
    }
    else
      out[indout] = 0.f;
  }
}

__host__ void deskew_GPU(GPUBuffer &inBuf, int nx, int ny, int nz,
                         GPUBuffer &deskewMatrix, GPUBuffer &outBuf, int newNx)
{
  dim3 block(128, 1, 1);
  unsigned nxBlocks = (unsigned ) ceil(newNx / (float) block.x);
  dim3 grid(nxBlocks, ny, nz);

  deskew_kernel<<<grid, block>>>((float *) inBuf.getPtr(),
                                 nx, ny, nz, 
                                 (float *) outBuf.getPtr(), newNx,
                                 (float *) deskewMatrix.getPtr());
  std::cout<< cudaGetErrorString(cudaGetLastError()) << std::endl;
}

__global__ void rotate_kernel(float *in, int nx, int ny, int nz,
                              float *out, float *rotMat)
{
  unsigned xout = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned yout = blockIdx.y;
  unsigned zout = blockIdx.z;

  if (xout < nx) {
    float xout_centered, zout_centered;
    xout_centered = xout - nx/2.;
    zout_centered = zout - nz/2.;

    unsigned nxy = nx * ny;
    unsigned yind = yout * nx;

    float zin = rotMat[0] * zout_centered + rotMat[1] * xout_centered + nz/2.;
    float xin = rotMat[2] * zout_centered + rotMat[3] * xout_centered + nx/2.;

    unsigned indout = (nz-1-zout) * nxy + yind + xout; // flip z indices

    if (xin >= 0 && xin < nx-1 && zin >=0 && zin < nz-1) {

      unsigned indin00 = (unsigned) floor(zin) * nxy + yind + (unsigned) floor(xin);
      unsigned indin01 = indin00 + 1;
      unsigned indin10 = indin00 + nxy;
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
                         GPUBuffer &rotMatrix, GPUBuffer &outBuf)
{
  dim3 block(128, 1, 1);
  unsigned nxBlocks = (unsigned ) ceil(nx / (float) block.x);
  dim3 grid(nxBlocks, ny, nz);

  rotate_kernel<<<grid, block>>>((float *) inBuf.getPtr(),
                                 nx, ny, nz,
                                 (float *) outBuf.getPtr(),
                                 (float *) rotMatrix.getPtr());
  std::cout<< cudaGetErrorString(cudaGetLastError()) << std::endl;
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

  std::cout<< "cropGPU(): " << cudaGetErrorString(cudaGetLastError()) << std::endl;
}
