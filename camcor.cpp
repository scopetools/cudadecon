#include "linearDecon.h"


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
