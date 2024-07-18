#ifndef CAMCOR_CONTEXT_H
#define CAMCOR_CONTEXT_H

#include <cuda_runtime.h>

struct CamcorContext {
  unsigned nx, ny, nz, nxy, nxyz;
  cudaTextureObject_t camparam_texObj = 0;  // Texture object for camera parameters
  cudaTextureObject_t data_texObj = 0;      // Texture object for data, if needed

  // Constructor
  CamcorContext(int nx, int ny, int nz)
      : nx(nx), ny(ny), nz(nz), nxy(nx * ny), nxyz(nx * ny * nz) {}
};

#endif  // CAMCOR_CONTEXT_H