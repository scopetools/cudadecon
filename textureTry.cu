#include <helper_cuda.h> //checkCudaErrors()

#define cimg_use_tiff
#include <CImg.h>
using namespace cimg_library;

texture<float, cudaTextureType2D, cudaReadModeElementType> texRef1, texRef2;


// Simple transformation kernel
__global__ void scaleKernel(int width, int height, int scalex, 
                            int scaley, float* output1, float* output2)
{
// Calculate normalized texture coordinates
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x<width*scalex && y<height*scaley) {
    float u = ((float) x) / scalex / width;
    float v = ((float) y) / scaley / height;
    // Read from texture and write to global memory
    unsigned ind = y * width * scalex + x;
    output1[ind] = tex2D(texRef1, u, v);
    output2[ind] = tex2D(texRef2, u, v);
  }
}



int main(int argc, char *argv[])
{
  // load input file
  CImg<> indata(argv[1]);
  int nx = indata.width()/2;  // 2 because OTF is of float2 type
  int ny = indata.height();
  // split real and imag parts into two CImg arrays:
  CImg<> realpart(nx, ny), imagpart(nx, ny);

#pragma omp parallel for  
  cimg_forXY(realpart, x, y) {
    realpart(x, y) = indata(2*x  , y);
    imagpart(x, y) = indata(2*x+1, y);
  }
  

  // Allocate CUDA array in device memory
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

  cudaArray* cuArray1, *cuArray2;
  cudaMallocArray(&cuArray1, &channelDesc, nx, ny);
  cudaMallocArray(&cuArray2, &channelDesc, nx, ny);

  // Copy to device memory
  checkCudaErrors(cudaMemcpyToArray(cuArray1, 0, 0, realpart.data(),
                                    realpart.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToArray(cuArray2, 0, 0, imagpart.data(),
                                    imagpart.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));

  // cudaMemcpy(realpart.data(), cuArray1,
  //            nx * ny * sizeof(float),
  //            cudaMemcpyDeviceToHost);
  // cudaMemcpy(imagpart.data(), cuArray2,
  //            nx * ny * sizeof(float),
  //            cudaMemcpyDeviceToHost);
  // realpart.display();
  // imagpart.display();

  // Set texture reference parameters
  texRef1.addressMode[0] = cudaAddressModeClamp;
  texRef1.addressMode[1] = cudaAddressModeClamp;
  texRef1.filterMode = cudaFilterModeLinear;
  texRef1.normalized = true;  // wonder what "false" would do here
  texRef2.addressMode[0] = cudaAddressModeClamp;
  texRef2.addressMode[1] = cudaAddressModeClamp;
  texRef2.filterMode = cudaFilterModeLinear;
  texRef2.normalized = true;
  // Bind the array to the texture reference
  checkCudaErrors(cudaBindTextureToArray(texRef1, cuArray1, channelDesc));
  checkCudaErrors(cudaBindTextureToArray(texRef2, cuArray2, channelDesc));

  // Allocate result of scaling in device memory
  float* output1, *output2;
  int scalex=2, scaley=2;
  checkCudaErrors(cudaMalloc(&output1, nx * scalex * ny * scaley * sizeof(float)));
  checkCudaErrors(cudaMalloc(&output2, nx * scalex * ny * scaley * sizeof(float)));

  // Invoke kernel
  dim3 dimBlock(16, 16);
  dim3 dimGrid( (int) (ceil( ((float) nx*scalex) / dimBlock.x)),
                (int) (ceil( ((float) ny*scaley) / dimBlock.y)) );
  scaleKernel<<<dimGrid, dimBlock>>>(nx, ny, scalex, scaley, output1, output2);

  // Copy outputs back into host memory
  CImg<> realpartScaled(nx*scalex, ny*scaley), imagpartScaled(nx*scalex, ny*scaley);

  checkCudaErrors(cudaMemcpy(realpartScaled.data(), output1,
                             nx * scalex * ny * scaley * sizeof(float),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(imagpartScaled.data(), output2,
                             nx * scalex * ny * scaley * sizeof(float),
                             cudaMemcpyDeviceToHost));
 
  realpartScaled.display();
  imagpartScaled.display();

  CImg<> combined(nx*scalex*2, ny*scaley);
#pragma omp parallel for  
  cimg_forXY(realpartScaled, x, y) {
    combined(2*x  , y) = realpartScaled(x, y);
    combined(2*x+1, y) = imagpartScaled(x, y);
  }
  
  combined.save("scaledOTF.tif");

  // Free device memory
  cudaFreeArray(cuArray1);
  cudaFreeArray(cuArray2);

  cudaFree(output1);
  cudaFree(output2);
  return 0;
}
