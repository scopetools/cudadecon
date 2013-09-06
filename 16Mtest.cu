#include <helper_cuda.h> //checkCudaErrors()


// Simple transformation kernel
__global__ void scaleKernel(int width, float* output1, int *output2)
{
// Calculate normalized texture coordinates
  unsigned ind = blockIdx.x;
  
  output1[ind] = 16777206 + ind;
  output2[ind] = 16777206 + ind;
}



int main(int argc, char *argv[])
{

  // Allocate result of scaling in device memory
  int nx=20;
  float* d_output;
  checkCudaErrors(cudaMalloc(&d_output, nx*sizeof(float)));
  int* d_outputI;
  checkCudaErrors(cudaMalloc(&d_outputI, nx*sizeof(int)));

  scaleKernel<<<nx, 1>>>(nx, d_output, d_outputI);

  // Copy outputs back into host memory
  float *output = (float *)malloc(nx*sizeof(float));

  checkCudaErrors(cudaMemcpy(output, d_output,
                             nx * sizeof(float),
                             cudaMemcpyDeviceToHost));
  cudaFree(d_output);

  int *outputI = (int *)malloc(nx*sizeof(int));

  checkCudaErrors(cudaMemcpy(outputI, d_outputI,
                             nx * sizeof(int),
                             cudaMemcpyDeviceToHost));
  cudaFree(d_outputI);

  for (int i=0; i<nx; i++) {
    printf("%.2f   ", output[i]);
    printf("%d\n", outputI[i]);
  }
  return 0;
}
