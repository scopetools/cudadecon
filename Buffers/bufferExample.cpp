#include <Buffer.h>
#include <GPUBuffer.h>
#include <CPUBuffer.h>
#include <PinnedCPUBuffer.h>

#include <iostream>

int main(int argn, char** argv)
{
  // Size of vector
  static const int N = 10;
  //
  // Input data
  float v[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};

  // Create a Buffer on the host side
  CPUBuffer v_cpu(sizeof(v));

  // Set the buffer from the input data
  v_cpu.setFrom(v, 0, sizeof(v), 0);

  // Print contents of v_cpu
  std::cout << "Data before transfer to GPU:\n";
  v_cpu.dump(std::cout, N);

  // Create Buffer on the GPU (cuda device ID 0)
  GPUBuffer v_gpu(sizeof(v), 0);

  // Copy data from CPU to GPU
  v_cpu.set(&v_gpu, 0, v_cpu.getSize(), 0);

  // Create another Buffer on the CPU
  CPUBuffer v2_cpu(sizeof(v));

  // Transfer data back from GPU to the CPU
  v_gpu.set(&v2_cpu, 0, v_gpu.getSize(), 0);

  // Print contents of output buffer
  std::cout << "Data after transfer to GPU:\n";
  v2_cpu.dump(std::cout, N);

  // Upon exiting the scope of the program the memory associated with
  // the various Buffers (CPU and GPU) is automatically deallocated.
  return 0;
}

/** \example bufferExample.cpp
 * @brief This is an example of how to use the Buffer classes for memory
 * management and data transfers.
 */

