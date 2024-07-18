#include "PinnedCPUBuffer.h"
#include "CPUBuffer.h"
#include "GPUBuffer.h"
#include "../cutilSafeCall.h"

PinnedCPUBuffer::PinnedCPUBuffer() :
  size_(0), ptr_(0)
{
}

PinnedCPUBuffer::PinnedCPUBuffer(size_t size) :
  size_(size), ptr_(0)
{
  cudaError_t err = cudaHostAlloc((void**)&ptr_, size_, cudaHostAllocDefault);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaHostAlloc() failed.");
  }
}

PinnedCPUBuffer::PinnedCPUBuffer(const Buffer& toCopy) :
  size_(toCopy.getSize()), ptr_(0)
{
  cudaError_t err = cudaHostAlloc((void**)&ptr_, size_, cudaHostAllocDefault);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaHostAlloc() failed.");
  }
  toCopy.set(this, 0, size_, 0);
}

PinnedCPUBuffer& PinnedCPUBuffer::operator=(const Buffer& rhs) {
  cudaError_t err;
  if (this != &rhs) {
    size_ = rhs. getSize();
    if (ptr_ != 0) {
      err = cudaFreeHost(ptr_);
      if (err != cudaSuccess) {
        throw std::runtime_error("cudaFreeHost() failed."); 
      }
    }
    err = cudaHostAlloc((void**)&ptr_, size_, cudaHostAllocDefault);
    if (err != cudaSuccess) {
      throw std::runtime_error("cudaHostAlloc failed.");
    }
    rhs.set(this, 0, size_, 0);
  }
  return *this;
}

PinnedCPUBuffer::~PinnedCPUBuffer() noexcept(false) {
  if (ptr_) {
    cudaError_t err = cudaFreeHost(ptr_);
    if (err != cudaSuccess) {
      throw std::runtime_error("cudaFreeHost() failed.");
    }
  }
}

void PinnedCPUBuffer::resize(size_t newsize) {
  cudaError_t err;
  if (ptr_) {
    err = cudaFreeHost(ptr_);
    if (err != cudaSuccess) {
      throw std::runtime_error("cudaFreeHost() failed.");
    }
    ptr_ = 0;
  }
  size_ = newsize;
  if (newsize > 0) {
    err = cudaHostAlloc((void**)&ptr_, size_, cudaHostAllocDefault);
    if (err != cudaSuccess) {
      throw std::runtime_error("cudaHostAlloc() failed.");
    }
  }
}

void PinnedCPUBuffer::set(Buffer* dest, size_t srcBegin, size_t srcEnd,
    size_t destBegin) const {
  dest->setFrom(*this, srcBegin, srcEnd, destBegin);
}

void PinnedCPUBuffer::setFrom(const CPUBuffer& src, size_t srcBegin,
    size_t srcEnd, size_t destBegin)  {
  if (srcEnd - srcBegin > size_ - destBegin) {
    throw std::runtime_error("Buffer overflow.");
  }
  memcpy(ptr_ + destBegin, (char*)src.getPtr() + srcBegin,
      srcEnd - srcBegin);
}

void PinnedCPUBuffer::setFrom(const PinnedCPUBuffer& src, size_t srcBegin,
    size_t srcEnd, size_t destBegin)  {
  setFrom((const CPUBuffer&)src, srcBegin, srcEnd, destBegin);
}


void PinnedCPUBuffer::setFrom(const GPUBuffer& src, size_t srcBegin,
    size_t srcEnd, size_t destBegin)  {
  if (srcEnd - srcBegin > size_ - destBegin) {
    throw std::runtime_error("Buffer overflow.");
  }
  cudaError_t err = cudaMemcpyAsync(ptr_ + destBegin,
      (char*)src.getPtr() + srcBegin, srcEnd - srcBegin,
      cudaMemcpyDeviceToHost, 0);
  if (err != cudaSuccess) {
    std::cout << "Pinned CPUBuffer cudaMemcpy failed. Error code: " << err << std::endl;
    std::cout << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error("cudaMemcpy failed.");
  }
}

void PinnedCPUBuffer::setFrom(const void* src, size_t srcBegin,
    size_t srcEnd, size_t destBegin)
{
  // CPUBuffer::setFrom(src, srcBegin, srcEnd, destBegin);
  if (srcEnd - srcBegin > size_ - destBegin) {
    throw std::runtime_error("Buffer overflow.");
  }
  memcpy(ptr_ + destBegin, (char*)src + srcBegin, srcEnd - srcBegin);
}

bool PinnedCPUBuffer::hasNaNs(bool verbose) const 
{
  size_t numEntries = size_ / sizeof(float);
  float* arr = (float*)ptr_;
  size_t i = 0;
  bool haveNaNs = false;
  if (verbose) {
    for (i = 0; i < numEntries; ++i) {
#ifndef _WIN32
      bool in = std::isnan(arr[i]);
#else
      bool in = _isnan(arr[i]);
#endif
      if (in) {
        std::cout << "NaN entry in array at: " << i << std::endl;
      }
      haveNaNs |= in;
    }
  } else {
    while ((!haveNaNs) && i < numEntries) {
#ifndef _WIN32
      haveNaNs = haveNaNs || std::isnan(arr[i]);
#else
      haveNaNs = haveNaNs || _isnan(arr[i]);
#endif
      ++i;
    }
  }
  return haveNaNs;
}
