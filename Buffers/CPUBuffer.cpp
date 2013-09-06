#include "CPUBuffer.h"

#include "PinnedCPUBuffer.h"
#include "GPUBuffer.h"

CPUBuffer::CPUBuffer() :
  size_(0), ptr_(0)
{
}

CPUBuffer::CPUBuffer(size_t size) :
  size_(size), ptr_(0)
{
  ptr_ = new char[size_];
}

CPUBuffer::CPUBuffer(const Buffer& toCopy) :
  size_(toCopy.getSize()), ptr_(0)
{
  ptr_ = new char[size_];
  toCopy.set(this, 0, size_, 0);
}

CPUBuffer& CPUBuffer::operator=(const Buffer& rhs) {
  if (this != &rhs) {
    size_ = rhs. getSize();
    if (ptr_ != 0) {
      delete [] ptr_;
    }
    ptr_ = new char[size_];
    rhs.set(this, 0, size_, 0);
  }
  return *this;
}

CPUBuffer::~CPUBuffer() {
  if (ptr_) {
    delete [] ptr_;
  }
}

void CPUBuffer::resize(size_t newsize) {
  if (ptr_) {
    delete [] ptr_;
    ptr_ = 0;
  }
  size_ = newsize;
  if (newsize > 0) {
    ptr_ = new char[size_];
  }
}

void CPUBuffer::set(Buffer* dest, size_t srcBegin, size_t srcEnd,
    size_t destBegin) const {
  dest->setFrom(*this, srcBegin, srcEnd, destBegin);
}

void CPUBuffer::setFrom(const CPUBuffer& src, size_t srcBegin,
    size_t srcEnd, size_t destBegin)  {
  if (srcEnd - srcBegin > size_ - destBegin) {
    throw std::runtime_error("Buffer overflow.");
  }
  memcpy(ptr_ + destBegin, (char*)src.getPtr() + srcBegin,
      srcEnd - srcBegin);
}

void CPUBuffer::setFrom(const PinnedCPUBuffer& src, size_t srcBegin,
    size_t srcEnd, size_t destBegin)  {
  setFrom((const CPUBuffer&)src, srcBegin, srcEnd, destBegin);
}


void CPUBuffer::setFrom(const GPUBuffer& src, size_t srcBegin,
    size_t srcEnd, size_t destBegin)  {
  if (srcEnd - srcBegin > size_ - destBegin) {
    throw std::runtime_error("Buffer overflow.");
  }
  cudaError_t err = cudaMemcpy(ptr_ + destBegin,
      (char*)src.getPtr() + srcBegin, srcEnd - srcBegin,
      cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cout << "Error code: " << err << std::endl;
    std::cout << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error("cudaMemcpy failed.");
  }
}

void CPUBuffer::setFrom(const void* src, size_t srcBegin,
    size_t srcEnd, size_t destBegin) {
  if (srcEnd - srcBegin > size_ - destBegin) {
    throw std::runtime_error("Buffer overflow.");
  }
  memcpy(ptr_ + destBegin, (char*)src + srcBegin, srcEnd - srcBegin);
}

void CPUBuffer::takeOwnership(const void* src, size_t num) {
  size_ = num;
  if (ptr_) {
    delete [] ptr_;
  }
  ptr_ = (char*) src;
}

void CPUBuffer::setPlainArray(void* dest, size_t srcBegin,
    size_t srcEnd, size_t destBegin) const {
  memcpy(dest, ptr_ + srcBegin, srcEnd - srcBegin);
}

void CPUBuffer::setToZero()
{
  memset((void*)ptr_, 0, size_);
}

void CPUBuffer::dump(std::ostream& stream, int numCols)
{
  Buffer::dump(stream, numCols);
}

void CPUBuffer::dump(std::ostream& stream, int numCols,
    size_t begin, size_t end)
{
  float* arr = (float*)ptr_;
  arr += begin / sizeof(float);
  size_t numEntries = (end - begin) / sizeof(float);
  int i = 0;
  int row = 0;
  while (i < numEntries - numCols) {
    for (int j = 0; j < numCols; ++j) {
      stream << arr[i] << " ";
      ++i;
    }
    stream << std::endl;
    ++row;
  }
  for(; i < numEntries; ++i) {
    stream << arr[i] << " ";
  }
  stream << std::endl;
}

bool CPUBuffer::hasNaNs(bool verbose) const
{
  int numEntries = size_ / sizeof(float);
  float* arr = (float*)ptr_;
  int i = 0;
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
      haveNaNs |= std::isnan(arr[i]);
#else
      haveNaNs |= _isnan(arr[i]);
#endif
      ++i;
    }
  }
  return haveNaNs;
}

