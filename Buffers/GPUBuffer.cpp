#include "GPUBuffer.h"

#include "CPUBuffer.h"
#include "PinnedCPUBuffer.h"

bool firstcall = true;

GPUBuffer::GPUBuffer() :
  device_(0), size_(0), ptr_(0), Hostptr_(0)
{
}

GPUBuffer::GPUBuffer(int device) :
  device_(device), size_(0), ptr_(0), Hostptr_(0)
{
}

GPUBuffer::GPUBuffer(size_t size, int device) :
device_(device), size_(size), ptr_(0), Hostptr_(0)
{
  cudaError_t err = cudaSetDevice(device_);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaSetDevice failed.");
  }
  err = cudaMalloc((void**)&ptr_, size_);
  if (err != cudaSuccess) {
	  err = cudaHostAlloc((void**)&Hostptr_, size_, cudaHostAllocMapped); // if device allocation fails, try to allocate on Host
	  if (err != cudaSuccess) 
		  throw std::runtime_error("cudaMalloc and cudaHostAlloc failed.");
	  else {
		  cudaHostGetDevicePointer((void**)&ptr_, Hostptr_, 0);
		  size_t free;
		  size_t total;
		  cudaMemGetInfo(&free, &total);
		  if (firstcall)
			  std::cout << "Want new " << size_ / (1024 * 1024) << " MB of GPU RAM. " << free / (1024 * 1024) << " MB free / " << total / (1024 * 1024) << " MB total. Use Host RAM..." << std::endl;
		  firstcall = false;
	  }
  }
}

GPUBuffer::GPUBuffer(const GPUBuffer& toCopy) :
device_(toCopy.device_), size_(toCopy.size_), ptr_(0), Hostptr_(0)
{
  this->resize(size_);
  toCopy.set(this, 0, size_, 0);
}

GPUBuffer::GPUBuffer(const Buffer& toCopy, int device) :
device_(device), size_(toCopy.getSize()), ptr_(0), Hostptr_(0)
{
  this->resize(size_);
  toCopy.set(this, 0, size_, 0);
}

GPUBuffer& GPUBuffer::operator=(const GPUBuffer& rhs) {
  if (this->device_ != rhs.device_) {
    throw std::runtime_error(
        "Different devices in GPUBuffer::operator=.");
  }
  if (this != &rhs) {
    size_ = rhs.getSize();
    this->resize(size_);
    rhs.set(this, 0, size_, 0);
  }
  return *this;
}

GPUBuffer& GPUBuffer::operator=(const CPUBuffer& rhs) {
  size_ = rhs.getSize();
  this->resize(size_);
  rhs.set(this, 0, size_, 0);
  return *this;
}

GPUBuffer::~GPUBuffer() {
	if (Hostptr_){
		cudaError_t err = cudaFreeHost(Hostptr_);
		ptr_ = 0;
		Hostptr_ = 0;
	}
	else
		if (ptr_) {
			cudaError_t err = cudaFree(ptr_);
			if (err != cudaSuccess) {
				std::cout << "sCudaFree failed. Error code: " << err << std::endl;
				std::cout << "ptr_: " << (long long int)ptr_ << std::endl;
				throw std::runtime_error("cudaFree failed.");
			}
			ptr_ = 0;
		}
}

void GPUBuffer::resize(size_t newsize) {
	if (Hostptr_){
		cudaError_t err = cudaFreeHost(Hostptr_);
		ptr_ = 0;
		Hostptr_ = 0;
	}
	else
		if (ptr_) {
		  cudaError_t err = cudaFree(ptr_);
		  if (err != cudaSuccess) {
		      throw std::runtime_error("cudaFree failed.");
			  }
		 ptr_ = 0;
		}
  cudaError_t err = cudaSetDevice(device_);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaSetDevice failed.");
  }
  size_ = newsize;
  if (newsize > 0) {
	  err = cudaMalloc((void**)&ptr_, size_);
	  if (err != cudaSuccess) {
		  err = cudaHostAlloc((void**)&Hostptr_, size_, cudaHostAllocMapped); // if device allocation fails, try to allocate on Host
		  if (err != cudaSuccess)
			  throw std::runtime_error("cudaMalloc and cudaHostAlloc failed.");
		  else {
			  cudaHostGetDevicePointer((void**)&ptr_, Hostptr_, 0);
			  size_t free;
			  size_t total;
			  cudaMemGetInfo(&free, &total);
			  if (firstcall)
				  std::cout << "Want resize" << size_ / (1024 * 1024) << " MB of GPU RAM. " << free / (1024 * 1024) << " MB free / " << total / (1024 * 1024) << " MB total. Use Host RAM..." << std::endl;
			  firstcall = false;
		  }
	  }
  }
}

void GPUBuffer::setPtr(char* ptr, size_t size, int device)
{
	if (Hostptr_){
		std::cout << "setPtr Line:" << __LINE__ << std::endl;
		cutilSafeCall(cudaFreeHost(Hostptr_));
	}
	else
	  if (ptr_)
		cutilSafeCall(cudaFree(ptr_));
  
  ptr_ = ptr;
  size_ = size;
  device_ = device;

}

void GPUBuffer::set(Buffer* dest, size_t srcBegin, size_t srcEnd,
    size_t destBegin) const {
  dest->setFrom(*this, srcBegin, srcEnd, destBegin);
}

void GPUBuffer::setFrom(const CPUBuffer& src, size_t srcBegin,
    size_t srcEnd, size_t destBegin)  {
  if (srcEnd - srcBegin > size_ - destBegin) {
    std::cout << "Trying to write " << srcEnd - srcBegin << " bytes\n";
    std::cout << "To buffer of size " << size_ << " bytes\n";
    std::cout << "With offset " << destBegin << " bytes\n";
    std::cout << "Overflow by " << (int)size_ - destBegin - (srcEnd - srcBegin) << "\n";
    std::cout << std::endl;
    throw std::runtime_error("Buffer overflow.");
  }
  cudaError_t err = cudaMemcpy(ptr_ + destBegin,
      (char*)src.getPtr() + srcBegin, srcEnd - srcBegin,
      cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed.");
  }
}

void GPUBuffer::setFrom(const PinnedCPUBuffer& src, size_t srcBegin,
    size_t srcEnd, size_t destBegin)  {
  if (srcEnd - srcBegin > size_ - destBegin) {
    std::cout << "Trying to write " << srcEnd - srcBegin << " bytes\n";
    std::cout << "To buffer of size " << size_ << " bytes\n";
    std::cout << "With offset " << destBegin << " bytes\n";
    std::cout << "Overflow by " << (int)size_ - destBegin - (srcEnd - srcBegin) << "\n";
    std::cout << std::endl;
    throw std::runtime_error("Buffer overflow.");
  }
  cudaError_t err = cudaMemcpyAsync(ptr_ + destBegin,
      (char*)src.getPtr() + srcBegin, srcEnd - srcBegin,
      cudaMemcpyHostToDevice, 0);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed.");
  }
}

void GPUBuffer::setFrom(const GPUBuffer& src, size_t srcBegin,
    size_t srcEnd, size_t destBegin)  {
  if (this->device_ != src.device_) {
    throw std::runtime_error(
        "Currently setFrom only supports transferring data within the "
        "same device or between host and device.");
  }
  if (srcEnd - srcBegin > size_ - destBegin) {
    throw std::runtime_error("Buffer overflow.");
  }
  cudaError_t err = cudaMemcpy(ptr_ + destBegin,
      (char*)src.getPtr() + srcBegin, srcEnd - srcBegin,
      cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaMemcpy failed.");
  }
}

void GPUBuffer::dump(std::ostream& stream, int numCols)
{
  Buffer::dump(stream,numCols);
}

void GPUBuffer::setToZero()
{
  cudaMemset(ptr_, 0, size_);
}

void GPUBuffer::dump(std::ostream& stream, int numCols,
    size_t begin, size_t end)
{
  CPUBuffer cpuBuff(*this);
  cpuBuff.dump(stream, numCols, begin, end);
}

bool GPUBuffer::hasNaNs(bool verbose) const
{
  CPUBuffer tmp(size_);
  this->set(&tmp, 0, size_, 0);
  return tmp.hasNaNs(verbose);
}
