#include "GPUBuffer.h"
#include "CPUBuffer.h"
#include "PinnedCPUBuffer.h"
#include <Windows.h>


bool firstcall = true;

GPUBuffer::GPUBuffer() :
device_(0), size_(0), ptr_(0), Hostptr_(0), UseCudaHostOnly_(false)
{
}

GPUBuffer::GPUBuffer(int device, bool UseCudaHostOnly) :
device_(device), size_(0), ptr_(0), Hostptr_(0), UseCudaHostOnly_(UseCudaHostOnly)
{
}

GPUBuffer::GPUBuffer(size_t size, int device, bool UseCudaHostOnly) :
device_(device), size_(size), ptr_(0), Hostptr_(0), UseCudaHostOnly_(UseCudaHostOnly)
{
  cudaError_t err = cudaSetDevice(device_);
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaSetDevice failed.");
  }

  if (!UseCudaHostOnly_)
	err = cudaMalloc((void**)&ptr_, size_);

  if (err != cudaSuccess || UseCudaHostOnly_) {
      err = cudaHostAlloc((void**)&Hostptr_, size_, cudaHostAllocMapped); // if device allocation fails, try to allocate on Host
      if (err != cudaSuccess) 
          throw std::runtime_error("cudaMalloc and cudaHostAlloc failed.");
      else {
          cudaHostGetDevicePointer((void**)&ptr_, Hostptr_, 0);
          size_t free;
          size_t total;
          cudaMemGetInfo(&free, &total);
		  if (firstcall){
			  HANDLE  hConsole;
			  hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
			  SetConsoleTextAttribute(hConsole, 6); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
			  std::cout << "Want new " << size_ / (1024 * 1024) << " MB of GPU RAM. " << free / (1024 * 1024) << " MB free / " << total / (1024 * 1024) << " MB total. Use Host RAM..." << std::endl;
			  SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
		  }
          firstcall = false;
      }
  }
}

GPUBuffer::GPUBuffer(const GPUBuffer& toCopy) :
device_(toCopy.device_), size_(0), ptr_(0), Hostptr_(0), UseCudaHostOnly_(toCopy.UseCudaHostOnly_)
{
	this->resize(toCopy.size_);
	size_ = toCopy.size_;
	toCopy.set(this, 0, size_, 0);
}

GPUBuffer::GPUBuffer(const Buffer& toCopy, int device, bool UseCudaHostOnly) :
device_(device), size_(0), ptr_(0), Hostptr_(0), UseCudaHostOnly_(UseCudaHostOnly)
{
	this->resize(toCopy.getSize());
	size_ = toCopy.getSize();
	toCopy.set(this, 0, size_, 0);
}

GPUBuffer& GPUBuffer::operator=(const GPUBuffer& rhs) {
  if (this->device_ != rhs.device_) {
    throw std::runtime_error(
        "Different devices in GPUBuffer::operator=.");
  }
  if (this != &rhs) {
    this->resize(rhs.getSize()); //Set the left hand side to the correct size
	//std::cout << "Resize complete. ";
	size_ = rhs.getSize();
	rhs.set(this, 0, size_, 0);  //Copy data from right hand side to left hand side.
	//std::cout << "rhs.set complete. ";
  }
  return *this;
}

GPUBuffer& GPUBuffer::operator=(const CPUBuffer& rhs) {
  this->resize(rhs.getSize());	//Set the left hand side to the correct size
  size_ = rhs.getSize();
  rhs.set(this, 0, size_, 0);   //Copy data from right hand side to left hand side.
  return *this;					
}

GPUBuffer::~GPUBuffer() {
    if (Hostptr_){
        cudaError_t err = cudaFreeHost(Hostptr_);
        if (err != cudaSuccess) {
			std::cerr << "cudaFreeHost failed during destructor. Error code: " << err << ". " << cudaGetErrorString(err) << std::endl;
			std::cerr << "Hostptr_: " << (long long int)Hostptr_ << std::endl;
          throw std::runtime_error("cudaFreeHost failed.");
        }
        ptr_ = 0;
        Hostptr_ = 0;
    }
    else
        if (ptr_) {
            cudaError_t err = cudaFree(ptr_);
            if (err != cudaSuccess) {
				std::cerr << "CudaFree failed during destructor. Error code: " << err << ". " << cudaGetErrorString(err) << std::endl;
				std::cerr << "ptr_: " << (long long int)ptr_ << std::endl;
                throw std::runtime_error("cudaFree failed.");
            }
            ptr_ = 0;
        }
}

void GPUBuffer::resize(size_t newsize) {
	if (size_ != newsize){	// if we need to resize
		// std::cout << "Need to resize.  size_ = " << size_ << " newsize = " << newsize << std::endl;
		if (Hostptr_){				// if this is a host pointer, then free it.
			cudaError_t err = cudaFreeHost(Hostptr_);
			if (err != cudaSuccess) {
				std::cerr << "cudaFreeHost failed during resize. Error code: " << err << ". " << cudaGetErrorString(err) << std::endl;
				std::cerr << "Hostptr_: " << (long long int)Hostptr_ << std::endl;
				throw std::runtime_error("cudaFreeHost failed.");
			}
			ptr_ = 0;
			Hostptr_ = 0;
		}

		else
			if (ptr_) {				// if this is a GPU pointer, then free it.
				cudaError_t err = cudaFree(ptr_);
				if (err != cudaSuccess) {
					throw std::runtime_error("cudaFree failed during resize.");
				}
				ptr_ = 0;
			}


		cudaError_t err = cudaSetDevice(device_);
		if (err != cudaSuccess) {
			throw std::runtime_error("cudaSetDevice failed during resize.");
		}

		size_ = newsize;

		if (newsize > 0) {
			if (!UseCudaHostOnly_)
				err = cudaMalloc((void**)&ptr_, size_);

			if (err != cudaSuccess || UseCudaHostOnly_) { // if device allocation fails, try to allocate on Host
				err = cudaHostAlloc((void**)&Hostptr_, size_, cudaHostAllocMapped);
				if (err != cudaSuccess) // if Host allocation failed.
					throw std::runtime_error("cudaMalloc and cudaHostAlloc failed during resize.");
				else {
					err = cudaHostGetDevicePointer((void**)&ptr_, Hostptr_, 0); //if succeeded, then get pointer
					if (err != cudaSuccess) // if getting pointer failed.
						throw std::runtime_error("cudaHostGetDevicePointer failed during resize.");
					size_t free;
					size_t total;
					cudaMemGetInfo(&free, &total);
					if (firstcall)
					{
						HANDLE  hConsole;
						hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
						SetConsoleTextAttribute(hConsole, 6); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
						std::cout << "Resizing buffer. " << size_ / (1024 * 1024) << " MB of GPU RAM. " << free / (1024 * 1024) << " MB free / " << total / (1024 * 1024) << " MB total. Use Host RAM..." << std::endl;
						SetConsoleTextAttribute(hConsole, 7); // colors are 9=blue 10=green and so on to 15=bright white 7=normal http://stackoverflow.com/questions/4053837/colorizing-text-in-the-console-with-c
					}
					firstcall = false;
				}
			}

		}
	} //end if we need to resize
	
	//else
		

}

void GPUBuffer::setPtr(char* ptr, char* Hostptr, size_t size, int device)
{
    if (Hostptr_){
        std::cout << "setPtr Line:" << __LINE__ << std::endl;
        cutilSafeCall(cudaFreeHost(Hostptr_));
    }
    else
      if (ptr_)
        cutilSafeCall(cudaFree(ptr_));
  
  ptr_ = ptr;
  Hostptr_ = Hostptr;
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
