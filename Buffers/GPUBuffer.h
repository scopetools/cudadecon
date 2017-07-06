#ifndef GPU_BUFFER_H
#define GPU_BUFFER_H

#include "Buffer.h"
#include <cstring>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ostream>
#include <iostream>
#include "../cutilSafeCall.h"

class CPUBuffer;
class PinnedCPUBuffer;

/** A class for managing flat GPU memory.  The GPU memory managed by a
 * GPUBuffer is freed when the buffer is destroyed (e.g. when it goes
 * out of scope). 
 * @brief Class for managing GPU memory.
 * */
class GPUBuffer : public Buffer {

  public:
    /** Constructor.  Creates a buffer on default cuda device 0.*/
    GPUBuffer();

    /** Create a buffer on a specific cuda device.
     * @param device Cuda device on which to create the Buffer.
     * */
	GPUBuffer(int device, bool UseCudaHostOnly);

    /** Create a buffer of a certain size on a specific cuda device.
     * @param size Size of buffer in bytes.
     * @param device Cuda device on which to create the Buffer.
     * */
	GPUBuffer(size_t size, int device, bool UseCudaHostOnly);

    /** Copy constructor.*/
    GPUBuffer(const GPUBuffer& toCopy);

    /** Copy a GPU Buffer to a different device.
     * @param toCopy GPUBuffer that is to be copied.
     * @param device Cuda device on which to create the new GPUBuffer.*/
	GPUBuffer(const Buffer& toCopy, int device, bool UseCudaHostOnly);

    /** Set a GPUBuffer from a different GPUBuffer.
     * @param rhs GPUBuffer from which to set this GPUBuffer.*/
    GPUBuffer& operator=(const GPUBuffer& rhs);

    /** Set a GPUBuffer from a CPUBuffer.
     * @param rhs CPUBuffer from which to set this GPUBuffer.*/
    GPUBuffer& operator=(const CPUBuffer& rhs);

    /** Destructor.  Frees the GPU memory managed by this GPUBuffer.*/
    virtual ~GPUBuffer();

    virtual size_t getSize() const { return size_; } ;
    virtual void* getPtr() { return ptr_; } ;
    virtual const void* getPtr() const { return ptr_; } ;
    void * getHostptr() { return Hostptr_; };
    const void* getHostptr() const { return Hostptr_; };

    /** Set the pointer managed by this GPUBuffer to ptr.  The memory
     * managed previously by this Buffer is released.
     * @param ptr Device pointer to GPU memory.
     * @param size Size of memory pointed to by ptr.
     * @param device Cuda device on which the memory pointed to by ptr
     * is located.*/
    virtual void setPtr(char* ptr, char *Hostptr, size_t size, int device);

    /** Change the size of the GPUBuffer.  The data held by the buffer
     * becomes invalid, even when the size of the buffer is increased.
     * Setting the size of the buffer to zero frees all GPU memory.
     * @param newsize New size of GPU buffer.*/
    virtual void resize(size_t newsize);

    /** Copy a slice of this buffer into dest.  The slice in this starts
     * at srcBegin and ends at srcEnd.  The slice is copied into dest
     * starting at destBegin.  The parameters srcBegin, srcEnd, and
     * destBegin are in bytes.
     * @param dest        Buffer that is to be set.
     * @param srcBegin    Beginning of slice that is copied into Buffer.
     * @param srcEnd      End of slice that is copied into Buffer.
     * @param destBegin   Offset into dest.
     * */
    virtual void set(Buffer* dest, size_t srcBegin, size_t srcEnd,
        size_t destBegin) const;

    /** Set this buffer from a src CPUBuffer.
     * @param src         Source buffer.
     * @param srcBegin    Beginning of slice that is copied into this.
     * @param srcEnd      End of slice that is copied into this.
     * @param destBegin   Offset in dest.
     * */
    virtual void setFrom(const CPUBuffer& src, size_t srcBegin,
        size_t srcEnd, size_t destBegin);

    /** Set this buffer from a src PinnedCPUBuffer.  Data transfers
     * between GPUBuffer and PinnedCPUBuffer objects are done
     * asynchronously.
     * @param src         Source buffer.
     * @param srcBegin    Beginning of slice that is copied into this.
     * @param srcEnd      End of slice that is copied into this.
     * @param destBegin   Offset in dest.
     * */
    virtual void setFrom(const PinnedCPUBuffer& src, size_t srcBegin,
        size_t srcEnd, size_t destBegin);

    /** Set this buffer from a src GPUBuffer.  Currently only setting
     * from a buffer on the same device is supported.
     * @param src         Source buffer.
     * @param srcBegin    Beginning of slice that is copied into this.
     * @param srcEnd      End of slice that is copied into this.
     * @param destBegin   Offset in dest.
     * */
    virtual void setFrom(const GPUBuffer& src, size_t srcBegin,
        size_t srcEnd, size_t destBegin);

    virtual void setToZero();

    void dump(std::ostream& stream, int numCols);
    virtual void dump(std::ostream& stream, int numCols,
        size_t begin, size_t end);

    virtual bool hasNaNs(bool verbose = false) const;

  private:
    int device_;
    size_t size_;
    char* ptr_;
    char* Hostptr_;
	bool UseCudaHostOnly_;
};

#endif

