#ifndef CPU_BUFFER_H
#define CPU_BUFFER_H

#include "Buffer.h"
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

class PinnedCPUBuffer;
class GPUBuffer;

/**
 * @brief Buffer class for managing memory on CPU side.
 */
class CPUBuffer : public Buffer {

  public:
    /** Constructor. */
    CPUBuffer();
    /** Constructor with a certain size.
     * @param size Size of buffer.*/
    CPUBuffer(size_t size);
    /** Copy constructor.*/
    CPUBuffer(const Buffer& toCopy);
    /** Assignment operator.*/
    CPUBuffer& operator=(const Buffer& rhs);
    /** Destructor.*/
    virtual ~CPUBuffer();

    /** Get current size of buffer.*/
    virtual size_t getSize() const { return size_; } ;
    /** Get pointer to the memory managed by the buffer.  This is a host
     * pointer.*/
    virtual void* getPtr() { return ptr_; } ;
    virtual const void* getPtr() const { return ptr_; } ;
    /** Resize the buffer.  The old data becomes invalid after a call to
     * resize.
     * @param newsize New size of buffer.
     * */
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
    /** Set this buffer from a plain array.
     * @param src         Source buffer.
     * @param srcBegin    Beginning of slice that is copied into this.
     * @param srcEnd      End of slice that is copied into this.
     * @param destBegin   Offset in dest.
     * */
    virtual void setFrom(const void* src, size_t srcBegin,
        size_t srcEnd, size_t destBegin);

    /** Use a host pointer for the buffer.  Note that CPUBuffer calls
     * delete [] on this pointer when it is destroyed.  src must not be
     * deleted by the client code.
     * @param src   Pointer to valid host memory.  Has to be at least of
     *              size num.
     * @param num   Size in bytes of src.*/
    void takeOwnership(const void* src, size_t num);
    /** Copy the contents of the buffer to a plain array.  Semantics is
     * identical to the set method.
     * @param dest        Plain array that is to be set.
     * @param srcBegin    Beginning of slice that is copied into dest.
     * @param srcEnd      End of slice that is copied into dest.
     * @param destBegin   Offset into dest.
     * */
    virtual void setPlainArray(void* dest, size_t srcBegin,
        size_t srcEnd, size_t destBegin) const;

    virtual void setToZero();

    void dump(std::ostream& stream, int numCols);
    virtual void dump(std::ostream& stream, int numCols,
        size_t begin, size_t end);

    virtual bool hasNaNs(bool verbose = false) const;

  private:
    size_t size_;
    char* ptr_;
};

#endif

