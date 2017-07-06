#ifndef PINNED_CPU_BUFFER_H
#define PINNED_CPU_BUFFER_H

#include "Buffer.h"
#include "CPUBuffer.h"

#include <cuda.h>

class GPUBuffer;

/**
 * @brief Buffer class for managing pinned host memory.
 */
class PinnedCPUBuffer : public CPUBuffer {

  public:
    /** Constructor. */
    PinnedCPUBuffer();
    /** Constructor with a certain size.
     * @param size Size of buffer.*/
    PinnedCPUBuffer(size_t size);
    /** Copy constructor.*/
    PinnedCPUBuffer(const Buffer& toCopy);
    /** Assignment operator.*/
    PinnedCPUBuffer& operator=(const Buffer& rhs);
    /** Destructor.*/
    virtual ~PinnedCPUBuffer();

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

    virtual void set(Buffer* dest, size_t srcBegin, size_t srcEnd,
        size_t destBegin) const;

    virtual void setFrom(const CPUBuffer& src, size_t srcBegin,
        size_t srcEnd, size_t destBegin);
    virtual void setFrom(const PinnedCPUBuffer& src, size_t srcBegin,
        size_t srcEnd, size_t destBegin);
    virtual void setFrom(const GPUBuffer& src, size_t srcBegin,
        size_t srcEnd, size_t destBegin);
    virtual void setFrom(const void* src, size_t srcBegin,
        size_t srcEnd, size_t destBegin);

    virtual bool hasNaNs(bool verbose = false) const;

  private:
    size_t size_;
    char* ptr_;
};

#endif
