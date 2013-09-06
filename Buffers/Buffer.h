#ifndef BUFFER_H
#define BUFFER_H

#include <cstring>
#include <ostream>

class CPUBuffer;
class PinnedCPUBuffer;
class GPUBuffer;

/** Buffer class for managing flat memory.
 *
 * This class serves two main purposes:
 * 1) Manage memory resources
 * 2) Transfer data between different buffers that could reside on
 * different devices (e.g. host vs gpu or in a multi-gpu environment)
 *
 * @brief Interface for Buffer classes used for managing memory on CPUs,
 * GPUs, and data transfers.
 * */
class Buffer {

  public:
    /** Destructor*/
    virtual ~Buffer();

    /** Get current size of buffer.*/
    virtual size_t getSize() const = 0;
    /** Get pointer to the memory managed by the buffer.*/
    virtual void* getPtr() = 0;
    /** Get a const pointer to the memory managed by the buffer.*/
    virtual const void* getPtr() const = 0;
    /** Resize the buffer.  The old data becomes invalid after a call to
     * resize.
     * @param newsize New size of buffer.
     * */
    virtual void resize(size_t newsize) = 0;

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
        size_t destBegin) const = 0;

    /** Set this buffer from a src CPUBuffer.
     * @param src         Source buffer.
     * @param srcBegin    Beginning of slice that is copied into this.
     * @param srcEnd      End of slice that is copied into this.
     * @param destBegin   Offset in dest.
     * */
    virtual void setFrom(const CPUBuffer& src, size_t srcBegin,
        size_t srcEnd, size_t destBegin) = 0;
    /** Set this buffer from a src PinnedCPUBuffer.  Data transfers
     * between GPUBuffer and PinnedCPUBuffer objects are done
     * asynchronously.
     * @param src         Source buffer.
     * @param srcBegin    Beginning of slice that is copied into this.
     * @param srcEnd      End of slice that is copied into this.
     * @param destBegin   Offset in dest.
     * */
    virtual void setFrom(const PinnedCPUBuffer& src, size_t srcBegin,
        size_t srcEnd, size_t destBegin) = 0;
    /** Set this buffer from a src GPUBuffer.  Currently only setting
     * from a buffer on the same device is supported.
     * @param src         Source buffer.
     * @param srcBegin    Beginning of slice that is copied into this.
     * @param srcEnd      End of slice that is copied into this.
     * @param destBegin   Offset in dest.
     * */
    virtual void setFrom(const GPUBuffer& src, size_t srcBegin,
        size_t srcEnd, size_t destBegin) = 0;

    /** Set contents of Buffer to zero. */
    virtual void setToZero() = 0;

    /** Dump contents of buffer to a stream
     * @param stream Stream to which to write.
     * @param numCols Number of entries per line
     */
    void dump(std::ostream& stream, int numCols);
    /** Dump part of a Buffer to a stream
     * @param stream Stream to which to write.
     * @param numCols Number of entries per line
     * @param begin Start of part of buffer to be dumped (in bytes)
     * @param end End of part of the buffer to be dumped (in bytes)
     * */
    virtual void dump(std::ostream& stream, int numCols,
        size_t begin, size_t end) = 0;

    /** Check if there are any nans in the buffer.  This is an expensive
     * operation that is carried out on the CPU.  The entire buffer is
     * traversed as if it were an array of floats and each entry is
     * checked to see whether it is nan.  To avoid excessive slow down
     * in the application this function should typically be wrapped in a
     * conditional compilation section (e.g. using #ifndef NDEBUG).
     * @param verbose If this flag is set to true all nan entries are
     * printed to the std::cout.  If the flag is false the function
     * returns when the first nan value is encountered.
     * */
    virtual bool hasNaNs(bool verbose = false) const = 0;
};

/**
 * \mainpage Buffer classes for managing memory allocation and data transfers
 *
 *
 * \section intro_sec Introduction
 *
 * The purpose of the various Buffer classes is to automate memory
 * allocation and deallocation and to make data transfers between
 * different memory spaces (e.g. CPU and GPU) easier. 
 *
 *
 * \section allocation Memory allocation and deallocation
 *
 * In general, memory resources are allocated by creating Buffer
 * objects.  Buffers can be resized if the memory requirements change of
 * if the memory is no longer needed.  In case of the latter one should
 * resize to 0.  Memory is released automatically when the Buffer object
 * goes out of scope.
 *
 *
 * \section transfers Data transfers
 *
 * Data transfers are typically done using the set method on the source
 * buffer with a pointer to the target buffer as an argument.  The
 * various setFrom() methods are a lower level mechanism that allows set
 * to work polymorphically for any buffer as the destination.  In
 * general the set method should be preferred over the setFrom() methods
 * in application code.  The purpose of the other arguments of the set
 * method is to enable copying a slice of the source buffer to a
 * specific offset in the target buffer (e.g. a two dimensional slice in
 * the x-y plane). 
 *
 * Asynchronous data transfers can be achieved by using PinnedCPUBuffer
 * objects on the host side.
 *
 * It is possible to sidestep the data transfer mechanisms provided by
 * the Buffer classes.  To this end one can get the raw pointers to the
 * underlying memory using the getPtr() methods.
 *
 *
 * \section example Example
 *
 * An example of how this may be used is shown by the following program.
 *
 * \include bufferExample.cpp
 *
* 
 */

/** \def NDEBUG
 * This macro controlls whether assert statements are evaluated or not.
 * If NDEBUG is defined assert statements are not compiled.
 */
#endif

