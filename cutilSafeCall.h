#ifndef CUTIL_SAVE_CALL_H
#define CUTIL_SAVE_CALL_H

#include <cstdio>

// utilities for safe cuda api calls copied from cuda sdk.
// (currently these aren't exported -- file local)
#define cutilSafeCallNoSync(err)     __cudaSafeCallNoSync(err, __FILE__, __LINE__)
#define cutilSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
inline static void __cudaSafeCall(cudaError_t err, const char *file, const int line)
{
    if (cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}
inline static void __cudaSafeCallNoSync(cudaError_t err, const char *file, const int line)
{
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : cudaSafeCallNoSync() Runtime API error %d : %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

#endif
