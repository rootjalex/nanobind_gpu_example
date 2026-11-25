// Replaces cudaMalloc and cudaFree in spmspv-adaptive without editing it.
// Pass this file to nvcc with -include, after lay_compat.h.
//
// spmspv-adaptive allocates with cudaMalloc, which is far slower than the
// CUDA memory pool that the other systems in the benchmark allocate from.
// Give it the same kind of pool.
//
// The replacement also works around a buffer overrun:
// csc_sort_load_balance.cuh line 327 allocates
//   numPartitions + 1 * sizeof(int)
// bytes for numPartitions + 1 ints, and the next kernel writes all of them,
// which corrupts other buffers. Give the allocations of that file 4 times
// the bytes they ask for, which is enough room.
#pragma once
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cstring>

#include "../../pool.h"

template <typename T>
inline cudaError_t lay_compat_cudaMalloc(T** ptr, size_t size, const char* file) {
  const char* name = strrchr(file, '/');
  name = name ? name + 1 : file;
  if (strcmp(name, "csc_sort_load_balance.cuh") == 0) {
    size *= 4;
  }
  return harness::pool_malloc(ptr, size);
}
#define cudaMalloc(ptr, size) lay_compat_cudaMalloc((ptr), (size), __FILE__)
#define cudaFree harness::pool_free
#endif  // __CUDACC__
