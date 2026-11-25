// Lets the unmodified spmspv-adaptive headers (and the ModernGPU 1.x they
// include) compile with CUDA 13. Pass this file to nvcc with -include.
#pragma once
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda/std/utility>

// thrust::pair was removed in Thrust 3, the algorithms return this type
namespace thrust {
using ::cuda::std::pair;
}

// ModernGPU uses __shfl_up, which only exists before sm_70. spmspv-adaptive
// includes ModernGPU but never calls these.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__device__ __forceinline__ int __shfl_up(int var, unsigned int delta, int width = 32) {
  return __shfl_up_sync(__activemask(), var, delta, width);
}
__device__ __forceinline__ float __shfl_up(float var, unsigned int delta, int width = 32) {
  return __shfl_up_sync(__activemask(), var, delta, width);
}
#endif
#endif  // __CUDACC__
