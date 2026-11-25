// Lets unmodified GraphBLAST (and its copy of ModernGPU 1.x) compile with
// CUDA 13. Pass this file to nvcc with -include.
#pragma once
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cusparse.h>
// CUB must be included before graphblas/util.hpp, which has
// `using namespace graphblas;` and its own templates named first and second
#include <cub/cub.cuh>

// cudaThreadSynchronize was removed in CUDA 12
inline cudaError_t cudaThreadSynchronize() { return cudaDeviceSynchronize(); }

// The cuSPARSE csrgemm2 functions were removed in CUDA 12. GraphBLAST only
// uses them for SpGEMM (mxm), so these do nothing.
typedef struct grb_compat_csrgemm2Info* csrgemm2Info_t;
inline cusparseStatus_t cusparseCreateCsrgemm2Info(csrgemm2Info_t*) { return CUSPARSE_STATUS_NOT_SUPPORTED; }
template <typename... Args> cusparseStatus_t cusparseScsrgemm2_bufferSizeExt(Args&&...) { return CUSPARSE_STATUS_NOT_SUPPORTED; }
template <typename... Args> cusparseStatus_t cusparseXcsrgemm2Nnz(Args&&...) { return CUSPARSE_STATUS_NOT_SUPPORTED; }
template <typename... Args> cusparseStatus_t cusparseScsrgemm2(Args&&...) { return CUSPARSE_STATUS_NOT_SUPPORTED; }

// ModernGPU uses __shfl_up, which only exists before sm_70
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
__device__ __forceinline__ int __shfl_up(int var, unsigned int delta, int width = 32) {
  return __shfl_up_sync(__activemask(), var, delta, width);
}
__device__ __forceinline__ float __shfl_up(float var, unsigned int delta, int width = 32) {
  return __shfl_up_sync(__activemask(), var, delta, width);
}
#endif

// mgpu::shfl_add and mgpu::shfl_max are inline PTX `shfl.up.b32`, which
// ptxas rejects from sm_70 on. Rename the originals while their header is
// parsed, and define the same functions with `shfl.sync.up.b32`.
#define shfl_add grb_compat_orig_shfl_add
#define shfl_max grb_compat_orig_shfl_max
#include <device/intrinsics.cuh>
#undef shfl_add
#undef shfl_max
namespace mgpu {
MGPU_DEVICE int shfl_add(int x, int offset, int width = WARP_SIZE) {
  int result = 0;
  int mask = (WARP_SIZE - width) << 8;
  asm("{.reg .s32 r0;"
      ".reg .pred p;"
      "shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
      "@p add.s32 r0, r0, %4;"
      "mov.s32 %0, r0; }"
      : "=r"(result) : "r"(x), "r"(offset), "r"(mask), "r"(x), "r"(__activemask()));
  return result;
}
MGPU_DEVICE int shfl_max(int x, int offset, int width = WARP_SIZE) {
  int result = 0;
  int mask = (WARP_SIZE - width) << 8;
  asm("{.reg .s32 r0;"
      ".reg .pred p;"
      "shfl.sync.up.b32 r0|p, %1, %2, %3, %5;"
      "@p max.s32 r0, r0, %4;"
      "mov.s32 %0, r0; }"
      : "=r"(result) : "r"(x), "r"(offset), "r"(mask), "r"(x), "r"(__activemask()));
  return result;
}
}  // namespace mgpu

// The masks must be __activemask(). ModernGPU calls these inside branches,
// and a full mask hangs the kernel.

// ModernGPU uses __ballot, which only exists before sm_70
#define __ballot(pred) __ballot_sync(__activemask(), (pred))

// cudaDeviceProp::clockRate and memoryClockRate were removed in CUDA 13.
// ModernGPU only prints them.
#define memoryClockRate memoryBusWidth
#define clockRate memoryBusWidth
#endif  // __CUDACC__
