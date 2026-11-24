#include <cuda_runtime.h>

#include "free.h"

// Wrapper function for cudaFree
void releaseBuffer(void* ptr) noexcept {
    // Can't throw on failure due to `noexcept`
    cudaFree(ptr);
}
