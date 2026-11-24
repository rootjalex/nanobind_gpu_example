#include <cuda_runtime.h>

#include "free.h"
#include "pool.h"

// Wrapper function for gpu_free
void releaseBuffer(void* ptr) noexcept {
    // Can't throw on failure due to `noexcept`
    gpu_free(ptr);
}
