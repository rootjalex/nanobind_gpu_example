#include <cuda_runtime.h>

#include "free.h"

void releaseBuffer(void* ptr) noexcept {
    // Runs from a capsule destructor, so it must not throw: a failed free is
    // deliberately ignored.
    cudaFree(ptr);
}
