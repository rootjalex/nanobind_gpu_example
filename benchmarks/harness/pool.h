// A CUDA memory pool that keeps freed buffers, as in src/cuda/kernels/pool.cu.
// Every system in the benchmark allocates its GPU memory from such a pool:
// ours through src/cuda/kernels/pool.cu, the others through this file.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace harness {

inline cudaMemPool_t get_pool() {
    static cudaMemPool_t pool = nullptr;
    if (!pool) {
        int device;
        cudaGetDevice(&device);
        cudaMemPoolProps props = {};
        props.allocType = cudaMemAllocationTypePinned;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id = device;
        cudaMemPoolCreate(&pool, &props);
        uint64_t threshold = UINT64_MAX;
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                &threshold);
    }
    return pool;
}

// Stands in for cudaMalloc
template <typename T>
cudaError_t pool_malloc(T** ptr, size_t bytes) {
    // The pool can't allocate 0 bytes
    return cudaMallocFromPoolAsync(reinterpret_cast<void**>(ptr),
                                   bytes ? bytes : 1, get_pool(), nullptr);
}

// Stands in for cudaFree
inline cudaError_t pool_free(void* ptr) {
    return ptr ? cudaFreeAsync(ptr, nullptr) : cudaSuccess;
}

}  // namespace harness
