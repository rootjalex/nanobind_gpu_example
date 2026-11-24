#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "pool.h"
#include "utils.h"

namespace {

cudaMemPool_t get_pool() {
    static cudaMemPool_t pool = nullptr;
    if (!pool) {
        int device;
        CHECK_CUDA(cudaGetDevice(&device));
        cudaMemPoolProps props = {};
        props.allocType = cudaMemAllocationTypePinned;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id = device;
        CHECK_CUDA(cudaMemPoolCreate(&pool, &props));
        // Keep freed buffers in the pool
        uint64_t threshold = UINT64_MAX;
        CHECK_CUDA(cudaMemPoolSetAttribute(
            pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    }
    return pool;
}

}  // namespace

void* gpu_alloc(size_t bytes) {
    void* ptr = nullptr;
    // The pool can't allocate 0 bytes
    CHECK_CUDA(cudaMallocFromPoolAsync(&ptr, std::max<size_t>(bytes, 1),
                                       get_pool(), DEFAULT_STREAM));
    return ptr;
}

void gpu_free(void* ptr) noexcept {
    // Can't throw on failure due to `noexcept`
    cudaFreeAsync(ptr, DEFAULT_STREAM);
}

void gpu_free_unused() { CHECK_CUDA(cudaMemPoolTrimTo(get_pool(), 0)); }
