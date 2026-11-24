#include <cuda_runtime.h>

#include <climits>
#include <stdexcept>

#include "add.h"
#include "pool.h"
#include "utils.h"

template <typename value_t, typename index_t>
__global__ void add_kernel(value_t* __restrict__ r,
                           const value_t* __restrict__ a,
                           const value_t* __restrict__ b, const index_t n) {
    // i must be 64-bit, n can exceed INT_MAX
    const index_t i =
        static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        r[i] = a[i] + b[i];
    }
}

float* gpu_add_f32(const float* x, const float* y, const uint64_t n) {
    Buffer<float> result(n);
    gpu_add_out_f32(x, y, result.ptr, n);
    return result.release();
}

void gpu_add_out_f32(const float* x, const float* y, float* result,
                     const uint64_t n) {
    if (n == 0) {
        return;
    }

    static int blockSize = -1;
    if (blockSize == -1) {
        // Only run the first call
        // Get a good launch config based on occupancy
        int minGridSize = 0;
        int occupancyBlockSize = 0;
        const size_t dynamicSharedMemPerBlock = 0;
        CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &occupancyBlockSize, add_kernel<float, uint64_t>,
            dynamicSharedMemPerBlock));
        blockSize = occupancyBlockSize;
    }

    const uint64_t gridSize = (n + blockSize - 1) / blockSize;
    if (gridSize > INT_MAX) {
        throw std::length_error("Failed to launch kernel, vector is too long");
    }

    add_kernel<float, uint64_t>
        <<<static_cast<unsigned int>(gridSize), blockSize>>>(result, x, y, n);

    CHECK_CUDA(cudaGetLastError());
}
