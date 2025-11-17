#include <cuda_runtime.h>

#include <iostream>
#include <sstream>

#include "add.h"
#include "utils.h"

template<typename value_t, typename size_t>
__global__ void add_kernel(value_t *__restrict__ r, const value_t *__restrict__ a, const value_t *__restrict__ b, const size_t n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        r[i] = a[i] + b[i];
    }
}

float *gpu_add_f32(const float *x, const float *y, const uint64_t n) {
    float *result;
    CHECK_CUDA( cudaMalloc(&result, n * sizeof(float)) );
    gpu_add_out_f32(x, y, result, n);
    return result;
}

void gpu_add_out_f32(const float *x, const float *y, float *result, const uint64_t n) {
    // Get device properties
    static int blockSize = -1;
    if (blockSize == -1) {
        // Only run the first call
	// Get device properties
	int device;
        CHECK_CUDA(cudaGetDevice(&device));

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        // Get a good launch config based on occupancy
        int minGridSize = 0;
        CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
            &minGridSize,
            &blockSize,
            add_kernel<float, uint64_t>,
            0           // dynamic shared memory per block
        ));
    }
    int gridSize = (n + blockSize - 1) / blockSize;

    add_kernel<float, uint64_t><<<gridSize, blockSize>>>(result, x, y, n);

    CHECK_CUDA( cudaGetLastError() );
}

