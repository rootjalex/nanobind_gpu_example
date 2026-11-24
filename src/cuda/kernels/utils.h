#pragma once

#include <cuda_runtime.h>

#include <stdexcept>

#define CHECK_CUDA(func)                                                      \
    do {                                                                      \
        const cudaError_t check_cuda_status_ = (func);                        \
        if (check_cuda_status_ != cudaSuccess) {                              \
            throw std::runtime_error(cudaGetErrorString(check_cuda_status_)); \
        }                                                                     \
    } while (false)

// All GPU work is submitted to the default stream
const cudaStream_t DEFAULT_STREAM = nullptr;

// Waits for all GPU work to finish
void synchronize();

// Copies one value from the GPU. Blocks until the stream reaches the copy.
template <typename T>
T read_scalar(const T* ptr) {
    T value;
    CHECK_CUDA(cudaMemcpyAsync(&value, ptr, sizeof(T), cudaMemcpyDeviceToHost,
                               DEFAULT_STREAM));
    CHECK_CUDA(cudaStreamSynchronize(DEFAULT_STREAM));
    return value;
}
