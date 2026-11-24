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

// Waits for all GPU work to finish
void synchronize();
