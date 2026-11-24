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

// Blocks until all GPU work submitted so far has finished, and reports any
// error raised by that work.
void synchronize();
