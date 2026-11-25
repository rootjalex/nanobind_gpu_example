// Stands in for helper_cuda.h of the CUDA samples, which the toolkit no
// longer ships. spmspv.h only uses checkCudaErrors from it.
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define checkCudaErrors(call)                                          \
    do {                                                               \
        const cudaError_t lay_compat_error = (call);                   \
        if (lay_compat_error != cudaSuccess) {                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,     \
                    __LINE__, cudaGetErrorString(lay_compat_error));   \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)
