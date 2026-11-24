#include "utils.h"

void synchronize() {
    // Errors from asynchronous kernel launches surface here.
    CHECK_CUDA(cudaDeviceSynchronize());
}
