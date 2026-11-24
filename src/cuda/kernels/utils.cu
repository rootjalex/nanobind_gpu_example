#include "utils.h"

void synchronize() {
    // cudaDeviceSynchronize reports kernel errors
    CHECK_CUDA(cudaDeviceSynchronize());
}
