#include <cuda_runtime.h>

void synchronize() {
    cudaDeviceSynchronize();
}
