#include "common.h"

namespace {

// Thread q writes the number of products for the q-th nonzero of x.
// The entry after the last nonzero is 0.
__global__ void column_lengths_kernel(const int32_t* pos, const int32_t* x_crd,
                                      int32_t x_nnz, int32_t* lengths) {
    const int32_t q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q > x_nnz) {
        return;
    }
    if (q == x_nnz) {
        lengths[q] = 0;
        return;
    }
    const int32_t j = x_crd[q];
    lengths[q] = pos[j + 1] - pos[j];
}

}  // namespace

int32_t partition(const CSCMatrix& A, const SparseVector& x, int32_t* T) {
    column_lengths_kernel<<<num_blocks(x.nnz + 1), BLOCK_SIZE, 0,
                            DEFAULT_STREAM>>>(A.pos, x.crd, x.nnz, T);
    CHECK_CUDA(cudaGetLastError());

    // The offsets are the prefix sum of the lengths, in place
    run_cub([&](void* temp, size_t& bytes) {
        return cub::DeviceScan::ExclusiveSum(temp, bytes, T, x.nnz + 1,
                                             DEFAULT_STREAM);
    });
    return read_scalar(T + x.nnz);
}
