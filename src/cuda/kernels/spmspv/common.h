#pragma once

#include <climits>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda/std/algorithm>
#include <cuda/std/bit>

#include "kernels/pool.h"
#include "kernels/utils.h"
#include "spmspv.h"

const uint32_t BLOCK_SIZE = 256;

// Number of blocks to launch n threads
inline uint32_t num_blocks(int32_t n) {
    // Thread indices are int32_t
    if (n > INT32_MAX - static_cast<int32_t>(BLOCK_SIZE)) {
        throw std::length_error("Failed to launch kernel, too many threads");
    }
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// Calls a cub algorithm twice, first to get the temporary storage size
template <typename F>
void run_cub(F&& f) {
    size_t bytes = 0;
    CHECK_CUDA(f(nullptr, bytes));
    Buffer<char> temp(bytes);
    CHECK_CUDA(f(temp.ptr, bytes));
}

// Fills T (nnz(x) + 1 entries) with the offset of each column's products.
// Returns the total number of products n, which is also T[nnz(x)].
int32_t partition(const CSCMatrix& A, const SparseVector& x, int32_t* T);

struct Product {
    int32_t i;
    float v;
};

// Row and value of the t-th product
__device__ inline Product get_product(const CSCMatrix& A, const SparseVector& x,
                                      const int32_t* T, int32_t t) {
    const int32_t q = cuda::std::upper_bound(T, T + x.nnz, t) - T - 1;
    const int32_t p = A.pos[x.crd[q]] + t - T[q];
    return {A.crd[p], A.val[p] * x.val[q]};
}

// Thread t computes one product and scatters it
template <typename Scatter>
__global__ void execute_kernel(CSCMatrix A, SparseVector x, const int32_t* T,
                               int32_t n, Scatter scatter) {
    const int32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) {
        return;
    }
    const Product product = get_product(A, x, T, t);
    scatter(t, product.i, product.v);
}

template <typename Scatter>
void execute(const CSCMatrix& A, const SparseVector& x, const int32_t* T,
             int32_t n, Scatter scatter) {
    if (n == 0) {
        return;
    }
    execute_kernel<Scatter>
        <<<num_blocks(n), BLOCK_SIZE, 0, DEFAULT_STREAM>>>(A, x, T, n, scatter);
    CHECK_CUDA(cudaGetLastError());
}

// Sorts n (key, val) pairs on the bits [0, end_bit) of the keys.
// Returns true if the sorted pairs are in keys_alt and vals_alt.
inline bool sort_pairs(uint32_t* keys, uint32_t* keys_alt, float* vals,
                       float* vals_alt, int32_t n, int end_bit) {
    cub::DoubleBuffer<uint32_t> d_keys(keys, keys_alt);
    cub::DoubleBuffer<float> d_vals(vals, vals_alt);
    run_cub([&](void* temp, size_t& bytes) {
        return cub::DeviceRadixSort::SortPairs(temp, bytes, d_keys, d_vals, n,
                                               0, end_bit, DEFAULT_STREAM);
    });
    return d_keys.Current() == keys_alt;
}

// Number of bits in a row index of a matrix with m rows
inline int row_bits(int32_t m) {
    return cuda::std::bit_width(static_cast<uint32_t>(m - 1));
}
