#include <algorithm>
#include <cuda/functional>
#include <utility>

#include "common.h"
#include "compress.h"

namespace {

// One slot of a hash table. key is the row + 1, so zeroed memory is an
// empty table.
struct Slot {
    int32_t key;
    float val;
};

// Slots of each table in shared memory
const int32_t SMEM_SLOTS = 1024;
// Each block adds this many products, so its table is at most half full
const int32_t BLOCK_PRODUCTS = SMEM_SLOTS / 2;

using Hash = cuda::hash<int32_t, cuda::hash_algorithm::xxhash_32>;

// Adds v to the value of row i
__device__ void table_add(Slot* table, uint32_t n_slots, int32_t i, float v) {
    uint32_t slot = Hash{}(i) % n_slots;
    while (true) {
        int32_t key = table[slot].key;
        if (key == 0) {
            key = atomicCAS(&table[slot].key, 0, i + 1);
        }
        if (key == 0 || key == i + 1) {
            atomicAdd(&table[slot].val, v);
            return;
        }
        slot = slot + 1 == n_slots ? 0 : slot + 1;
    }
}

// Each block adds its products to a table in shared memory, then flushes
// the table to the global table
__global__ void hash_kernel(CSCMatrix A, SparseVector x, const int32_t* T,
                            int32_t n, Slot* table, uint32_t n_slots) {
    __shared__ Slot smem[SMEM_SLOTS];

    for (int32_t slot = threadIdx.x; slot < SMEM_SLOTS; slot += blockDim.x) {
        smem[slot] = {0, 0};
    }
    __syncthreads();

    const int32_t begin = blockIdx.x * BLOCK_PRODUCTS;
    const int32_t end = min(begin + BLOCK_PRODUCTS, n);
    for (int32_t t = begin + threadIdx.x; t < end; t += blockDim.x) {
        const Product product = get_product(A, x, T, t);
        table_add(smem, SMEM_SLOTS, product.i, product.v);
    }
    __syncthreads();

    for (int32_t slot = threadIdx.x; slot < SMEM_SLOTS; slot += blockDim.x) {
        if (smem[slot].key != 0) {
            table_add(table, n_slots, smem[slot].key - 1, smem[slot].val);
        }
    }
}

// Every used slot of the global table holds one row
struct TableSlots {
    const Slot* table;

    __device__ bool nonzero(int32_t slot) const { return table[slot].key != 0; }
    __device__ int32_t crd(int32_t slot) const { return table[slot].key - 1; }
    __device__ float val(int32_t slot) const { return table[slot].val; }
};

}  // namespace

SparseVectorOut spmspv_hash(const CSCMatrix& A, const SparseVector& x) {
    Buffer<int32_t> T(x.nnz + 1);
    const int32_t n = partition(A, x, T.ptr);
    if (n == 0) {
        return {};
    }
    // Block and slot indices are int32_t
    if (n > INT32_MAX - BLOCK_PRODUCTS || std::min(n, A.m) > INT32_MAX / 2) {
        throw std::length_error("Failed to launch kernel, too many products");
    }

    // z has at most min(n, m) nonzeros, so the table is at most 2/3 full
    const uint32_t max_nnz = std::min(n, A.m);
    const uint32_t n_slots = max_nnz + (max_nnz + 1) / 2;
    Buffer<Slot> table(n_slots);
    CHECK_CUDA(
        cudaMemsetAsync(table.ptr, 0, n_slots * sizeof(Slot), DEFAULT_STREAM));

    const uint32_t n_blocks = (n + BLOCK_PRODUCTS - 1) / BLOCK_PRODUCTS;
    hash_kernel<<<n_blocks, BLOCK_SIZE, 0, DEFAULT_STREAM>>>(
        A, x, T.ptr, n, table.ptr, n_slots);
    CHECK_CUDA(cudaGetLastError());

    // Compress
    const SparseVectorOut z = compress(TableSlots{table.ptr}, n_slots);
    Buffer<int32_t> z_crd(z.crd);
    Buffer<float> z_val(z.val);

    // Sort
    Buffer<int32_t> crd_alt(z.nnz);
    Buffer<float> val_alt(z.nnz);
    const bool in_alt =
        sort_pairs(reinterpret_cast<uint32_t*>(z_crd.ptr),
                   reinterpret_cast<uint32_t*>(crd_alt.ptr), z_val.ptr,
                   val_alt.ptr, z.nnz, row_bits(A.m));
    if (in_alt) {
        std::swap(z_crd.ptr, crd_alt.ptr);
        std::swap(z_val.ptr, val_alt.ptr);
    }

    return {z_crd.release(), z_val.release(), z.nnz};
}
