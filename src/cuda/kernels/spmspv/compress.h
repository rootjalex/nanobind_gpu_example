#pragma once

#include "common.h"

// Count, sum, emit compression of `size` slots into a sparse vector.
// The nonzeros keep the order of their slots.
//
// A Slots type says which slots hold a nonzero:
//     bool nonzero(int32_t slot)
//     int32_t crd(int32_t slot)
//     float val(int32_t slot)

// warpSize for host code. It is 32 on every NVIDIA GPU.
const int32_t WARP_SIZE = 32;

// Count: each warp writes the mask and the count of its nonzeros
template <typename Slots>
__global__ void count_kernel(Slots slots, int32_t size, uint32_t* masks,
                             int32_t* counts) {
    const int32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t mask =
        __ballot_sync(0xffffffff, slot < size && slots.nonzero(slot));
    if (slot < size && slot % warpSize == 0) {
        masks[slot / warpSize] = mask;
        counts[slot / warpSize] = cuda::std::popcount(mask);
    }
}

// Emit: thread o writes the o-th nonzero
template <typename Slots>
__global__ void emit_kernel(Slots slots, const uint32_t* masks,
                            const int32_t* offsets, int32_t n_warps,
                            int32_t z_nnz, int32_t* z_crd, float* z_val) {
    const int32_t o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= z_nnz) {
        return;
    }

    const int32_t warp =
        cuda::std::upper_bound(offsets, offsets + n_warps, o) - offsets - 1;
    // Lane of the (o - offsets[warp])-th nonzero in the warp
    const int32_t lane = __fns(masks[warp], 0, o - offsets[warp] + 1);
    const int32_t slot = warp * warpSize + lane;
    z_crd[o] = slots.crd(slot);
    z_val[o] = slots.val(slot);
}

template <typename Slots>
SparseVectorOut compress(Slots slots, int32_t size) {
    const int32_t n_warps = (size + WARP_SIZE - 1) / WARP_SIZE;

    // Count
    Buffer<uint32_t> masks(n_warps);
    Buffer<int32_t> counts(n_warps + 1);
    count_kernel<Slots><<<num_blocks(size), BLOCK_SIZE, 0, DEFAULT_STREAM>>>(
        slots, size, masks.ptr, counts.ptr);
    CHECK_CUDA(cudaGetLastError());
    // The last count is 0, so the last offset is the total
    CHECK_CUDA(cudaMemsetAsync(counts.ptr + n_warps, 0, sizeof(int32_t),
                               DEFAULT_STREAM));

    // Sum
    Buffer<int32_t> offsets(n_warps + 1);
    run_cub([&](void* temp, size_t& bytes) {
        return cub::DeviceScan::ExclusiveSum(
            temp, bytes, counts.ptr, offsets.ptr, n_warps + 1, DEFAULT_STREAM);
    });
    const int32_t z_nnz = read_scalar(offsets.ptr + n_warps);
    if (z_nnz == 0) {
        return {};
    }

    // Emit
    Buffer<int32_t> z_crd(z_nnz);
    Buffer<float> z_val(z_nnz);
    emit_kernel<Slots><<<num_blocks(z_nnz), BLOCK_SIZE, 0, DEFAULT_STREAM>>>(
        slots, masks.ptr, offsets.ptr, n_warps, z_nnz, z_crd.ptr, z_val.ptr);
    CHECK_CUDA(cudaGetLastError());

    return {z_crd.release(), z_val.release(), z_nnz};
}
