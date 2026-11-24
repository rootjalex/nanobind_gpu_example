#include <algorithm>
#include <cub/device/device_reduce.cuh>
#include <cuda/std/functional>

#include "common.h"

namespace {

// Thread t writes its product to slot t
struct EscScatter {
    uint32_t* keys;
    float* vals;

    __device__ void operator()(int32_t t, int32_t i, float v) const {
        keys[t] = i;
        vals[t] = v;
    }
};

}  // namespace

SparseVectorOut spmspv_esc(const CSCMatrix& A, const SparseVector& x,
                           bool min_bits) {
    Buffer<int32_t> T(x.nnz + 1);
    const int32_t n = partition(A, x, T.ptr);
    if (n == 0) {
        return {};
    }

    // Expand. The radix sort needs a second buffer for keys and vals.
    Buffer<uint32_t> keys(n), keys_alt(n);
    Buffer<float> vals(n), vals_alt(n);
    execute(A, x, T.ptr, n, EscScatter{keys.ptr, vals.ptr});

    // Sort
    const bool in_alt =
        sort_pairs(keys.ptr, keys_alt.ptr, vals.ptr, vals_alt.ptr, n,
                   min_bits ? row_bits(A.m) : 32);
    const uint32_t* sorted_keys = in_alt ? keys_alt.ptr : keys.ptr;
    const float* sorted_vals = in_alt ? vals_alt.ptr : vals.ptr;

    // Compress
    const int32_t max_nnz = std::min(n, A.m);
    Buffer<int32_t> z_crd(max_nnz);
    Buffer<float> z_val(max_nnz);
    Buffer<int32_t> z_nnz(1);
    run_cub([&](void* temp, size_t& bytes) {
        return cub::DeviceReduce::ReduceByKey(
            temp, bytes, sorted_keys, reinterpret_cast<uint32_t*>(z_crd.ptr),
            sorted_vals, z_val.ptr, z_nnz.ptr, ::cuda::std::plus<>{}, n,
            DEFAULT_STREAM);
    });

    return {z_crd.release(), z_val.release(), read_scalar(z_nnz.ptr)};
}
