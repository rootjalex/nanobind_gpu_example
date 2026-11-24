#include "kernels/spmspv/spmspv.h"

#include <nanobind/stl/pair.h>

#include <stdexcept>
#include <utility>

#include "backend.h"
#include "kernels/pool.h"

using namespace nb::literals;

namespace {

using SparseResult = std::pair<GPUVector<int32_t>, GPUVector<float>>;

CSCMatrix make_csc(const GPUVector<int32_t>& pos, const GPUVector<int32_t>& crd,
                   const GPUVector<float>& val, int32_t m) {
    if (pos.shape(0) == 0) {
        throw std::invalid_argument("A_pos must have n + 1 entries");
    }
    if (crd.shape(0) != val.shape(0)) {
        throw std::invalid_argument("A_crd and A_val have different lengths");
    }
    if (m < 0) {
        throw std::invalid_argument("m is negative");
    }
    const int32_t n = static_cast<int32_t>(pos.shape(0)) - 1;
    const int32_t nnz = static_cast<int32_t>(crd.shape(0));
    return {pos.data(), crd.data(), val.data(), m, n, nnz};
}

SparseVector make_sparse_vector(const GPUVector<int32_t>& crd,
                                const GPUVector<float>& val) {
    if (crd.shape(0) != val.shape(0)) {
        throw std::invalid_argument("x_crd and x_val have different lengths");
    }
    return {crd.data(), val.data(), static_cast<int32_t>(crd.shape(0))};
}

SparseResult make_result(const SparseVectorOut& z) {
    return {make_gpu_vector(z.crd, z.nnz), make_gpu_vector(z.val, z.nnz)};
}

const char* doc =
    "Returns (z_crd, z_val) for z = A x, with A (m rows) in CSC and x sparse";

// Binds a variant with the signature (A, x) -> z
void bind_variant(nb::module_& m, const char* name,
                  SparseVectorOut (*variant)(const CSCMatrix&,
                                             const SparseVector&)) {
    m.def(
        name,
        [variant](
            const GPUVector<int32_t>& A_pos, const GPUVector<int32_t>& A_crd,
            const GPUVector<float>& A_val, const GPUVector<int32_t>& x_crd,
            const GPUVector<float>& x_val, int32_t m) {
            return make_result(variant(make_csc(A_pos, A_crd, A_val, m),
                                       make_sparse_vector(x_crd, x_val)));
        },
        "A_pos"_a.noconvert(), "A_crd"_a.noconvert(), "A_val"_a.noconvert(),
        "x_crd"_a.noconvert(), "x_val"_a.noconvert(), "m"_a, doc);
}

}  // namespace

void bind_spmspv(nb::module_& m) {
    bind_variant(m, "spmspv_dense", spmspv_dense);
    bind_variant(m, "spmspv_hash", spmspv_hash);
    bind_variant(m, "spmspv_cusparse", spmspv_cusparse);

    m.def(
        "spmspv_esc",
        [](const GPUVector<int32_t>& A_pos, const GPUVector<int32_t>& A_crd,
           const GPUVector<float>& A_val, const GPUVector<int32_t>& x_crd,
           const GPUVector<float>& x_val, int32_t m, bool min_bits) {
            return make_result(spmspv_esc(make_csc(A_pos, A_crd, A_val, m),
                                          make_sparse_vector(x_crd, x_val),
                                          min_bits));
        },
        "A_pos"_a.noconvert(), "A_crd"_a.noconvert(), "A_val"_a.noconvert(),
        "x_crd"_a.noconvert(), "x_val"_a.noconvert(), "m"_a,
        "min_bits"_a = false, doc);

    m.def("free_unused_memory", gpu_free_unused,
          "Returns GPU buffers cached by the extension to CUDA");
}
