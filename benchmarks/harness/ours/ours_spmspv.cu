// Times our SpMSpV variants on inputs written by benchmarks/spmspv.py, see
// ../harness.h.
//
//   ours_spmspv <dir> --variant dense|esc|esc_min_bits|hash|cusparse

#include <cstring>
#include <functional>

#include "../harness.h"
#include "kernels/pool.h"
#include "kernels/spmspv/spmspv.h"

int main(int argc, char** argv) {
    if (argc != 4 || strcmp(argv[2], "--variant") != 0) {
        fprintf(stderr, "Usage: %s <dir> --variant <name>\n", argv[0]);
        return 1;
    }
    const std::string name = argv[3];
    std::function<SparseVectorOut(const CSCMatrix&, const SparseVector&)>
        variant;
    if (name == "dense") {
        variant = spmspv_dense;
    } else if (name == "esc" || name == "esc_min_bits") {
        const bool min_bits = name == "esc_min_bits";
        variant = [min_bits](const CSCMatrix& A, const SparseVector& x) {
            return spmspv_esc(A, x, min_bits);
        };
    } else if (name == "hash") {
        variant = spmspv_hash;
    } else if (name == "cusparse") {
        variant = spmspv_cusparse;
    } else {
        fprintf(stderr, "Unknown variant %s\n", name.c_str());
        return 1;
    }

    const harness::Matrix host = harness::read_matrix(argv[1]);
    const CSCMatrix A{harness::to_device(host.pos, host.n + 1),
                      harness::to_device(host.crd, host.nnz),
                      harness::to_device(host.val, host.nnz),
                      host.m,
                      host.n,
                      host.nnz};
    SparseVector x{};

    // Out of GPU memory and too many products are exceptions
    auto compute = [&](SparseVectorOut& z) {
        try {
            z = variant(A, x);
            return 0;
        } catch (const std::exception&) {
            gpu_free_unused();
            return 1;
        }
    };

    harness::benchmark(
        argv[1], host.n, /* takes_empty_x = */ true,
        [&](const int* crd, const float* val, int nnz) { x = {crd, val, nnz}; },
        [&]() {
            SparseVectorOut z;
            const int status = compute(z);
            gpu_free(z.crd);
            gpu_free(z.val);
            return status;
        },
        [&]() {
            SparseVectorOut z;
            harness::Result result;
            result.status = compute(z);
            result.crd = harness::to_host(z.crd, z.nnz);
            result.val = harness::to_host(z.val, z.nnz);
            gpu_free(z.crd);
            gpu_free(z.val);
            return result;
        });
    return 0;
}
