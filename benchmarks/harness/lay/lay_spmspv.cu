// Times the column-based SpMSpV kernels of spmspv-adaptive (Li, Ao, Yang,
// "Adaptive SpMV/SpMSpV on GPUs for Input Vectors of Varied Sparsity") on
// inputs written by benchmarks/spmspv.py, see ../harness.h.
//
//   lay_spmspv <dir> [--kernel <name>]
//
// Kernels, named <write-back>-<workload distribution>:
//   atomic-merge (default), atomic-naive, sort-merge, sort-naive
//   atomic-my, sort-my: these return wrong results
// An empty vector gets status 1 and is not run.
//
// The kernels write into vectors of the caller, which are allocated from a
// CUDA memory pool in every call. The atomic kernels do not return nnz(z),
// so it is read from the prefix sum of their compaction.

#include <cstring>
#include <functional>

#include "../harness.h"

// spmspv.h needs this but does not include it
#include <cusparse.h>

// For the scratch buffer of the handle, which holds that prefix sum
#define private public
#include "spmspv/csc-spmspv/spmspv.h"
#undef private

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <dir> [--kernel <name>]\n", argv[0]);
        return 1;
    }
    const std::string dir = argv[1];
    std::string kernel = "atomic-merge";
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            kernel = argv[++i];
        } else {
            fprintf(stderr, "Unknown option %s\n", argv[i]);
            return 1;
        }
    }

    const harness::Matrix host = harness::read_matrix(dir);
    const int m = host.m, n = host.n, nnz = host.nnz;
    int* pos = harness::to_device(host.pos, n + 1);
    int* crd = harness::to_device(host.crd, nnz);
    float* val = harness::to_device(host.val, nnz);

    // Per-matrix setup, not timed. allocPreBuffer sizes the scratch buffer
    // from n and nnz only, but the atomic kernels compact y in it and need
    // 2 m ints, so allocate it as if A had at least 2 m / 3 entries.
    SpmspvHandle<int, unsigned int, float> A(m, n,
                                             std::max(nnz, 2 * (m / 3) + 1));
    A.allocPreBuffer();
    A.InputCSC(nnz, crd, pos, val);
    A.set_vector_type(0);

    // The atomic kernels accumulate into y and compact it into z. All three
    // have m entries.
    float* y = nullptr;
    int* z_crd = nullptr;
    float* z_val = nullptr;

    const float alpha = 1;
    int z_nnz = 0;
    // Each kernel fills z_crd and z_val, only the sort kernels set z_nnz
    std::function<int()> spmspv;
    bool sets_nnz = true;
    if (kernel == "sort-naive") {
        spmspv = [&]() {
            return A.CscBasedSortNaiveSpmspv(alpha, &z_nnz, z_crd, z_val, 0);
        };
    } else if (kernel == "sort-merge") {
        spmspv = [&]() {
            return A.CscBasedSortMergeSpmspv(false, alpha, &z_nnz, y, z_crd,
                                             z_val);
        };
    } else if (kernel == "sort-my") {
        spmspv = [&]() {
            return A.CscBasedSortMySpmspv(false, alpha, &z_nnz, y, z_crd,
                                          z_val);
        };
    } else if (kernel == "atomic-naive") {
        sets_nnz = false;
        spmspv = [&]() {
            return A.CscBasedNoSortNaiveSpmspv_keyval(alpha, &z_nnz, y, z_crd,
                                                      z_val);
        };
    } else if (kernel == "atomic-merge") {
        sets_nnz = false;
        spmspv = [&]() {
            return A.CscBasedNoSortMergeSpmspv_keyval(false, alpha, &z_nnz, y,
                                                      z_crd, z_val);
        };
    } else if (kernel == "atomic-my") {
        sets_nnz = false;
        spmspv = [&]() {
            return A.CscBasedMyNoSortSpmspv_keyval(false, alpha, &z_nnz, y,
                                                   z_crd, z_val);
        };
    } else {
        fprintf(stderr, "Unknown kernel %s\n", kernel.c_str());
        return 1;
    }

    // The compaction of the atomic kernels leaves a flag per row of y, and
    // then the exclusive prefix sum of the flags, in the scratch buffer
    auto flags = [&]() {
        return static_cast<const int*>(A.d_ptr_pre_alloc_buffer_);
    };

    // Allocates the vectors, runs the kernel, and has nnz(z) on the host
    auto compute = [&]() {
        // The sort kernels leave z_nnz alone if no column of A is hit
        z_nnz = 0;
        y = static_cast<float*>(harness::pool_alloc(m * sizeof(float)));
        z_crd = static_cast<int*>(harness::pool_alloc(m * sizeof(int)));
        z_val = static_cast<float*>(harness::pool_alloc(m * sizeof(float)));
        const int status = spmspv();
        if (!sets_nnz) {
            cudaMemcpyAsync(&z_nnz, flags() + 2 * m - 1, sizeof(int),
                            cudaMemcpyDeviceToHost, nullptr);
            cudaStreamSynchronize(nullptr);
        }
        return status;
    };
    auto free_vectors = [&]() {
        harness::pool_free(y);
        harness::pool_free(z_crd);
        harness::pool_free(z_val);
    };

    harness::benchmark(
        dir, n, /* takes_empty_x = */ false,
        [&](int* x_crd, float* x_val, int x_nnz) {
            A.set_sparse_x(x_crd, x_val, x_nnz);
        },
        [&]() {
            const int status = compute();
            free_vectors();
            return status;
        },
        [&]() {
            harness::Result result;
            result.status = compute();
            if (!sets_nnz) {
                // An inclusive prefix sum would not need the last flag
                int last_flag;
                cudaMemcpy(&last_flag, flags() + m - 1, sizeof(int),
                           cudaMemcpyDeviceToHost);
                z_nnz += last_flag;
            }
            result.crd = harness::to_host(z_crd, z_nnz);
            result.val = harness::to_host(z_val, z_nnz);
            free_vectors();
            return result;
        });

    // Skip the destructors, Destroy() would free the arrays of A
    _exit(0);
}
