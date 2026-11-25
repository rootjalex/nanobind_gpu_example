// Times GraphBLAST's SpMSpV (vxm with a sparse vector) on inputs written by
// benchmarks/spmspv.py, see ../harness.h.
//
//   grb_spmspv <dir> [GraphBLAST options]

#define GRB_USE_CUDA
#include "../harness.h"

// GraphBLAST's examples and tests all define this before including it
#define private public
#include "graphblas/graphblas.hpp"

int main(int argc, char** argv) {
    const std::string dir = argv[1];
    const harness::Matrix host = harness::read_matrix(dir);
    const int m = host.m, n = host.n, nnz = host.nnz;

    // GraphBLAST sorts on the bits needed for n rows, which is wrong for m > n
    std::vector<char*> options = {argv[0], const_cast<char*>("--endbit"),
                                  const_cast<char*>(m <= n ? "1" : "0")};
    // Skip <dir>, the rest are GraphBLAST options
    options.insert(options.end(), argv + 2, argv + argc);
    po::variables_map vm;
    parseArgs(options.size(), options.data(), &vm);

    int* pos = harness::to_device(host.pos, n + 1);
    int* crd = harness::to_device(host.crd, nnz);
    float* val = harness::to_device(host.val, nnz);

    // A in CSC is A^T in CSR, and vxm(x, A^T) = A x
    graphblas::Matrix<float> At(n, m);
    At.build(pos, crd, val, nnz);
    graphblas::Vector<float> x(n);
    graphblas::Descriptor desc;
    desc.loadArgs(vm);
    // Keep x sparse
    desc.set(graphblas::GrB_MXVMODE, graphblas::GrB_PUSHONLY);

    // z is a new vector in every call, which allocates it
    auto compute = [&](graphblas::Vector<float>& z) {
        return static_cast<int>(graphblas::vxm<float, float, float, float>(
            &z, GrB_NULL, GrB_NULL, graphblas::PlusMultipliesSemiring<float>(),
            &x, &At, &desc));
    };

    harness::benchmark(
        dir, n, /* takes_empty_x = */ true,
        [&](int* x_crd, float* x_val, int x_nnz) {
            x.build(x_crd, x_val, x_nnz);
        },
        [&]() {
            graphblas::Vector<float> z(m);
            const int status = compute(z);
            // nnz(z) is on the host
            graphblas::Index z_nnz = 0;
            z.nvals(&z_nnz);
            return status;
        },
        [&]() {
            graphblas::Vector<float> z(m);
            harness::Result result;
            result.status = compute(z);
            graphblas::Index z_nnz = 0;
            z.nvals(&z_nnz);
            z.extractTuples(&result.crd, &result.val, &z_nnz);
            result.crd.resize(z_nnz);
            result.val.resize(z_nnz);
            return result;
        });

    // Skip the destructors, GraphBLAST frees the buffers of x twice
    _exit(0);
}
