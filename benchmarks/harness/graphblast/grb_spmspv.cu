// Times GraphBLAST's SpMSpV (vxm with a sparse vector) on inputs written by
// benchmarks/spmspv.py.
//
//   grb_spmspv <dir> [GraphBLAST options]
//
// <dir>/A.bin: m, n, nnz, then the pos, crd and val arrays of A in CSC
// <dir>/x.bin: number of vectors, then nnz, crd and val of each vector
// <dir>/z.bin (written): status, nnz, crd and val of each result
// stdout: one line per vector, "RESULT <index> <status> <median ms>"

#define GRB_USE_CUDA
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

// GraphBLAST's examples and tests all define this before including it
#define private public
#include "graphblas/graphblas.hpp"

static const int ITERS = 10;

static double now_ms() {
    cudaDeviceSynchronize();
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

template <typename T>
static std::vector<T> read_array(FILE* file, size_t n) {
    std::vector<T> values(n);
    if (fread(values.data(), sizeof(T), n, file) != n) {
        fprintf(stderr, "Failed to read input\n");
        _exit(2);
    }
    return values;
}

template <typename T>
static T* to_device(const std::vector<T>& values, size_t capacity) {
    T* ptr;
    if (cudaMalloc(&ptr, capacity * sizeof(T)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate GPU memory\n");
        _exit(3);
    }
    cudaMemcpy(ptr, values.data(), values.size() * sizeof(T),
               cudaMemcpyHostToDevice);
    return ptr;
}

int main(int argc, char** argv) {
    const std::string dir = argv[1];
    po::variables_map vm;
    // Skip <dir>, the rest are GraphBLAST options
    parseArgs(argc - 1, argv + 1, &vm);

    FILE* A_file = fopen((dir + "/A.bin").c_str(), "rb");
    const std::vector<int> shape = read_array<int>(A_file, 3);
    const int m = shape[0], n = shape[1], nnz = shape[2];
    int* pos = to_device(read_array<int>(A_file, n + 1), n + 1);
    int* crd = to_device(read_array<int>(A_file, nnz), nnz);
    float* val = to_device(read_array<float>(A_file, nnz), nnz);
    fclose(A_file);

    // A in CSC is A^T in CSR, and vxm(x, A^T) = A x
    graphblas::Matrix<float> At(n, m);
    At.build(pos, crd, val, nnz);
    graphblas::Vector<float> x(n), w(m);
    graphblas::Descriptor desc;
    desc.loadArgs(vm);
    // Keep x sparse
    desc.set(graphblas::GrB_MXVMODE, graphblas::GrB_PUSHONLY);

    // GraphBLAST's sparse vectors have room for n entries
    int* x_crd = to_device(std::vector<int>(), n);
    float* x_val = to_device(std::vector<float>(), n + 1);

    FILE* x_file = fopen((dir + "/x.bin").c_str(), "rb");
    FILE* z_file = fopen((dir + "/z.bin").c_str(), "wb");
    const int n_vectors = read_array<int>(x_file, 1)[0];
    for (int k = 0; k < n_vectors; k++) {
        const int x_nnz = read_array<int>(x_file, 1)[0];
        const std::vector<int> h_crd = read_array<int>(x_file, x_nnz);
        const std::vector<float> h_val = read_array<float>(x_file, x_nnz);
        cudaMemcpy(x_crd, h_crd.data(), x_nnz * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(x_val, h_val.data(), x_nnz * sizeof(float),
                   cudaMemcpyHostToDevice);
        x.build(x_crd, x_val, x_nnz);

        auto run = [&]() {
            const double t0 = now_ms();
            const graphblas::Info info =
                graphblas::vxm<float, float, float, float>(
                    &w, GrB_NULL, GrB_NULL,
                    graphblas::PlusMultipliesSemiring<float>(), &x, &At, &desc);
            const double ms = now_ms() - t0;
            return std::make_pair(static_cast<int>(info), ms);
        };

        // Slow calls get fewer iterations
        const auto first = run();
        const bool slow = first.second >= 50;
        for (int i = 0; i < (slow ? 0 : 3); i++) {
            run();
        }
        std::vector<double> times;
        int status = first.first;
        for (int i = 0; i < (slow ? 3 : ITERS); i++) {
            const auto result = run();
            status = std::max(status, result.first);
            times.push_back(result.second);
        }
        std::sort(times.begin(), times.end());
        printf("RESULT %d %d %.6f\n", k, status, times[times.size() / 2]);
        fflush(stdout);

        graphblas::Index z_nnz = 0;
        std::vector<graphblas::Index> z_crd;
        std::vector<float> z_val;
        w.nvals(&z_nnz);
        w.extractTuples(&z_crd, &z_val, &z_nnz);
        fwrite(&status, sizeof(int), 1, z_file);
        fwrite(&z_nnz, sizeof(int), 1, z_file);
        fwrite(z_crd.data(), sizeof(int), z_nnz, z_file);
        fwrite(z_val.data(), sizeof(float), z_nnz, z_file);
        // A later vector may crash GraphBLAST
        fflush(z_file);
    }
    fclose(z_file);

    // Skip the destructors, GraphBLAST frees the buffers of x twice
    _exit(0);
}
