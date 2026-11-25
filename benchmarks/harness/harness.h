// Shared by the benchmark executable of every system, so that all of them
// read the same inputs and are timed the same way.
//
//   <exe> <dir> [options]
//
// <dir>/A.bin: m, n, nnz, then the pos, crd and val arrays of A in CSC
// <dir>/x.bin: number of vectors, then nnz, crd and val of each vector
// <dir>/z.bin (written): status, nnz, crd and val of each result
// stdout: one line per vector, "RESULT <index> <status> <median ms>"
//
// The time of a vector is the time of one call that allocates the vectors
// the system writes, computes z, has nnz(z) on the host, and frees the
// vectors again. It is the median of RUNS calls, after one warmup call.
#pragma once

#include <cuda_runtime.h>
#include <unistd.h>

#include "pool.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace harness {

// Timed calls per vector
const int RUNS = 5;

// Waits for the GPU, so that a difference of two times covers all GPU work
inline double now_ms() {
    cudaDeviceSynchronize();
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

template <typename T>
std::vector<T> read_array(FILE* file, size_t n) {
    std::vector<T> values(n);
    if (fread(values.data(), sizeof(T), n, file) != n) {
        fprintf(stderr, "Failed to read input\n");
        _exit(2);
    }
    return values;
}

// Copies values to a new GPU buffer with room for capacity entries
template <typename T>
T* to_device(const std::vector<T>& values, size_t capacity) {
    T* ptr;
    // cudaMalloc returns a null pointer for 0 bytes
    if (cudaMalloc(&ptr, std::max<size_t>(capacity, 1) * sizeof(T)) !=
        cudaSuccess) {
        fprintf(stderr, "Failed to allocate GPU memory\n");
        _exit(3);
    }
    cudaMemcpy(ptr, values.data(), values.size() * sizeof(T),
               cudaMemcpyHostToDevice);
    return ptr;
}

template <typename T>
std::vector<T> to_host(const T* ptr, size_t n) {
    std::vector<T> values(n);
    cudaMemcpy(values.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost);
    return values;
}

// Allocates from the pool, for systems that take their output vectors from
// the caller
inline void* pool_alloc(size_t bytes) {
    void* ptr = nullptr;
    if (pool_malloc(&ptr, bytes) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate GPU memory\n");
        _exit(3);
    }
    return ptr;
}

// A in CSC on the host
struct Matrix {
    int m, n, nnz;
    std::vector<int> pos, crd;
    std::vector<float> val;
};

inline Matrix read_matrix(const std::string& dir) {
    FILE* file = fopen((dir + "/A.bin").c_str(), "rb");
    if (!file) {
        fprintf(stderr, "Failed to open %s/A.bin\n", dir.c_str());
        _exit(2);
    }
    const std::vector<int> shape = read_array<int>(file, 3);
    Matrix A{shape[0], shape[1], shape[2]};
    A.pos = read_array<int>(file, A.n + 1);
    A.crd = read_array<int>(file, A.nnz);
    A.val = read_array<float>(file, A.nnz);
    fclose(file);
    return A;
}

// z on the host
struct Result {
    int status = 0;
    std::vector<int> crd;
    std::vector<float> val;
};

// Times a system on every vector in <dir>/x.bin.
//
//   set_x(crd, val, nnz)  x is in these GPU buffers, which have room for
//                         n + 1 entries. Not timed.
//   run()                 The timed call. Returns a status, 0 for success.
//   result()              The same computation, returns z. Not timed.
//
// A system that cannot take an empty x gets status 1 for it.
template <typename SetX, typename Run, typename GetResult>
void benchmark(const std::string& dir, int n, bool takes_empty_x, SetX set_x,
               Run run, GetResult result) {
    int* x_crd = to_device(std::vector<int>(), n + 1);
    float* x_val = to_device(std::vector<float>(), n + 1);

    FILE* x_file = fopen((dir + "/x.bin").c_str(), "rb");
    FILE* z_file = fopen((dir + "/z.bin").c_str(), "wb");
    const int n_vectors = read_array<int>(x_file, 1)[0];
    for (int k = 0; k < n_vectors; k++) {
        const int x_nnz = read_array<int>(x_file, 1)[0];
        const std::vector<int> crd = read_array<int>(x_file, x_nnz);
        const std::vector<float> val = read_array<float>(x_file, x_nnz);
        cudaMemcpy(x_crd, crd.data(), x_nnz * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(x_val, val.data(), x_nnz * sizeof(float),
                   cudaMemcpyHostToDevice);
        set_x(x_crd, x_val, x_nnz);

        Result z;
        double median = 0;
        if (x_nnz == 0 && !takes_empty_x) {
            z.status = 1;
        } else {
            auto time_run = [&]() {
                const double t0 = now_ms();
                int status = run();
                const double ms = now_ms() - t0;
                if (cudaGetLastError() != cudaSuccess) {
                    status = 1;
                }
                return std::make_pair(status, ms);
            };

            int status = time_run().first;
            std::vector<double> times;
            for (int i = 0; i < RUNS; i++) {
                const auto timed = time_run();
                status = std::max(status, timed.first);
                times.push_back(timed.second);
            }
            std::sort(times.begin(), times.end());
            median = times[times.size() / 2];

            z = result();
            z.status = std::max(z.status, status);
        }
        printf("RESULT %d %d %.6f\n", k, z.status, median);
        fflush(stdout);

        const int z_nnz = z.crd.size();
        fwrite(&z.status, sizeof(int), 1, z_file);
        fwrite(&z_nnz, sizeof(int), 1, z_file);
        fwrite(z.crd.data(), sizeof(int), z_nnz, z_file);
        fwrite(z.val.data(), sizeof(float), z_nnz, z_file);
        // A later vector may crash the system
        fflush(z_file);
    }
    fclose(z_file);
}

}  // namespace harness
