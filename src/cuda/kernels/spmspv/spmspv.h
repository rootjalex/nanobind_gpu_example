#pragma once

#include <cstdint>

// m x n matrix in CSC
struct CSCMatrix {
    const int32_t* pos;
    const int32_t* crd;
    const float* val;
    int32_t m;
    int32_t n;
    int32_t nnz;
};

struct SparseVector {
    const int32_t* crd;
    const float* val;
    int32_t nnz;
};

// crd and val are allocated with gpu_alloc, crd is sorted.
// An empty vector has no buffers.
struct SparseVectorOut {
    int32_t* crd = nullptr;
    float* val = nullptr;
    int32_t nnz = 0;
};

// z = A x, with a different scatter strategy for the products

// Atomic adds into a dense vector, then compresses the dense vector
SparseVectorOut spmspv_dense(const CSCMatrix& A, const SparseVector& x);

// Expand, sort, compress. min_bits only sorts on the bits needed for m rows
SparseVectorOut spmspv_esc(const CSCMatrix& A, const SparseVector& x,
                           bool min_bits);

// Hash tables in shared memory, flushed to a global hash table. The global
// table is then compressed and sorted.
SparseVectorOut spmspv_hash(const CSCMatrix& A, const SparseVector& x);

// cuSPARSE SpGEMM, with x as a matrix with one row
SparseVectorOut spmspv_cusparse(const CSCMatrix& A, const SparseVector& x);
