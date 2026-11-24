#include "common.h"
#include "compress.h"

namespace {

struct DenseScatter {
    float* w;

    __device__ void operator()(int32_t t, int32_t i, float v) const {
        atomicAdd(&w[i], v);
    }
};

// Slot i of the dense vector holds row i
struct DenseSlots {
    const float* w;

    __device__ bool nonzero(int32_t slot) const { return w[slot] != 0; }
    __device__ int32_t crd(int32_t slot) const { return slot; }
    __device__ float val(int32_t slot) const { return w[slot]; }
};

}  // namespace

SparseVectorOut spmspv_dense(const CSCMatrix& A, const SparseVector& x) {
    Buffer<int32_t> T(x.nnz + 1);
    const int32_t n = partition(A, x, T.ptr);
    if (n == 0) {
        return {};
    }

    Buffer<float> w(A.m);
    CHECK_CUDA(cudaMemsetAsync(w.ptr, 0, A.m * sizeof(float), DEFAULT_STREAM));
    execute(A, x, T.ptr, n, DenseScatter{w.ptr});

    return compress(DenseSlots{w.ptr}, A.m);
}
