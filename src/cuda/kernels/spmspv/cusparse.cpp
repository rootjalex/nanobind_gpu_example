#include <cusparse.h>

#include "kernels/pool.h"
#include "kernels/utils.h"
#include "spmspv.h"

#define CHECK_CUSPARSE(func)                                     \
    do {                                                         \
        const cusparseStatus_t check_cusparse_status_ = (func);  \
        if (check_cusparse_status_ != CUSPARSE_STATUS_SUCCESS) { \
            throw std::runtime_error(                            \
                cusparseGetErrorString(check_cusparse_status_)); \
        }                                                        \
    } while (false)

namespace {

// Destroys the cuSPARSE objects of one call
struct Descriptors {
    cusparseConstSpMatDescr_t xT = nullptr;
    cusparseConstSpMatDescr_t AT = nullptr;
    cusparseSpMatDescr_t zT = nullptr;
    cusparseSpGEMMDescr_t spgemm = nullptr;

    ~Descriptors() {
        cusparseDestroySpMat(xT);
        cusparseDestroySpMat(AT);
        cusparseDestroySpMat(zT);
        cusparseSpGEMM_destroyDescr(spgemm);
    }
};

}  // namespace

SparseVectorOut spmspv_cusparse(const CSCMatrix& A, const SparseVector& x) {
    if (x.nnz == 0 || A.nnz == 0) {
        return {};
    }

    static cusparseHandle_t handle = nullptr;
    if (!handle) {
        CHECK_CUSPARSE(cusparseCreate(&handle));
        CHECK_CUSPARSE(cusparseSetStream(handle, DEFAULT_STREAM));
    }

    // SpGEMM needs CSR, so compute z^T = x^T A^T.
    // A in CSC is A^T in CSR, and x^T is a CSR matrix with one row.
    const int32_t x_pos_host[2] = {0, x.nnz};
    Buffer<int32_t> x_pos(2), z_pos(2);
    CHECK_CUDA(cudaMemcpyAsync(x_pos.ptr, x_pos_host, sizeof(x_pos_host),
                               cudaMemcpyHostToDevice, DEFAULT_STREAM));

    Descriptors d;
    CHECK_CUSPARSE(cusparseCreateConstCsr(
        &d.xT, 1, A.n, x.nnz, x_pos.ptr, x.crd, x.val, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateConstCsr(
        &d.AT, A.n, A.m, A.nnz, A.pos, A.crd, A.val, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateCsr(
        &d.zT, 1, A.m, 0, z_pos.ptr, nullptr, nullptr, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&d.spgemm));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_DEFAULT;

    size_t size1 = 0;
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
        handle, op, op, &alpha, d.xT, d.AT, &beta, d.zT, CUDA_R_32F, alg,
        d.spgemm, &size1, nullptr));
    Buffer<char> buffer1(size1);
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(
        handle, op, op, &alpha, d.xT, d.AT, &beta, d.zT, CUDA_R_32F, alg,
        d.spgemm, &size1, buffer1.ptr));

    size_t size2 = 0;
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, op, op, &alpha, d.xT, d.AT,
                                          &beta, d.zT, CUDA_R_32F, alg,
                                          d.spgemm, &size2, nullptr));
    Buffer<char> buffer2(size2);
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, op, op, &alpha, d.xT, d.AT,
                                          &beta, d.zT, CUDA_R_32F, alg,
                                          d.spgemm, &size2, buffer2.ptr));

    int64_t rows, cols, z_nnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(d.zT, &rows, &cols, &z_nnz));
    Buffer<int32_t> z_crd(z_nnz);
    Buffer<float> z_val(z_nnz);
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(d.zT, z_pos.ptr, z_crd.ptr, z_val.ptr));
    CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, op, op, &alpha, d.xT, d.AT,
                                       &beta, d.zT, CUDA_R_32F, alg, d.spgemm));

    return {z_crd.release(), z_val.release(), static_cast<int32_t>(z_nnz)};
}
