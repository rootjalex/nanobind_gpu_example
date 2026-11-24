import nanobind_gpu_example
import numpy as np
import pytest
import scipy.sparse
import torch

IMPLS = {
    "dense": nanobind_gpu_example.spmspv_dense,
    "esc": nanobind_gpu_example.spmspv_esc,
    "esc_min_bits": lambda *args: nanobind_gpu_example.spmspv_esc(*args, min_bits=True),
    "hash": nanobind_gpu_example.spmspv_hash,
    "cusparse": nanobind_gpu_example.spmspv_cusparse,
}


@pytest.fixture(scope="module", autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("SpMSpV is CUDA only")


def random_csc(m: int, n: int, density: float, seed: int) -> scipy.sparse.csc_matrix:
    rng = np.random.default_rng(seed)
    return scipy.sparse.random(
        m,
        n,
        density=density,
        format="csc",
        dtype=np.float32,
        random_state=rng,
        # Positive values, so no sum cancels to zero
        data_rvs=lambda k: rng.uniform(0.5, 1.5, k),
    )


def random_sparse_vector(n: int, nnz: int, seed: int):
    rng = np.random.default_rng(seed)
    crd = np.sort(rng.choice(n, nnz, replace=False)).astype(np.int32)
    val = rng.uniform(0.5, 1.5, nnz).astype(np.float32)
    return crd, val


def run(impl: str, A: scipy.sparse.csc_matrix, x_crd: np.ndarray, x_val: np.ndarray):
    """Run impl on the GPU, return (z_crd, z_val) as numpy arrays."""
    args = [
        torch.from_numpy(A.indptr.astype(np.int32)),
        torch.from_numpy(A.indices.astype(np.int32)),
        torch.from_numpy(A.data.astype(np.float32)),
        torch.from_numpy(x_crd),
        torch.from_numpy(x_val),
    ]
    z_crd, z_val = IMPLS[impl](*[arg.cuda() for arg in args], A.shape[0])
    return z_crd.cpu().numpy(), z_val.cpu().numpy()


def reference(A: scipy.sparse.csc_matrix, x_crd: np.ndarray, x_val: np.ndarray):
    """Compute z = A x with scipy, return sorted (z_crd, z_val)."""
    n = A.shape[1]
    x = scipy.sparse.csc_matrix((x_val, x_crd, [0, len(x_crd)]), shape=(n, 1))
    z = (A @ x).tocoo()
    order = np.argsort(z.row)
    return z.row[order], z.data[order]


def check(impl: str, A: scipy.sparse.csc_matrix, x_crd: np.ndarray, x_val: np.ndarray):
    z_crd, z_val = run(impl, A, x_crd, x_val)
    ref_crd, ref_val = reference(A, x_crd, x_val)

    assert z_crd.dtype == np.int32
    assert z_val.dtype == np.float32
    assert np.array_equal(z_crd, ref_crd)
    # The order of the additions differs between implementations
    assert np.allclose(z_val, ref_val, rtol=1e-4)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize(
    "m, n, density, x_nnz",
    [
        (1, 1, 1.0, 1),
        (33, 17, 0.5, 5),
        (1000, 1000, 0.01, 1),
        (1000, 1000, 0.01, 100),
        (1000, 1000, 0.01, 1000),
        (100, 5000, 0.2, 2000),
        (50000, 300, 0.05, 150),
        (100000, 100000, 0.0002, 5000),
    ],
)
def test_spmspv(impl: str, m: int, n: int, density: float, x_nnz: int):
    A = random_csc(m, n, density, seed=0)
    x_crd, x_val = random_sparse_vector(n, x_nnz, seed=1)
    check(impl, A, x_crd, x_val)


@pytest.mark.parametrize("impl", IMPLS)
def test_spmspv_empty_x(impl: str):
    A = random_csc(100, 100, 0.1, seed=0)
    x_crd, x_val = random_sparse_vector(100, 0, seed=1)
    check(impl, A, x_crd, x_val)


@pytest.mark.parametrize("impl", IMPLS)
def test_spmspv_empty_columns(impl: str):
    """x only has nonzeros in empty columns of A."""
    A = random_csc(100, 100, 0.1, seed=0).tolil()
    A[:, 50:] = 0
    A = A.tocsc()
    x_crd, x_val = random_sparse_vector(50, 20, seed=1)
    check(impl, A, x_crd + 50, x_val)


@pytest.mark.parametrize("impl", IMPLS)
def test_spmspv_some_empty_columns(impl: str):
    A = random_csc(1000, 1000, 0.01, seed=0).tolil()
    A[:, ::3] = 0
    A = A.tocsc()
    x_crd, x_val = random_sparse_vector(1000, 400, seed=1)
    check(impl, A, x_crd, x_val)


@pytest.mark.parametrize("impl", IMPLS)
def test_spmspv_dense_rows(impl: str):
    """Most products land in a few rows."""
    A = random_csc(5000, 5000, 0.001, seed=0).tolil()
    A[[7, 4099], :] = 1.0
    A = A.tocsc()
    x_crd, x_val = random_sparse_vector(5000, 3000, seed=1)
    check(impl, A, x_crd, x_val)


@pytest.mark.parametrize("impl", IMPLS)
def test_spmspv_skewed_columns(impl: str):
    """A few columns have most of the nonzeros."""
    A = random_csc(20000, 2000, 0.001, seed=0).tolil()
    A[:, [3, 1500]] = 1.0
    A = A.tocsc()
    x_crd, x_val = random_sparse_vector(2000, 500, seed=1)
    x_crd[:2] = [3, 4]
    x_crd[-1] = 1999
    x_crd = np.unique(np.append(x_crd, 1500)).astype(np.int32)
    x_val = np.resize(x_val, len(x_crd))
    check(impl, A, x_crd, x_val)


@pytest.mark.parametrize("impl", IMPLS)
def test_spmspv_repeated_calls(impl: str):
    """Results must not depend on earlier calls."""
    A = random_csc(3000, 3000, 0.01, seed=0)
    for seed in range(5):
        x_crd, x_val = random_sparse_vector(3000, 50 * (seed + 1), seed)
        check(impl, A, x_crd, x_val)
    B = random_csc(200, 200, 0.05, seed=3)
    x_crd, x_val = random_sparse_vector(200, 20, seed=4)
    check(impl, B, x_crd, x_val)
