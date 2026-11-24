import time

import nanobind_gpu_example
import pytest
import torch

ITERS = 10


@pytest.fixture(scope="module")
def device_type() -> str:
    """Detect if CUDA or MPS is available; return ('cuda' | 'mps')."""
    if torch.cuda.is_available():
        print("\nUsing NVIDIA CUDA backend.")
        return "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("\nUsing Apple Metal (MPS) backend.")
        return "mps"
    else:
        pytest.skip("No supported GPU backend found (CUDA or MPS).")


def get_torch_sync(device_type: str):
    """Synchronize GPU execution depending on backend."""
    if device_type == "cuda":
        return torch.cuda.synchronize
    elif device_type == "mps":
        return torch.mps.synchronize
    else:
        raise RuntimeError(f"Unsupported device type: {device_type}")


def require_gpu_memory(device_type: str, n_bytes: int) -> None:
    """Skip if the GPU does not have enough free memory."""
    if device_type != "cuda":
        # No reliable way to check free memory on MPS
        return
    # Release memory cached by earlier tests
    torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < n_bytes:
        pytest.skip(f"Not enough free GPU memory (need {n_bytes / 2**30:.1f} GiB).")


def bench_ms(fn, sync=lambda: None) -> float:
    """Average time of fn() in ms."""
    # Warmup (important to trigger GPU pipeline compilation)
    for _ in range(ITERS):
        fn()
        sync()

    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    # Drop low and high outliers, take average of remaining
    times.sort()
    return sum(times[1:-1]) / (len(times) - 2)


@pytest.mark.parametrize("n", [10**i for i in range(2, 10)])
def test_gpu_add(n: int, device_type: str):
    # 4 float32 vectors on the GPU
    require_gpu_memory(device_type, 4 * 4 * n)
    print()

    device = torch.device(device_type)
    x = torch.randn(n, dtype=torch.float32, device=device)
    y = torch.randn(n, dtype=torch.float32, device=device)
    torch_sync = get_torch_sync(device_type)

    # Pre-allocate output buffers
    r_torch = torch.empty_like(x)  # PyTorch output
    r_nbgpu = torch.empty_like(x)  # nanobind GPU output

    # --- Torch GPU add timing ---
    avg_ms = bench_ms(lambda: torch.add(x, y, out=r_torch), torch_sync)
    print(f"Torch {device_type}: {avg_ms:.3f} ms")

    # --- nanobind GPU add timing ---
    avg_ms = bench_ms(
        lambda: nanobind_gpu_example.vecf_add_out(x, y, r_nbgpu),
        nanobind_gpu_example.synchronize,
    )
    print(f"nanobind {device_type}: {avg_ms:.3f} ms")

    # --- Validation ---
    assert torch.equal(r_torch, r_nbgpu), (
        f"Mismatch between Torch and {device_type} results"
    )

    # --- CPU timing ---
    x_cpu = x.to("cpu")
    y_cpu = y.to("cpu")
    r_cpu = torch.empty_like(x_cpu)

    avg_ms = bench_ms(lambda: torch.add(x_cpu, y_cpu, out=r_cpu))
    print(f"Torch CPU: {avg_ms:.3f} ms")

    assert torch.equal(r_cpu, r_torch.to("cpu")), (
        "Mismatch between Torch (CPU) and Torch (GPU) results"
    )


@pytest.mark.parametrize("n", [1, 1000, 10**6])
def test_gpu_add_allocating(n: int, device_type: str):
    x = torch.randn(n, dtype=torch.float32, device=device_type)
    y = torch.randn(n, dtype=torch.float32, device=device_type)
    # Torch must finish writing x and y first (important on MPS)
    get_torch_sync(device_type)()

    r = nanobind_gpu_example.vecf_add(x, y)
    nanobind_gpu_example.synchronize()

    assert r.dtype == torch.float32
    assert r.device.type == device_type
    assert torch.equal(r, x + y)


def test_gpu_add_empty(device_type: str):
    x = torch.empty(0, dtype=torch.float32, device=device_type)
    out = torch.empty(0, dtype=torch.float32, device=device_type)

    nanobind_gpu_example.vecf_add_out(x, x, out)
    nanobind_gpu_example.synchronize()


def test_gpu_add_past_int32_index(device_type: str):
    """Check elements past INT_MAX get written."""
    if device_type != "cuda":
        pytest.skip("CUDA only")
    n = 2**31 + 1000
    require_gpu_memory(device_type, 3 * 4 * n)

    x = torch.ones(n, dtype=torch.float32, device=device_type)
    y = torch.ones(n, dtype=torch.float32, device=device_type)
    out = torch.zeros(n, dtype=torch.float32, device=device_type)

    nanobind_gpu_example.vecf_add_out(x, y, out)
    nanobind_gpu_example.synchronize()

    assert bool((out[-2000:] == 2).all())


@pytest.mark.parametrize(
    "len_a, len_b, len_out",
    [(100, 10, 100), (10, 100, 100), (100, 100, 10), (100, 100, 1000)],
)
def test_length_mismatch_is_rejected(
    len_a: int, len_b: int, len_out: int, device_type: str
):
    a = torch.ones(len_a, dtype=torch.float32, device=device_type)
    b = torch.ones(len_b, dtype=torch.float32, device=device_type)
    out = torch.zeros(len_out, dtype=torch.float32, device=device_type)

    with pytest.raises(ValueError, match="has length"):
        nanobind_gpu_example.vecf_add_out(a, b, out)
    assert not out.any(), "out should not be modified"

    if len_a != len_b:
        with pytest.raises(ValueError, match="has length"):
            nanobind_gpu_example.vecf_add(a, b)


def test_out_is_never_converted(device_type: str):
    """vecf_add_out should reject an out tensor that nanobind would copy."""
    x = torch.ones(8, dtype=torch.float32, device=device_type)

    unsupported = {
        "wrong dtype": torch.zeros(8, dtype=torch.float16, device=device_type),
        "strided view": torch.zeros(16, dtype=torch.float32, device=device_type)[::2],
        "2-D": torch.zeros(2, 4, dtype=torch.float32, device=device_type),
        "CPU tensor": torch.zeros(8, dtype=torch.float32),
    }
    for label, bad in unsupported.items():
        with pytest.raises(TypeError):
            nanobind_gpu_example.vecf_add_out(x, x, bad)
        assert not bad.any(), f"out ({label}) should not be modified"


def test_inputs_may_be_converted(device_type: str):
    """Inputs with a different dtype or stride get converted."""
    x = torch.arange(8, dtype=torch.float32, device=device_type)
    out = torch.zeros(8, dtype=torch.float32, device=device_type)
    expected = x + x

    half = x.to(torch.float16)
    strided = torch.arange(0, 8, 0.5, dtype=torch.float32, device=device_type)[::2]
    for a in (half, strided):
        out.zero_()
        # Torch must finish writing the tensors first (important on MPS)
        get_torch_sync(device_type)()
        nanobind_gpu_example.vecf_add_out(a, x, out)
        nanobind_gpu_example.synchronize()
        assert torch.equal(out, expected)

        r = nanobind_gpu_example.vecf_add(x, a)
        nanobind_gpu_example.synchronize()
        assert r.dtype == torch.float32
        assert torch.equal(r, expected)

    # Tensors on another device are not converted
    with pytest.raises(TypeError):
        nanobind_gpu_example.vecf_add(x, x.to("cpu"))
