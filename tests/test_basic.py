import time

import nanobind_gpu_example
import pytest
import torch


def get_torch_sync(device_type: str):
    """Synchronize GPU execution depending on backend."""
    if device_type == "cuda":
        return torch.cuda.synchronize
    elif device_type == "mps":
        return torch.mps.synchronize
    else:
        raise RuntimeError(f"Unsupported device type: {device_type}")


@pytest.fixture(scope="module")
def device_type():
    """Detect if CUDA or MPS is available; return ('cuda' | 'mps')."""
    if torch.cuda.is_available():
        print("\nUsing NVIDIA CUDA backend.")
        return "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("\nUsing Apple Metal (MPS) backend.")
        return "mps"
    else:
        pytest.skip("No supported GPU backend found (CUDA or MPS).")


@pytest.mark.parametrize("n", [10 ** i for i in range(2, 10)])
def test_gpu_add(n: int, device_type: str):
    print()

    device = torch.device(device_type)
    x = torch.randn(n, dtype=torch.float32, device=device)
    y = torch.randn(n, dtype=torch.float32, device=device)
    torch_sync = get_torch_sync(device_type)

    # Pre-allocate output buffers
    r_torch = torch.empty_like(x)  # PyTorch output
    r_nbgpu = torch.empty_like(x)  # nanobind GPU output

    iters = 10

    with torch.no_grad():
        # Warmup
        for i in range(iters):
            torch.add(x, y, out=r_torch)
            torch_sync()

        torch_times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            torch.add(x, y, out=r_torch)
            torch_sync()
            t1 = time.perf_counter()
            torch_times.append((t1 - t0) * 1000.0)  # ms

        # Drop low and high outliers, take average of remaining
        torch_times.sort()
        avg_ms = sum(torch_times[1:-1]) / (len(torch_times) - 2)
        print(f"Torch {device_type}: {avg_ms:.3f} ms")

    torch_sync()
    # Warmup (important to trigger GPU pipeline compilation)
    # _ = nanobind_gpu_example.vecf_add(x, y)
    for i in range(iters):
        nanobind_gpu_example.vecf_add_out(x, y, r_nbgpu)
        nanobind_gpu_example.synchronize()

    # --- nanobind GPU add timing ---

    nbgpu_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        nanobind_gpu_example.vecf_add_out(x, y, r_nbgpu)
        nanobind_gpu_example.synchronize()
        t1 = time.perf_counter()
        nbgpu_times.append((t1 - t0) * 1000.0)  # ms

    # Drop low and high outliers, take average of remaining
    nbgpu_times.sort()
    avg_ms = sum(nbgpu_times[1:-1]) / (len(nbgpu_times) - 2)
    print(f"nanobind {device_type}: {avg_ms:.3f} ms")

    # --- Validation ---
    assert torch.equal(r_torch, r_nbgpu), f"Mismatch between Torch and {device_type} results"

    # --- CPU timing ---
    x_cpu = x.to("cpu")
    y_cpu = y.to("cpu")
    r_cpu = torch.empty_like(x_cpu)

    # Warmup
    for i in range(iters):
        torch.add(x_cpu, y_cpu, out=r_cpu)

    cpu_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        torch.add(x_cpu, y_cpu, out=r_cpu)
        t1 = time.perf_counter()
        cpu_times.append((t1 - t0) * 1000.0)

    cpu_times.sort()
    avg_ms = sum(cpu_times[1:-1]) / (len(cpu_times) - 2)
    print(f"Torch CPU: {avg_ms:.3f} ms")

    torch_cpu = r_torch.to("cpu")

    assert torch.equal(r_cpu, torch_cpu), "Mismatch between Torch (CPU) and Torch (GPU) results"
