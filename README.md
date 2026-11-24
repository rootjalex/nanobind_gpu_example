nanobind_gpu_example
====================

This repository contains a tiny project showing how to create C++ bindings
using [nanobind] and [scikit-build-core], that use GPU (currently Metal or CUDA)
compute shaders. It was derived from the original [nanobind example project]
developed by [@wjakob].

Requirements
------------

- Python 3.11 or newer
- **Linux / CUDA:** the CUDA toolkit, with `nvcc` on your `PATH` (or pointed
  to by the `CUDACXX` environment variable, e.g.
  `CUDACXX=/usr/local/cuda/bin/nvcc`)
- **Apple Silicon / Metal:** a copy of [metal-cpp] (see below)

Installation
------------

We strongly recommend using [uv](https://docs.astral.sh/uv) as your dependency
manager.

1. Clone this repository; `cd nanobind_gpu_example`
2. If running on Apple Silicon, copy [metal-cpp] into the root of this project
   directory.
3. Run `uv sync`

The GPU backend is chosen automatically: Metal on macOS and CUDA everywhere
else. To pick one explicitly, set the `GPU_BACKEND` CMake option to `cuda` or
`metal`, e.g. `SKBUILD_CMAKE_DEFINE="GPU_BACKEND=cuda" uv sync`.

`uv sync` and `uv run` rebuild the extension whenever a file under `src/` or a
CMake file changes.

Testing
-------

Afterward, you should be able to run the test suite, which profiles the Torch GPU backend (MPS or CUDA)
against raw Metal or CUDA via nanobind and Torch CPU. The largest benchmark
adds vectors of 10^9 floats, which takes 16 GB of GPU memory; on CUDA it is
skipped when the GPU does not have that much free.

The output below is from an NVIDIA RTX 5090 with CUDA 13.2.

```bash
% uv run pytest -vs
============================= test session starts ==============================
platform linux -- Python 3.13.13, pytest-9.2.0, pluggy-1.7.0 -- /path/to/nanobind_gpu_example/.venv/bin/python
cachedir: .pytest_cache
rootdir: /path/to/nanobind_gpu_example
configfile: pyproject.toml
testpaths: tests
collected 19 items

tests/test_basic.py::test_gpu_add[100] 
Using NVIDIA CUDA backend.

Torch cuda: 0.006 ms
nanobind cuda: 0.009 ms
Torch CPU: 0.001 ms
PASSED
tests/test_basic.py::test_gpu_add[1000] 
Torch cuda: 0.006 ms
nanobind cuda: 0.009 ms
Torch CPU: 0.001 ms
PASSED
tests/test_basic.py::test_gpu_add[10000] 
Torch cuda: 0.006 ms
nanobind cuda: 0.009 ms
Torch CPU: 0.001 ms
PASSED
tests/test_basic.py::test_gpu_add[100000] 
Torch cuda: 0.006 ms
nanobind cuda: 0.009 ms
Torch CPU: 0.005 ms
PASSED
tests/test_basic.py::test_gpu_add[1000000] 
Torch cuda: 0.008 ms
nanobind cuda: 0.011 ms
Torch CPU: 0.010 ms
PASSED
tests/test_basic.py::test_gpu_add[10000000] 
Torch cuda: 0.059 ms
nanobind cuda: 0.063 ms
Torch CPU: 0.950 ms
PASSED
tests/test_basic.py::test_gpu_add[100000000] 
Torch cuda: 0.766 ms
nanobind cuda: 0.770 ms
Torch CPU: 36.088 ms
PASSED
tests/test_basic.py::test_gpu_add[1000000000] 
Torch cuda: 7.622 ms
nanobind cuda: 7.605 ms
Torch CPU: 379.274 ms
PASSED
tests/test_basic.py::test_gpu_add_allocating[1] PASSED
tests/test_basic.py::test_gpu_add_allocating[1000] PASSED
tests/test_basic.py::test_gpu_add_allocating[1000000] PASSED
tests/test_basic.py::test_gpu_add_empty PASSED
tests/test_basic.py::test_gpu_add_past_int32_index PASSED
tests/test_basic.py::test_length_mismatch_is_rejected[100-10-100] PASSED
tests/test_basic.py::test_length_mismatch_is_rejected[10-100-100] PASSED
tests/test_basic.py::test_length_mismatch_is_rejected[100-100-10] PASSED
tests/test_basic.py::test_length_mismatch_is_rejected[100-100-1000] PASSED
tests/test_basic.py::test_out_is_never_converted PASSED
tests/test_basic.py::test_inputs_may_be_converted PASSED

============================= 19 passed in 12.57s ==============================
```

License
-------

This project is released under the BSD 3-Clause license; see
[LICENSE](./LICENSE). It is derived from Wenzel Jakob's
[nanobind example project], whose copyright notice is retained there.

[@wjakob]: https://github.com/wjakob

[metal-cpp]: https://developer.apple.com/metal/cpp

[nanobind example project]: https://github.com/wjakob/nanobind_example

[nanobind]: https://github.com/wjakob/nanobind

[scikit-build-core]: https://scikit-build-core.readthedocs.io/en/latest/index.html
