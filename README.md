nanobind_gpu_example
====================

This repository contains a tiny project showing how to create C++ bindings
using [nanobind] and [scikit-build-core], that use GPU (currently Metal or CUDA)
compute shaders. It was derived from the original [nanobind example project]
developed by [@wjakob].

Installation
------------

We strongly recommend using [uv](https://docs.astral.sh/uv) as your dependency
manager.

1. Clone this repository; `cd nanobind_gpu_example`
2. If running on Apple Silicon, copy [metal-cpp] into the root of this project
   directory.
3. If running on Linux, make sure the CUDA toolkit is installed and `nvcc` is
   on your `PATH` (or set `CUDACXX` to the path of `nvcc`).
4. Run `uv sync`

CMake selects the Metal backend on macOS and the CUDA backend on other
platforms. Set the `GPU_BACKEND` CMake option to `cuda` or `metal` to select a
backend manually.

After changing any file in `src`, run `uv sync` again to rebuild the extension.

Testing
-------

Afterward, you should be able to run the test suite, which profiles the Torch GPU backend (MPS or CUDA)
against raw Metal or CUDA via nanobind and Torch CPU. The largest test needs
16 GB of free GPU memory, and is skipped on CUDA GPUs with less free memory.

```bash
% uv run pytest -vs
============================= test session starts ==============================
platform linux -- Python 3.13.13, pytest-9.2.0, pluggy-1.7.0 -- /path/to/nanobind_gpu_example/.venv/bin/python
cachedir: .pytest_cache
rootdir: /path/to/nanobind_gpu_example
configfile: pyproject.toml
testpaths: tests
collected 18 items

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
Torch cuda: 0.009 ms
nanobind cuda: 0.011 ms
Torch CPU: 0.015 ms
PASSED
tests/test_basic.py::test_gpu_add[10000000] 
Torch cuda: 0.059 ms
nanobind cuda: 0.063 ms
Torch CPU: 0.955 ms
PASSED
tests/test_basic.py::test_gpu_add[100000000] 
Torch cuda: 0.765 ms
nanobind cuda: 0.770 ms
Torch CPU: 36.002 ms
PASSED
tests/test_basic.py::test_gpu_add[1000000000] 
Torch cuda: 7.620 ms
nanobind cuda: 7.612 ms
Torch CPU: 375.757 ms
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
tests/test_basic.py::test_conversion_is_rejected PASSED

============================= 18 passed in 13.37s ==============================
```

License
-------

This project uses the BSD 3-Clause [LICENSE](./LICENSE) from Wenzel's example.

[@wjakob]: https://github.com/wjakob

[metal-cpp]: https://developer.apple.com/metal/cpp

[nanobind example project]: https://github.com/wjakob/nanobind_example

[nanobind]: https://github.com/wjakob/nanobind

[scikit-build-core]: https://scikit-build-core.readthedocs.io/en/latest/index.html
