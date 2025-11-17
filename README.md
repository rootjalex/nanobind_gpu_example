nanobind_gpu_example
================

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
3. Run `uv sync`

Testing
-------

Afterward, you should be able to run the test suite, which profiles Torch MPS
against raw Metal with nanobind and Torch CPU.

```bash
% pytest -vs
============================= test session starts =============================
platform darwin -- Python 3.13.9, pytest-8.4.2, pluggy-1.6.0 -- /path/to/nanobind_gpu_example/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /path/to/nanobind_gpu_example
configfile: pyproject.toml
collected 8 items                                                                                                                                                                                               

tests/test_basic.py::test_gpu_add[100] 
Using Apple Metal (MPS) backend.

Torch mps: 0.184 ms
nanobind mps: 0.215 ms
Torch CPU: 0.000 ms
PASSED
tests/test_basic.py::test_gpu_add[1000] 
Torch mps: 0.177 ms
nanobind mps: 0.180 ms
Torch CPU: 0.001 ms
PASSED
tests/test_basic.py::test_gpu_add[10000] 
Torch mps: 0.180 ms
nanobind mps: 0.188 ms
Torch CPU: 0.001 ms
PASSED
tests/test_basic.py::test_gpu_add[100000] 
Torch mps: 0.188 ms
nanobind mps: 0.208 ms
Torch CPU: 0.042 ms
PASSED
tests/test_basic.py::test_gpu_add[1000000] 
Torch mps: 0.201 ms
nanobind mps: 0.237 ms
Torch CPU: 0.064 ms
PASSED
tests/test_basic.py::test_gpu_add[10000000] 
Torch mps: 0.581 ms
nanobind mps: 0.532 ms
Torch CPU: 0.670 ms
PASSED
tests/test_basic.py::test_gpu_add[100000000] 
Torch mps: 3.603 ms
nanobind mps: 3.595 ms
Torch CPU: 6.469 ms
PASSED
tests/test_basic.py::test_gpu_add[1000000000] 
Torch mps: 34.597 ms
nanobind mps: 34.160 ms
Torch CPU: 63.671 ms
PASSED

============================== 8 passed in 9.71s ==============================
```

License
-------

I kept the original [LICENSE](./LICENSE) file from Wenzel's example because I
don't really understand how licenses work and I don't want to get in trouble.

[@wjakob]: https://github.com/wjakob

[metal-cpp]: https://developer.apple.com/metal/cpp

[nanobind example project]: https://github.com/wjakob/nanobind_example

[nanobind]: https://github.com/wjakob/nanobind

[scikit-build-core]: https://scikit-build-core.readthedocs.io/en/latest/index.html
