find_package(CUDAToolkit REQUIRED)

target_sources(
    nanobind_gpu_example_ext
    PRIVATE
    add.cpp
    add.h
    nb_utils.h
    utils.h
    kernels/add.cu
    kernels/add.h
    kernels/free.cu
    kernels/free.h
    kernels/utils.cu
    kernels/utils.h
)

target_compile_definitions(
    nanobind_gpu_example_ext
    PRIVATE
    USE_CUDA
)

target_link_libraries(
    nanobind_gpu_example_ext
    PRIVATE
    CUDA::cudart
)
