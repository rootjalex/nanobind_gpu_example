find_package(CUDAToolkit REQUIRED)

# cuda::hash is not in a CCCL release yet, so use a pinned commit of CCCL main
# instead of the CCCL in the CUDA toolkit
include(FetchContent)
set(CCCL_COMMIT 66c54534566657584ea83c3e90d58c061e647716)
FetchContent_Declare(
    CCCL
    URL https://github.com/NVIDIA/cccl/archive/${CCCL_COMMIT}.tar.gz
    URL_HASH SHA256=7ac57ceb8cd6269046b8af90f83782825a98b477761f6fd1bf48f2ed0f72b24a
)
FetchContent_MakeAvailable(CCCL)

target_sources(
    nanobind_gpu_example_ext
    PRIVATE
    add.cpp
    spmspv.cpp
    kernels/add.cu
    kernels/add.h
    kernels/free.cu
    kernels/free.h
    kernels/pool.cu
    kernels/pool.h
    kernels/utils.cu
    kernels/utils.h
    kernels/spmspv/common.h
    kernels/spmspv/compress.h
    kernels/spmspv/cusparse.cpp
    kernels/spmspv/dense.cu
    kernels/spmspv/esc.cu
    kernels/spmspv/hash.cu
    kernels/spmspv/partition.cu
    kernels/spmspv/spmspv.h
)

target_compile_definitions(
    nanobind_gpu_example_ext
    PRIVATE
    USE_CUDA
)

target_link_libraries(
    nanobind_gpu_example_ext
    PRIVATE
    CCCL::CCCL
    CUDA::cudart
    CUDA::cusparse
)

target_include_directories(
    nanobind_gpu_example_ext
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
