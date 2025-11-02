target_sources(
    nanobind_gpu_example_ext
    PRIVATE
    nanobind_gpu_example_ext.cpp
)

if (GPU_BACKEND STREQUAL "cuda")
    add_subdirectory(cuda)
elseif (GPU_BACKEND STREQUAL "metal")
    add_subdirectory(metal)
endif ()
