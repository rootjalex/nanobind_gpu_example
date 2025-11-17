target_sources(
    nanobind_gpu_example_ext
    PRIVATE
    nanobind_gpu_example_ext.cpp
    nb_utils.h
)

target_include_directories(
    nanobind_gpu_example_ext
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if (GPU_BACKEND STREQUAL "cuda")
    add_subdirectory(cuda)
elseif (GPU_BACKEND STREQUAL "metal")
    add_subdirectory(metal)
endif ()
