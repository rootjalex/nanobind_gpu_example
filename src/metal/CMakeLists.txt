find_package(MetalCPP REQUIRED)

target_sources(
    nanobind_gpu_example_ext
    PRIVATE
    add.cpp
    add.h
    metal_impl.cpp
    metal_impl.h
    nb_utils.h
    utils.cpp
    utils.h
)

target_compile_definitions(
    nanobind_gpu_example_ext
    PRIVATE
    USE_METAL
)

target_link_libraries(
    nanobind_gpu_example_ext
    PRIVATE
    MetalCPP::MetalCPP
)
