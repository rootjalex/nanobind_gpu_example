cmake_minimum_required(VERSION 4.1)

project(nanobind_gpu_example LANGUAGES CXX)

# ------------------------------------------------------------------------------
# GPU Backend selection
# ------------------------------------------------------------------------------

if (NOT GPU_BACKEND)
    if (APPLE)
        set(GPU_BACKEND "metal")
    else ()
        set(GPU_BACKEND "cuda")
    endif ()
endif ()

set(GPU_BACKEND "${GPU_BACKEND}" CACHE STRING "GPU backend to use (cuda or metal)")

if (NOT GPU_BACKEND MATCHES "cuda|metal")
    message(FATAL_ERROR "Unsupported GPU_BACKEND: ${GPU_BACKEND}. Supported backends are 'cuda' and 'metal'.")
endif ()

# ------------------------------------------------------------------------------
# Global dependencies
# ------------------------------------------------------------------------------

# Attach custom find modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Try to import all Python components potentially needed by nanobind
find_package(
    Python 3.8
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule
)

# Import nanobind through CMake's find_package mechanism
find_package(nanobind CONFIG REQUIRED)

# ------------------------------------------------------------------------------
# Project target
# ------------------------------------------------------------------------------

nanobind_add_module(
    # Name of the extension
    nanobind_gpu_example_ext

    # Target the stable ABI for Python 3.12+, which reduces
    # the number of binary wheels that must be built. This
    # does nothing on older Python versions
    STABLE_ABI

    # Build libnanobind statically and merge it into the
    # extension (which itself remains a shared library)
    #
    # If your project builds multiple extensions, you can
    # replace this flag by NB_SHARED to conserve space by
    # reusing a shared libnanobind across libraries
    NB_STATIC
)

add_subdirectory(src)

if (SKBUILD)
    install(TARGETS nanobind_gpu_example_ext
            LIBRARY DESTINATION nanobind_gpu_example)
endif ()
