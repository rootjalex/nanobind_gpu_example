find_path(
    MetalCPP_INCLUDE_DIR Metal/Metal.hpp
    PATH_SUFFIXES metal-cpp
    HINTS "${PROJECT_SOURCE_DIR}"
)

find_library(Foundation_LIBRARY Foundation)
find_library(QuartzCore_LIBRARY QuartzCore)
find_library(Metal_LIBRARY Metal)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    MetalCPP
    REQUIRED_VARS
    MetalCPP_INCLUDE_DIR
    Foundation_LIBRARY
    QuartzCore_LIBRARY
    Metal_LIBRARY
)

if (MetalCPP_FOUND AND NOT TARGET MetalCPP::MetalCPP)
    add_library(MetalCPP::MetalCPP INTERFACE IMPORTED)
    target_include_directories(
        MetalCPP::MetalCPP
        INTERFACE
        "${MetalCPP_INCLUDE_DIR}"
    )
    target_link_libraries(
        MetalCPP::MetalCPP
        INTERFACE
        ${Foundation_LIBRARY}
        ${QuartzCore_LIBRARY}
        ${Metal_LIBRARY}
    )
endif ()
