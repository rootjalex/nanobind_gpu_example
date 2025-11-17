#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

#if defined(USE_CUDA)
#include "cuda/kernels/free.h"
using gpu_device = nb::device::cuda;
#elif defined(USE_METAL)
#include "metal/kernels/free.h"
using gpu_device = nb::device::metal;
#else
#error "Cannot compile without METAL or CUDA"
#endif

#include <iostream>

template <typename T>
using GPUVector =
    nb::ndarray<nb::pytorch, T, nb::shape<-1>, nb::c_contig, gpu_device>;

template <typename T>
using CPUVector =
    nb::ndarray<nb::pytorch, T, nb::shape<-1>, nb::c_contig, nb::device::cpu>;

template<typename T>
GPUVector<T> make_gpu_vector(T* ptr, const size_t n) {
    nb::capsule owner(ptr, releaseBuffer);

    const size_t shape[1] = {n};

    return GPUVector<T>(ptr, /* ndim = */ 1, shape, owner,
                        /* strides */ nullptr, nb::dtype<T>(),
                        /* explicitly set device type */ gpu_device::value);
}
