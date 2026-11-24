#pragma once

#include "nb_utils.h"

// Functions implemented by each backend (cuda or metal)

// Allocates and returns a new buffer
GPUVector<float> vecf_add(const GPUVector<float>& a, const GPUVector<float>& b);

// Writes result into a preallocated buffer
void vecf_add_out(const GPUVector<float>& a, const GPUVector<float>& b,
                  GPUVector<float>& out);

// Waits for all GPU work to finish
void synchronize();

#if defined(USE_CUDA)
// Adds the SpMSpV functions to the module (CUDA only)
void bind_spmspv(nb::module_& m);
#endif
