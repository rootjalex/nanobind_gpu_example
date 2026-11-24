#pragma once

#include "nb_utils.h"

// The interface every GPU backend (src/cuda, src/metal) implements. Argument
// validation happens once, in the binding layer, so backends may assume that
// all vectors passed to them have the same length.

// Allocates and returns a new buffer holding a + b.
GPUVector<float> vecf_add(const GPUVector<float>& a, const GPUVector<float>& b);

// Writes a + b into a preallocated buffer.
void vecf_add_out(const GPUVector<float>& a, const GPUVector<float>& b,
                  GPUVector<float>& out);

// Blocks until all GPU work submitted so far has finished.
void synchronize();
