#include <nanobind/nanobind.h>

#include <stdexcept>
#include <string>

#include "backend.h"

using namespace nb::literals;

namespace {

void check_same_length(const char* what, const size_t expected,
                       const size_t actual) {
    if (expected != actual) {
        throw std::invalid_argument(std::string(what) + " has length " +
                                    std::to_string(actual) + ", expected " +
                                    std::to_string(expected));
    }
}

}  // namespace

NB_MODULE(nanobind_gpu_example_ext, m) {
    m.doc() = "This is a \"hello world\" example with nanobind on GPU";

    // No conversions. A converted out is a temporary copy, and torch may not
    // have written a converted a or b before the kernel runs (e.g., on MPS).
    m.def(
        "vecf_add",
        [](const GPUVector<float>& a, const GPUVector<float>& b) {
            check_same_length("b", a.shape(0), b.shape(0));
            return vecf_add(a, b);
        },
        "a"_a.noconvert(), "b"_a.noconvert(),
        "Allocates and returns a new buffer");

    m.def(
        "vecf_add_out",
        [](const GPUVector<float>& a, const GPUVector<float>& b,
           GPUVector<float>& out) {
            check_same_length("b", a.shape(0), b.shape(0));
            check_same_length("out", a.shape(0), out.shape(0));
            // Empty tensors have no buffer
            if (a.shape(0) == 0) {
                return;
            }
            vecf_add_out(a, b, out);
        },
        "a"_a.noconvert(), "b"_a.noconvert(), "out"_a.noconvert(),
        "Writes result into a preallocated buffer");

    m.def("synchronize", synchronize, "Waits for all GPU work to finish");

#if defined(USE_CUDA)
    bind_spmspv(m);
#endif
}
