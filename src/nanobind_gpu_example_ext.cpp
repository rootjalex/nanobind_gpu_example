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

    m.def(
        "vecf_add",
        [](const GPUVector<float>& a, const GPUVector<float>& b) {
            check_same_length("b", a.shape(0), b.shape(0));
            return vecf_add(a, b);
        },
        "a"_a, "b"_a, "Return a new float32 GPU tensor holding a + b.");

    m.def(
        "vecf_add_out",
        [](const GPUVector<float>& a, const GPUVector<float>& b,
           GPUVector<float>& out) {
            check_same_length("b", a.shape(0), b.shape(0));
            check_same_length("out", a.shape(0), out.shape(0));
            vecf_add_out(a, b, out);
        },
        // `out` must not be implicitly converted: nanobind would hand us a
        // temporary copy, and the result written to it would be silently lost.
        "a"_a, "b"_a, "out"_a.noconvert(),
        "Write a + b into the preallocated float32 GPU tensor out.");

    m.def("synchronize", synchronize,
          "Block until all GPU work submitted so far has finished.");
}
