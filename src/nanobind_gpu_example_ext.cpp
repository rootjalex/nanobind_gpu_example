#include <nanobind/nanobind.h>


#if defined(USE_CUDA)
#include "cuda/add.h"
#include "cuda/utils.h"
#elif defined(USE_METAL)
#include "metal/add.h"
#include "metal/utils.h"
#else
#error "Cannot compile without METAL or CUDA"
#endif

namespace nb = nanobind;

NB_MODULE(nanobind_gpu_example_ext, m) {
    m.doc() = "This is a \"hello world\" example with nanobind on GPU";
    m.def("vecf_add", vecf_add);
    m.def("vecf_add_out", vecf_add_out);
    m.def("synchronize", synchronize);
}
