// binding.cpp
#include <torch/extension.h>

#include "reciprocal_kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reciprocal_approximate_ftz_tensor",
        &reciprocal_approximate_ftz_tensor);
}
