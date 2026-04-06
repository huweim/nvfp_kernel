// binding.cpp
#include <torch/extension.h>

#ifdef NVFP_WITH_REAL_GEMM
#include "nvfp4_quant.h"
#include "nvfp4_scaled_mm_entry.h"
#endif
#include "reciprocal_kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef NVFP_WITH_REAL_GEMM
  m.def("scaled_fp4_quant_sm1xxa", &scaled_fp4_quant_sm1xxa);
  m.def("cutlass_scaled_fp4_mm", &cutlass_scaled_fp4_mm);
#endif
  m.def("reciprocal_approximate_ftz_tensor",
        &reciprocal_approximate_ftz_tensor);
}
