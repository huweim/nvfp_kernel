/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <torch/all.h>

#include "cuda_utils.h"
#include "dispatch_utils.h"
#include "nvfp4_utils.cuh"

namespace vllm {

template <typename T>
__global__ void reciprocal_approximate_ftz_kernel(int64_t num_elements,
                                                  const T* input, T* output) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    // Convert to float, apply reciprocal_approximate_ftz, then convert back
    float input_val;
    if constexpr (std::is_same_v<T, half>) {
      input_val = __half2float(input[idx]);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      input_val = __bfloat162float(input[idx]);
    } else {
      input_val = static_cast<float>(input[idx]);
    }

    float output_val = reciprocal_approximate_ftz(input_val);

    if constexpr (std::is_same_v<T, half>) {
      output[idx] = __float2half(output_val);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      output[idx] = __float2bfloat16(output_val);
    } else {
      output[idx] = static_cast<T>(output_val);
    }
  }
}

template <typename T>
void invoke_reciprocal_approximate_ftz(int64_t num_elements, const T* input,
                                       T* output, cudaStream_t stream) {
  const int block_size = 256;
  const int grid_size = (num_elements + block_size - 1) / block_size;

  reciprocal_approximate_ftz_kernel<T>
      <<<grid_size, block_size, 0, stream>>>(num_elements, input, output);
}

// Explicit instantiations for float types
template void invoke_reciprocal_approximate_ftz<float>(int64_t num_elements,
                                                       const float* input,
                                                       float* output,
                                                       cudaStream_t stream);

template void invoke_reciprocal_approximate_ftz<half>(int64_t num_elements,
                                                      const half* input,
                                                      half* output,
                                                      cudaStream_t stream);

template void invoke_reciprocal_approximate_ftz<__nv_bfloat16>(
    int64_t num_elements, const __nv_bfloat16* input, __nv_bfloat16* output,
    cudaStream_t stream);

}  // namespace vllm

void reciprocal_approximate_ftz_tensor(torch::Tensor const& input,
                                       torch::Tensor& output) {
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Float ||
                  input.scalar_type() == at::ScalarType::Half ||
                  input.scalar_type() == at::ScalarType::BFloat16,
              "Input tensor must be float, half, or bfloat16");

  TORCH_CHECK(output.scalar_type() == input.scalar_type(),
              "Output tensor must have the same dtype as input tensor");

  TORCH_CHECK(input.sizes() == output.sizes(),
              "Input and output tensors must have the same shape");

  TORCH_CHECK(input.is_cuda() && output.is_cuda(),
              "Input and output tensors must be on CUDA device");

  int64_t num_elements = input.numel();

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "reciprocal_approximate_ftz_kernel", [&] {
        using cuda_type = scalar_t;
        auto input_ptr = static_cast<cuda_type const*>(input.data_ptr());
        auto output_ptr = static_cast<cuda_type*>(output.data_ptr());

        vllm::invoke_reciprocal_approximate_ftz(num_elements, input_ptr,
                                                output_ptr, stream);
      });
}
