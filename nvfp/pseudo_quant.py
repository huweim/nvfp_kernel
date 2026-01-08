# NOTE: This file is copied from https://github.com/NVIDIA/Fuser/blob/3bdbb55b62f1566c71944d1c88773a1dc16b5e5b/benchmarks/python/layers_for_inference_benchmark.py

import torch
from torch.testing._internal.common_quantized import _f32_to_floatx_unpacked
import torch.nn as nn
from .ops import reciprocal_approximate_ftz_tensor

# Ref: https://github.com/pytorch/pytorch/blob/bffc7dd1/test/test_matmul_cuda.py#L972-L974
def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


# Ref: https://github.com/pytorch/pytorch/blob/bffc7dd1/test/test_matmul_cuda.py#L977-L982
def pack_uint4(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))


# Ref: Based on `_bfloat16_to_float4_e2m1fn_x2` of https://github.com/pytorch/pytorch/blob/bffc7dd1/test/test_matmul_cuda.py#L985-L990
def to_fp4(x: torch.Tensor) -> torch.Tensor:
    x = _f32_to_floatx_unpacked(x.float(), ebits=2, mbits=1)
    x = pack_uint4(x)
    x = x.view(torch.float4_e2m1fn_x2)
    return x


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L8-L10
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L125-L148
def pytorch_nvfp4_quantize(a, a_global_scale):
    BLOCK_SIZE = 16
    assert (
        a.size(-1) % BLOCK_SIZE == 0
    ), "The inner-most dim must be divisible by block_size; Padding is not implemented."
    assert a.is_contiguous(), "Only contiguous tensors are supported."

    original_shape = a.shape
    a_fp32 = a.float().reshape(original_shape[0], -1, BLOCK_SIZE)

    # Find absolute maximum along blockwise dimension
    max_abs = torch.amax(torch.abs(a_fp32), dim=-1)
    block_scale_fp32 = (max_abs / FLOAT4_E2M1_MAX).float()

    scaled_block_scale_fp32 = block_scale_fp32 * a_global_scale
    scaled_block_scale_fp8 = torch.clamp(
        scaled_block_scale_fp32,
        min=FLOAT8_E4M3_EPS,
        max=FLOAT8_E4M3_MAX,
    ).to(torch.float8_e4m3fn)
    scaled_block_scale_fp8_fp32 = scaled_block_scale_fp8.to(torch.float)
    total_scale = scaled_block_scale_fp8_fp32 * reciprocal_approximate_ftz_tensor(a_global_scale)
    a_scaled = a_fp32 * reciprocal_approximate_ftz_tensor(total_scale.unsqueeze(-1))
    a_scaled = torch.clamp(a_scaled, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX)
    a_scaled = a_scaled.view(original_shape)
    return to_fp4(a_scaled), scaled_block_scale_fp8


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L63-L82
# apply swizzled on block scaling factor:
# 1. apply padding to [mn_t * 128 , k_t * 4]
# 2. apply swizzle
def linear_to_swizzled_128_4(a_sf_linear: torch.Tensor):
    mn, sf_k = a_sf_linear.shape
    m_tiles = (mn + 128 - 1) // 128
    mn_padded = m_tiles * 128
    k_tiles = (sf_k + 4 - 1) // 4
    k_padded = k_tiles * 4
    if mn_padded != mn or k_padded != sf_k:
        a_sf_padded = torch.empty(
            mn_padded, k_padded, dtype=a_sf_linear.dtype, device=a_sf_linear.device
        )
        a_sf_padded[0:mn, 0:sf_k] = a_sf_linear
    else:
        a_sf_padded = a_sf_linear
    # details about layout requirement on block-wise scaling factor
    # https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html#scale-factor-layouts
    tmp = torch.reshape(a_sf_padded, (m_tiles, 4, 32, k_tiles, 4))
    return tmp.transpose(1, 3).reshape(mn_padded, k_padded)


@torch.inference_mode()
def quantize_linear_weight_to_nvfp4(
    weight: torch.Tensor | nn.Parameter,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantize weight to nvfp4, returning (packed) e2m1 weight, e4m3 scale factor, fp32 global scale."""
    global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / weight.float().abs().amax()
    ).to(torch.float32)
    fp4_weight, weight_scaling_factor = pytorch_nvfp4_quantize(weight, global_scale)
    weight_scale_interleaved = linear_to_swizzled_128_4(weight_scaling_factor)
    return fp4_weight, weight_scale_interleaved, global_scale


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L13-L22
kE2M1ToFloatTensor = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
    ],
    dtype=torch.float32,
    device="cuda",
)


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L25-L32
# Convert FP4 into FP32
def e2m1_to_fp32_vectorized(int4_values):
    """
    Vectorized version of e2m1_to_fp32.
    int4_values: tensor of uint8, each element in [0, 15]
    """
    sign_bits = int4_values & 0x8  # shape: (...)
    abs_indices = int4_values & 0x7  # values in [0, 7]
    abs_indices = abs_indices.to(torch.int64)

    float_vals = kE2M1ToFloatTensor[abs_indices]  # shape same as int4_values

    float_vals = torch.where(sign_bits != 0, -float_vals, float_vals)
    return float_vals


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L35-L49
# Unpack float4_e2m1fn_x2 into two separate fp32 values
def unpack_fp4_bytes(a):
    assert a.dtype == torch.float4_e2m1fn_x2
    m, n = a.shape
    a = a.view(torch.uint8).flatten()  # shape: (m * n,)

    upper_half_byte = (a & 0xF0) >> 4  # high 4 bits
    lower_half_byte = a & 0x0F  # low 4 bits

    upper_half_float = e2m1_to_fp32_vectorized(upper_half_byte)
    lower_half_float = e2m1_to_fp32_vectorized(lower_half_byte)

    out = torch.stack((lower_half_float, upper_half_float), dim=-1).reshape(m, n * 2)
    return out


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L55-L60
# restore swizzled on block scaling factor:
# 1. restore swizzle
# 2. removes padding via slicing to [:mn, :k]
def swizzled_to_linear_128_4(a_sf_swizzled: torch.Tensor, mn, k):
    mn_padded, sf_k_padded = a_sf_swizzled.shape
    m_tiles = mn_padded // 128
    k_tiles = sf_k_padded // 4
    tmp = torch.reshape(a_sf_swizzled, (m_tiles, k_tiles, 32, 4, 4))
    return tmp.transpose(1, 3).reshape(mn_padded, sf_k_padded)[:mn, :k]


# Ref: https://github.com/NVIDIA/Fuser/blob/d70540f9/tests/python/utils/narrow_precision.py#L85-L101
def dequantize_to_dtype(tensor_fp4, tensor_sf, global_scale, block_size=16):
    """Dequantize the fp4 tensor back to high precision."""
    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.float4_e2m1fn_x2
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = unpack_fp4_bytes(tensor_fp4)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = swizzled_to_linear_128_4(tensor_sf, m, k)
    tensor_sf_dtype = tensor_sf.to(torch.double) / global_scale

    # scale the tensor
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out


def simple_fp4_pseudo_quantize(x: torch.Tensor) -> torch.Tensor:
    """
    Simple pseudo-quantization that converts float/float16/bfloat16 to FP4 and back to float32.
    This function does NOT handle scales - it's a simple quantize-dequantize operation.

    Args:
        x: Input tensor with dtype float, float16, or bfloat16. Must be a CUDA tensor
           with at least 2 dimensions and last dimension divisible by 2.

    Returns:
        torch.Tensor: Dequantized tensor with float32 dtype, same shape as input.
                     The tensor is quantized to FP4 and then dequantized back to float32,
                     simulating the precision loss of FP4 quantization.
    """
    assert x.is_cuda, "x must be a CUDA tensor"
    assert x.dtype in (
        torch.float,
        torch.float16,
        torch.bfloat16,
    ), f"x.dtype needs to be float, fp16 or bf16 but got {x.dtype}"
    assert x.ndim >= 2, f"x.ndim needs to be >= 1, but got {x.ndim}"
    assert (
        x.shape[-1] % 2 == 0
    ), f"last dim has to be multiple of 2, but got {x.shape[-1]}"

    original_shape = x.shape

    # Convert to FP4
    x_fp4 = to_fp4(x.reshape(-1, original_shape[-1]))

    # Convert back to float32 first
    x_dequantized = unpack_fp4_bytes(x_fp4)

    # Cast back to original dtype
    return x_dequantized.reshape(original_shape)


def nvfp4_pseudo_quantize(x: torch.Tensor) -> torch.Tensor:
    """
    NVIDIA FP4 pseudo-quantization that converts float/float16/bfloat16 to NVFP4 and back to float32.
    This function uses block-wise scaling with swizzled scale factors for optimal performance.

    The quantization process includes:
    1. Block-wise scaling (block size = 16) to preserve precision
    2. Global scaling to fit within FP4 range
    3. Swizzled scale factor layout for efficient memory access
    4. FP4 quantization using E2M1 format
    5. Dequantization back to float32

    Args:
        x: Input tensor with dtype float, float16, or bfloat16. Must be a CUDA tensor
           with at least 1 dimension and last dimension divisible by 16.

    Returns:
        torch.Tensor: Dequantized tensor with float32 dtype, same shape as input.
                     The tensor undergoes NVFP4 quantization with block-wise scaling
                     and is then dequantized back to float32, providing a realistic
                     simulation of NVFP4 precision loss and scaling effects.
    """
    assert x.is_cuda, "x must be a CUDA tensor"
    assert x.dtype in (
        torch.float,
        torch.float16,
        torch.bfloat16,
    ), f"x.dtype needs to be float, fp16 or bf16 but got {x.dtype}"
    assert x.ndim >= 1, f"x.ndim needs to be >= 1, but got {x.ndim}"
    assert (
        x.shape[-1] % 16 == 0
    ), f"last dim has to be multiple of 16, but got {x.shape[-1]}"
    org_shape = x.shape
    x = x.reshape(-1, org_shape[-1])
    fp4_weight, weight_scale_interleaved, weight_global_scale = (
        quantize_linear_weight_to_nvfp4(x)
    )
    quantized_x = dequantize_to_dtype(
        fp4_weight,
        weight_scale_interleaved,
        weight_global_scale,
        16,
    )
    return quantized_x.reshape(org_shape)
