import torch
import scaled_fp4_ops

def scaled_fp4_quant(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 and return quantized tensor and scale.

    This function quantizes the last dimension of the given tensor `input`. For
    every 16 consecutive elements, a single dynamically computed scaling factor
    is shared. This scaling factor is quantized using the `input_global_scale`
    and is stored in a swizzled layout (see
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x).

    Args:
        input: The input tensor to be quantized to FP4
        input_global_scale: A scalar scaling factor for the entire tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP4 but every
            two values are packed into a uint8 and float8_e4m3 scaling factors
            in the sizzled layout.
    """
    # assert not current_platform.is_rocm()
    assert input.is_cuda, "input must be a CUDA tensor"
    assert input_global_scale.is_cuda, "input_global_scale must be a CUDA tensor"
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    block_size = 16
    device = input.device

    assert n % block_size == 0, f"last dim has to be multiple of 16, but got {n}."
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."
    assert input_global_scale.dtype == torch.float, f"input_global_scale.dtype needs to be fp32 but got {input_global_scale.dtype}."

    # Two fp4 values will be packed into an uint8.
    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)

    # We use the rounded values to store the swizzled values. Due to the
    # requirement of the Tensor Core, the minimum tile is 128x4 for the scales.
    # So, we first pad the scales to multiples of 128 and 4. Then, the scales
    # (in float8_e4m3fn) are packed into an int32 for every 4 values. More:
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x
    round_up = lambda x, y: (x + y - 1) // y * y
    rounded_m = round_up(m, 128)
    scale_n = n // block_size
    rounded_n = round_up(scale_n, 4)
    output_scale = torch.empty(
        (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
    )

    scaled_fp4_ops.scaled_fp4_quant_sm1xxa(output, input, output_scale, input_global_scale)
    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale


def cutlass_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    assert a.is_cuda, "a must be a CUDA tensor"
    assert b.is_cuda, "b must be a CUDA tensor"
    assert block_scale_a.is_cuda, "block_scale_a must be a CUDA tensor"
    assert block_scale_b.is_cuda, "block_scale_b must be a CUDA tensor"
    assert alpha.is_cuda, "alpha must be a CUDA tensor"
    assert a.ndim == 2 and b.ndim == 2
    assert a.dtype in (
        torch.uint8,
    ), f"a.dtype needs to be uint8 but got {a.dtype}"
    assert b.dtype in (
        torch.uint8,
    ), f"b.dtype needs to be uint8 but got {b.dtype}"
    assert block_scale_a.dtype in (
        torch.float8_e4m3fn,
    ), f"block_scale_a.dtype needs to be float8_e4m3fn but got {block_scale_a.dtype}"
    assert block_scale_b.dtype in (
        torch.float8_e4m3fn,
    ), f"block_scale_b.dtype needs to be float8_e4m3fn but got {block_scale_b.dtype}"
    assert alpha.dtype == torch.float, f"alpha.dtype needs to be fp32 but got {alpha.dtype}"
    assert out_dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"out_dtype needs to be fp16 or bf16 but got {out_dtype}"
    m, n = a.shape[0], b.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    scaled_fp4_ops.cutlass_scaled_fp4_mm(out, a, b, block_scale_a, block_scale_b, alpha)
    return out

def reciprocal_approximate_ftz_tensor(x: torch.Tensor):
    y = torch.zeros_like(x)
    assert x.dtype == torch.float
    scaled_fp4_ops.reciprocal_approximate_ftz_tensor(x, y)
    return y