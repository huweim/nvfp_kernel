import os

import torch


def _configured_emulation_impl() -> str:
    impl = os.getenv("NVFP_EMULATION_IMPL", "beam_234fusion").strip().lower()
    valid = {
        "baseline",
        "beam_naive_triton",
        "beam_234fusion",
        "beam_234fusion_bmm",
    }
    if impl not in valid:
        valid_csv = ", ".join(sorted(valid))
        raise ValueError(
            f"Invalid NVFP_EMULATION_IMPL={impl!r}; expected one of: {valid_csv}"
        )
    return impl


def scaled_fp4_quant(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 using PyTorch emulation path.
    """
    from .pseudo_quant import linear_to_swizzled_128_4, pytorch_nvfp4_quantize

    assert input.is_cuda, "input must be a CUDA tensor"
    assert input_global_scale.is_cuda, "input_global_scale must be a CUDA tensor"
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1]).contiguous()
    m, n = input.shape
    block_size = 16
    device = input.device

    assert n % block_size == 0, f"last dim has to be multiple of 16, but got {n}."
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."
    assert input_global_scale.dtype == torch.float, (
        f"input_global_scale.dtype needs to be fp32 but got {input_global_scale.dtype}."
    )

    fp4, scale_fp8_linear = pytorch_nvfp4_quantize(input, input_global_scale)
    output = fp4.contiguous().view(torch.uint8)

    round_up = lambda x, y: (x + y - 1) // y * y
    scale_n = n // block_size
    rounded_m = round_up(m, 128)
    rounded_n = round_up(scale_n, 4)
    scale_linear = torch.zeros(
        (rounded_m, rounded_n), device=device, dtype=torch.float8_e4m3fn
    )
    scale_linear[:m, :scale_n] = scale_fp8_linear

    sw = linear_to_swizzled_128_4(scale_linear).contiguous()
    output_scale = sw.reshape(rounded_m, rounded_n)
    return output, output_scale


def cutlass_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    from emulation.core import MMAEngine

    assert a.is_cuda, "a must be a CUDA tensor"
    assert b.is_cuda, "b must be a CUDA tensor"
    assert block_scale_a.is_cuda, "block_scale_a must be a CUDA tensor"
    assert block_scale_b.is_cuda, "block_scale_b must be a CUDA tensor"
    assert alpha.is_cuda, "alpha must be a CUDA tensor"
    assert a.ndim == 2 and b.ndim == 2
    assert a.dtype in (torch.uint8,), f"a.dtype needs to be uint8 but got {a.dtype}"
    assert b.dtype in (torch.uint8,), f"b.dtype needs to be uint8 but got {b.dtype}"
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

    m, k_packed = a.shape
    n = b.shape[0]
    k = k_packed * 2
    impl = _configured_emulation_impl()
    if impl == "baseline":
        out = MMAEngine.emulation_scaled_fp4_mm(
            a, b, block_scale_a, block_scale_b, alpha, m, n, k
        )
    elif impl == "beam_naive_triton":
        out = MMAEngine.emulation_scaled_fp4_mm_triton(
            a,
            b,
            block_scale_a,
            block_scale_b,
            alpha,
            m,
            n,
            k,
            triton_use_stage3=True,
            triton_fuse_stage34=True,
        )
    elif impl == "beam_234fusion":
        out = MMAEngine.emulation_scaled_fp4_mm_triton_stage234_fused(
            a,
            b,
            block_scale_a,
            block_scale_b,
            alpha,
            m,
            n,
            k,
        )
    else:
        out = MMAEngine.emulation_scaled_fp4_mm_triton_stage234_fused_bmm(
            a,
            b,
            block_scale_a,
            block_scale_b,
            alpha,
            m,
            n,
            k,
        )
    return out.to(out_dtype)
