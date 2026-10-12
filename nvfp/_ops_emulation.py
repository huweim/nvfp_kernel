import os

import torch


def _configured_emulation_impl() -> str:
    from emulation.registry import DEFAULT_IMPL, IMPLS
    impl = os.getenv("NVFP_EMULATION_IMPL", DEFAULT_IMPL).strip().lower()
    if impl not in IMPLS:
        raise ValueError(f"Invalid NVFP_EMULATION_IMPL={impl!r}; expected one of: {', '.join(sorted(IMPLS))}")
    return impl


def _configured_accum_bits() -> int:
    # NVFP_EMULATION_W3 / _W4 (legacy names, both = the accumulator width) default to the probed 36 bits
    w3, w4 = int(os.getenv("NVFP_EMULATION_W3", "36")), int(os.getenv("NVFP_EMULATION_W4", "36"))
    if w3 != w4:
        raise ValueError("the probed template has one accumulator width: NVFP_EMULATION_W3 must equal NVFP_EMULATION_W4")
    return w3


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


def cutlass_scaled_fp4_mm(a, b, block_scale_a, block_scale_b, alpha, out_dtype):
    """Emulated NVFP4 GEMM with the ops.cutlass_scaled_fp4_mm signature; impl from NVFP_EMULATION_IMPL."""
    from emulation import registry
    m, n, k = a.shape[0], b.shape[0], a.shape[1] * 2
    out = registry.run(_configured_emulation_impl(), a, b, block_scale_a, block_scale_b, alpha, m, n, k,
                       accum_bits=_configured_accum_bits(), out_dtype=out_dtype)
    return out.to(out_dtype)
