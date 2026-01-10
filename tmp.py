import torch
import nvfp.pseudo_quant as pseudo_quant
# FLOAT8_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
# FLOAT8_E4M3_MAX = 448.0
# print(FLOAT8_E4M3_EPS)
# print(FLOAT8_E4M3_MAX / FLOAT8_E4M3_EPS)
# for i in range(256):
#     a = torch.tensor(i, dtype=torch.uint8, device="cuda")
#     a = a.view(torch.float8_e4m3fn).float()
#     print(a, FLOAT8_E4M3_MAX / a)

def nvfp4_pseudo_quantize(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        pseudo_quant.quantize_linear_weight_to_nvfp4(x)
    )
    """Dequantize the fp4 tensor back to high precision."""
    # Two fp4 values are packed into one uint8.
    assert fp4_weight.dtype == torch.float4_e2m1fn_x2
    m, packed_k = fp4_weight.shape
    k = packed_k * 2
    tensor_f32 = pseudo_quant.unpack_fp4_bytes(fp4_weight)
    tensor_f32 = tensor_f32.reshape(m, k)
    weight_scale_interleaved = weight_scale_interleaved.view(torch.float8_e4m3fn)
    weight_scale_interleaved = pseudo_quant.swizzled_to_linear_128_4(
        weight_scale_interleaved, m, k
    )

    return tensor_f32, weight_scale_interleaved, weight_global_scale


def nvfp4_gemm_simulation(
    A_fp4_u8: torch.Tensor,
    B_fp4_u8: torch.Tensor,
    scale_A_swizzled: torch.Tensor,
    scale_B_swizzled: torch.Tensor,
    alpha: float,
    out_dtype: torch.dtype = torch.float16,
):
    """
    NVIDIA NVFP4 GEMM Simulation Kernel (aim to match CUTLASS SM120 bitwise)
    """

    assert A_fp4_u8.is_cuda and B_fp4_u8.is_cuda
    assert A_fp4_u8.dtype == torch.uint8 and B_fp4_u8.dtype == torch.uint8
    assert scale_A_swizzled.dtype == torch.float8_e4m3fn and scale_B_swizzled.dtype == torch.float8_e4m3fn

    M, packedK = A_fp4_u8.shape
    N, packedK2 = B_fp4_u8.shape
    assert packedK == packedK2
    K = packedK * 2
    assert K % 64 == 0

    # De-swizzle scales into linear [M, K/16] and [N, K/16]
    scale_A = pseudo_quant.swizzled_to_linear_128_4(scale_A_swizzled, M, K).view(torch.float8_e4m3fn).float()
    scale_B = pseudo_quant.swizzled_to_linear_128_4(scale_B_swizzled, N, K).view(torch.float8_e4m3fn).float()

    A_vals = pseudo_quant.unpack_fp4_bytes(A_fp4_u8.view(torch.float4_e2m1fn_x2)).reshape(M, K).float()
    B_vals = pseudo_quant.unpack_fp4_bytes(B_fp4_u8.view(torch.float4_e2m1fn_x2)).reshape(N, K).float()

    # Apply per-16 scaling factors
    A_vals = A_vals.view(M, K // 16, 16) * scale_A.unsqueeze(-1)
    B_vals = B_vals.view(N, K // 16, 16) * scale_B.unsqueeze(-1)

    # Reference mainloop semantics: sum over k
    acc = torch.zeros((M, N), device=A_fp4_u8.device, dtype=torch.float32)
    for kb in range(K // 16):
        acc += (A_vals[:, kb, :].float() @ B_vals[:, kb, :].float().T)

    return (acc * alpha).to(out_dtype)

import nvfp.ops as ops
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0
tensor_a = torch.load("tensor_a.pt")
tensor_b = torch.load("tensor_b.pt")

A_amax = torch.abs(tensor_a).max().to(torch.float32)
A_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / A_amax
A_fp4, scale_A_fp4 = ops.scaled_fp4_quant(tensor_a, A_global_scale)

B_amax = torch.abs(tensor_b).max().to(torch.float32)
B_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / B_amax
B_fp4, scale_B_fp4 = ops.scaled_fp4_quant(tensor_b, B_global_scale)

alpha = 1.0 / (A_global_scale * B_global_scale)
output = ops.cutlass_scaled_fp4_mm(
    A_fp4, B_fp4, scale_A_fp4, scale_B_fp4, alpha, torch.float16
)


sim_val = (
    nvfp4_gemm_simulation(
        A_fp4,
        B_fp4,
        scale_A_fp4,
        scale_B_fp4,
        alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha),
        torch.float16,
    )
)

print(sim_val)
print(output)

diff = sim_val - output
neq_mask = sim_val != output
neq_count = int(neq_mask.sum().item())
print('neq_count:', neq_count)

if neq_count > 0:
    diff_abs = diff.abs()
    max_abs = float(diff_abs.max().item())
    mean_abs_neq = float(diff_abs[neq_mask].mean().item())
    print('max_abs_diff:', max_abs)
    print('mean_abs_diff(only_neq):', mean_abs_neq)

    idx = torch.nonzero(neq_mask, as_tuple=False)
    topk = idx[:32]
    for i in range(topk.shape[0]):
        m, n = topk[i].tolist()
        sv = sim_val[m, n].item()
        ov = output[m, n].item()
        dv = (sim_val[m, n] - output[m, n]).item()
        print(f'[{i}] pos=({m},{n}) sim={sv} out={ov} diff={dv}')
else:
    print('all equal')