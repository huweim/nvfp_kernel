"""
NVFP4 GEMM emulation engines.

  MMAEngine.emulation_scaled_fp4_mm_eton_fused   ETON-Fused, one Triton kernel (default; impl `beam_eton_fused`)
  MMAEngine.emulation_scaled_fp4_mm_fusednode    the same numerics, staged (impl `beam_fusednode`; test oracle)
  MMAEngine.emulation_scaled_fp4_mm              ETON-Base: the unfused staged PyTorch pipeline (impl `beam_base`,
                                                 the paper's Tab. IV "B." column; two-step W3/W4 model)
Implementations are selected through emulation/registry.py. Older variants (beam_234fusion, its bmm stage 1,
beam_naive_triton) are at git tag hpca2027-revision-1012.
"""
import torch
import nvfp.pseudo_quant as pseudo_quant

from .rounding import RoundStrategy


class HardwareCore:
    """负责硬件规约逻辑，支持可配置的 rounding strategy"""
    
    @staticmethod
    def to_float32_with_rounding(tensor_f64, rounding=RoundStrategy.RZ):
        """
        Convert float64 to float32 with specified rounding strategy.
        This is where rounding strategy is applied in hardware.
        """
        if rounding == RoundStrategy.RZ:
            # Round Toward Zero: truncate extra precision
            f32 = tensor_f64.to(torch.float32)
            mask = f32.abs() > tensor_f64.abs()
            res = f32.clone()
            if mask.any():
                res[mask] = torch.nextafter(f32[mask], torch.zeros_like(f32[mask]))
            return res
        elif rounding == RoundStrategy.RNE:
            # Round to Nearest Even: use PyTorch's round (banker's rounding) then cast
            # First convert to float32 (which rounds to nearest by default in PyTorch)
            return tensor_f64.to(torch.float32)
        else:
            raise NotImplementedError(
                f"Rounding {rounding} not implemented. "
                "Add handling in HardwareCore.to_float32_with_rounding."
            )

    @staticmethod
    def to_float32_rz_fast(tensor_f64):
        """
        Faster RZ-only float64 -> float32 cast.
        Semantics match to_float32_with_rounding(..., RZ).
        """
        f32 = tensor_f64.to(torch.float32)
        towards_zero = torch.nextafter(f32, torch.zeros_like(f32))
        return torch.where(f32.abs() > tensor_f64.abs(), towards_zero, f32)

    @staticmethod
    def hardware_add_wbits(acc_fp32, new_val_wbits, W=25, rounding=RoundStrategy.RZ):
        """
        Stage 4: FP32 accumulator + W bits new_val -> FP32 output.
        
        Hardware behavior:
        1. Align new_val to acc's exponent (truncate/RZ - fixed hardware behavior)
        2. Add in fixed-point domain
        3. Convert back with specified rounding strategy
        """
        # Step 1: Align to acc's exponent
        _, acc_exp = torch.frexp(acc_fp32.abs())
        scale = 2.0**(W - acc_exp)
        
        # acc is already FP32, convert to fixed-point without truncation
        acc_aligned = acc_fp32.double() * scale
        
        # new_val is W-bits, truncate to align with acc's exponent (fixed hardware behavior)
        new_val_aligned = torch.trunc(new_val_wbits.double() * scale)
        
        # Step 2: Accumulate in fixed-point
        sum_fixed = acc_aligned + new_val_aligned
        
        # Step 3: Convert back to float with specified rounding
        sum_f64 = sum_fixed / scale
        return HardwareCore.to_float32_with_rounding(sum_f64, rounding)

    @staticmethod
    def hardware_add_wbits_rz_fast(acc_fp32, new_val_wbits, W=25):
        """
        Faster Stage-4 add for Round-Toward-Zero path.
        Keeps the same sequential semantics as hardware_add_wbits(..., RZ).
        """
        _, acc_exp = torch.frexp(acc_fp32.abs())
        # Keep exact scale expression to match legacy numerical behavior.
        scale = 2.0**(W - acc_exp)

        acc_aligned = acc_fp32.to(torch.float64) * scale
        new_val_f64 = new_val_wbits if new_val_wbits.dtype == torch.float64 else new_val_wbits.to(torch.float64)
        new_val_aligned = torch.trunc(new_val_f64 * scale)
        sum_f64 = (acc_aligned + new_val_aligned) / scale
        return HardwareCore.to_float32_rz_fast(sum_f64)

    @staticmethod
    def inter_block_accumulate_sequential(summed_groups, W_stage4, stage4_rounding):
        """
        Sequential Stage-4 accumulation that preserves block order.
        """
        if stage4_rounding == RoundStrategy.RZ:
            acc = HardwareCore.to_float32_rz_fast(summed_groups[..., 0])
            for i in range(1, summed_groups.shape[-1]):
                acc = HardwareCore.hardware_add_wbits_rz_fast(acc, summed_groups[..., i], W=W_stage4)
            return acc

        acc = HardwareCore.to_float32_with_rounding(summed_groups[..., 0], stage4_rounding)
        for i in range(1, summed_groups.shape[-1]):
            acc = HardwareCore.hardware_add_wbits(
                acc, summed_groups[..., i], W=W_stage4, rounding=stage4_rounding
            )
        return acc

    @staticmethod
    def hardware_reduction_4to1(v_list, W=25, output_fp32=True, rounding=RoundStrategy.RZ):
        """
        Stage 3: 4-to-1 reduction.
        
        Hardware behavior:
        1. Find max value across all 4 elements
        2. Align all to max exponent and truncate (fixed RZ behavior)
        3. Sum in fixed-point domain
        4. If output_fp32: convert with specified rounding
           Else: keep as W-bit precision (float64 intermediate)
        """
        stacked = torch.stack(v_list, dim=0)
        max_val = torch.max(stacked.abs(), dim=0)[0]
        _, max_exp = torch.frexp(max_val)
        
        scale = 2.0**(W - max_exp)
        
        # Accumulate in fixed-point (truncate each element - fixed hardware RZ)
        v_aligned_sum = torch.zeros_like(v_list[0], dtype=torch.float64)
        for v in v_list:
            truncated = torch.trunc(v.double() * scale)  # RZ truncation
            v_aligned_sum += truncated
            
        summed_f64 = v_aligned_sum / scale
        
        if output_fp32:
            # Apply specified rounding when casting to FP32
            return HardwareCore.to_float32_with_rounding(summed_f64, rounding)
        else:
            # Keep as W-bit precision (intermediate for Stage 4)
            return summed_f64


class MMAEngine:
    """NVFP4 GEMM emulation (all entry points take the CUTLASS operand layout: packed e2m1 + swizzled ue4m3)."""

    @staticmethod
    def stage1_inner_mma_fp16_cuda_core(val_a, val_b):
        """Stage 1 of ETON-Base: 16-element group sums with broadcast multiply-adds on CUDA cores (fp16)."""
        M, K = val_a.shape
        N, _ = val_b.shape
        K_groups = K // 16
        a_grouped = val_a.view(M, K_groups, 16).to(torch.float16)
        b_grouped = val_b.view(N, K_groups, 16).to(torch.float16)
        partial_sum1 = torch.zeros((M, N, K_groups), device=val_a.device, dtype=torch.float16)
        for lane in range(16):
            partial_sum1.add_(a_grouped[:, None, :, lane] * b_grouped[None, :, :, lane])
        return partial_sum1

    @staticmethod
    def emulation_scaled_fp4_mm(
        a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K,
        W_stage3=36, W_stage4=36,
        stage3_rounding=RoundStrategy.RZ,
        stage4_rounding=RoundStrategy.RZ,
        m_chunk_size=128,
    ):
        """ETON-Base: stage 1 group sums (CUDA cores) -> scales -> W3-bit 4:1 reduction -> W4-bit sequential
        accumulation -> fp16(alpha * acc), chunked over M to bound memory."""
        from .utils import NVFP4Utils

        assert K % 64 == 0, "K must be a multiple of 64 (one mma.k64 per 4 groups)"
        G = K // 16
        num_4_groups = G // 4
        val_b = NVFP4Utils.unpack_nvfp4_to_fp16(b_fp4, (N, K))
        s_b = pseudo_quant.swizzled_to_linear_128_4(scale_b, N, G).to(torch.float32)
        s_a_all = pseudo_quant.swizzled_to_linear_128_4(scale_a, M, G).to(torch.float32)
        val_a_all = NVFP4Utils.unpack_nvfp4_to_fp16(a_fp4, (M, K))
        chunks = []
        for m0 in range(0, M, m_chunk_size):
            m1 = min(m0 + m_chunk_size, M)
            ps1 = MMAEngine.stage1_inner_mma_fp16_cuda_core(val_a_all[m0:m1], val_b)
            scaled = ps1.float() * (s_a_all[m0:m1].unsqueeze(1) * s_b.unsqueeze(0))
            v_list = [scaled.view(m1 - m0, N, num_4_groups, 4)[..., i] for i in range(4)]
            summed = HardwareCore.hardware_reduction_4to1(
                v_list, W=W_stage3, output_fp32=(num_4_groups == 1), rounding=stage3_rounding)
            if num_4_groups == 1:
                chunks.append(summed.squeeze(-1))
            else:
                chunks.append(HardwareCore.inter_block_accumulate_sequential(
                    summed_groups=summed, W_stage4=W_stage4, stage4_rounding=stage4_rounding))
            del ps1, scaled, v_list, summed
        return (torch.cat(chunks, dim=0) * alpha_tensor.item()).to(torch.float16)

    @staticmethod
    def emulation_scaled_fp4_mm_fusednode(
        a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K,
        F=35, m_chunk_size=512, triton_block_size=256,
    ):
        """
        Probed-template NVFP4 GEMM (see emulation/triton_fusednode.py). Stage 1 on fp16 tensor cores with fp32
        output (exact group sums) plus a nonzero-product count GEMM; then per k64 instruction one fused node
        (4 scale-anchored group terms + early C, RZ to 2^(emax-F), RZ to fp32); then fp16(alpha * acc).
        Same numerics as ETON-Fused (staged: the group sums go through memory); used as its test oracle.
        """
        from .utils import NVFP4Utils
        from .triton_fusednode import stage234_fusednode_rz_triton

        assert K % 64 == 0, "K must be a multiple of 64 (one mma.k64 per 4 groups)"
        G = K // 16
        s_b = pseudo_quant.swizzled_to_linear_128_4(scale_b, N, G).to(torch.float32)
        s_a_all = pseudo_quant.swizzled_to_linear_128_4(scale_a, M, G).to(torch.float32)
        val_a = NVFP4Utils.unpack_nvfp4_to_fp16(a_fp4, (M, K))
        val_b = NVFP4Utils.unpack_nvfp4_to_fp16(b_fp4, (N, K))
        # strided [G, 16, N] / [G, m, 16] views straight into bmm (no permute copies)
        b_g = val_b.view(N, G, 16).permute(1, 2, 0)
        b_ind = (val_b != 0).half().view(N, G, 16).permute(1, 2, 0)
        out = torch.empty((M, N), device=val_a.device, dtype=torch.float32)
        for m0 in range(0, M, m_chunk_size):
            m1 = min(m0 + m_chunk_size, M)
            a_c = val_a[m0:m1]
            a_g = a_c.view(m1 - m0, G, 16).permute(1, 0, 2)
            ps1 = torch.bmm(a_g, b_g, out_dtype=torch.float32)                         # [G, m, N] exact
            cnt = torch.bmm((a_c != 0).half().view(m1 - m0, G, 16).permute(1, 0, 2), b_ind)  # counts <= 16
            out[m0:m1] = stage234_fusednode_rz_triton(ps1.permute(1, 2, 0), cnt.permute(1, 2, 0),
                                                      s_a_all[m0:m1], s_b, F=F, block_size=triton_block_size)
            del ps1, cnt
        alpha_val = alpha_tensor.item()
        return (out * alpha_val).to(torch.float16)

    @staticmethod
    def emulation_scaled_fp4_mm_eton_fused(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, template=None,
                                           out_dtype=torch.float16, b_prepared=None):
        """
        ETON-Fused: the probed-template NVFP4 GEMM of emulation_scaled_fp4_mm_fusednode in ONE Triton kernel
        (tensor-core group sums + in-register fused node; see emulation/triton_eton_fused.py). template: a
        FusedTemplate, a probe json path, or None (sm_120a NVFP4). The default implementation.
        """
        from .triton_eton_fused import FusedTemplate, NVFP4_SM120, eton_fused_mm

        tmpl = NVFP4_SM120 if template is None else (
            template if isinstance(template, FusedTemplate) else FusedTemplate.from_probe(template))
        assert tmpl.scale_fmt == "ue4m3" and K % 64 == 0
        return eton_fused_mm(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, tmpl, out_dtype, b_prepared)
