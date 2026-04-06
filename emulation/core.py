"""
Core emulation classes: HardwareCore and MMAEngine
Fixed rounding strategy implementation
"""
import time

import torch
import nvfp.pseudo_quant as pseudo_quant

from .rounding import RoundStrategy, RoundingRegistry


def _init_profile_dict(enabled: bool):
    if not enabled:
        return None
    return {
        "preprocess_b_ms": 0.0,
        "preprocess_a_ms": 0.0,
        "stage1_ms": 0.0,
        "stage2_ms": 0.0,
        "stage3_ms": 0.0,
        "stage4_ms": 0.0,
        "concat_ms": 0.0,
        "stage5_ms": 0.0,
    }


def _profile_stage_start(profile):
    if profile is None:
        return None
    torch.cuda.synchronize()
    return time.perf_counter()


def _profile_stage_end(profile, key: str, t0):
    if profile is None:
        return
    torch.cuda.synchronize()
    profile[key] += (time.perf_counter() - t0) * 1000.0


def _assert_bit_exact_equal(ref: torch.Tensor, got: torch.Tensor, where: str):
    diff = (ref.float() - got.float()).abs()
    equal_mask = (ref == got)
    both_nan = ref.isnan() & got.isnan()
    valid_mask = equal_mask | both_nan
    diff[valid_mask] = 0.0
    diff[diff.isnan()] = float("inf")
    mismatch_count = int((diff != 0).sum().item())
    if mismatch_count != 0:
        raise RuntimeError(
            f"{where} mismatch: mismatch_count={mismatch_count}, max_diff={float(diff.max().item())}"
        )


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
            # Other strategies should be handled by RoundingRegistry
            raise NotImplementedError(f"Rounding {rounding} not implemented in to_float32_with_rounding")

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

    @staticmethod
    def hardware_reduction_4to1_grouped(v_grouped, W=25, output_fp32=True, rounding=RoundStrategy.RZ):
        """
        Vectorized Stage 3 reduction.

        Args:
            v_grouped: Tensor with shape [..., 4]
        Returns:
            Reduced tensor with shape [...]
        """
        assert v_grouped.shape[-1] == 4, "Last dimension must be 4 for 4-to-1 reduction"

        max_val = torch.max(v_grouped.abs(), dim=-1)[0]
        _, max_exp = torch.frexp(max_val)
        scale = 2.0**(W - max_exp)

        v_aligned_sum = torch.trunc(v_grouped.double() * scale.unsqueeze(-1)).sum(dim=-1)
        summed_f64 = v_aligned_sum / scale

        if output_fp32:
            return HardwareCore.to_float32_with_rounding(summed_f64, rounding)
        else:
            return summed_f64


class MMAEngine:
    """整合 MMA 仿真流程"""
    
    @staticmethod
    def stage1_inner_mma_fp16(val_a, val_b):
        M, K = val_a.shape
        N, _ = val_b.shape
        K_groups = K // 16
        a_grouped = val_a.view(M, K_groups, 16).to(torch.float16)
        b_grouped = val_b.view(N, K_groups, 16).to(torch.float16)
        partial_sum1 = torch.einsum('mgk,ngk->mgn', a_grouped, b_grouped).to(torch.float16)
        return partial_sum1.permute(0, 2, 1)

    @staticmethod
    def emulation_scaled_fp4_mm(
        a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, 
        W_stage3=25, W_stage4=25,
        stage3_rounding=RoundStrategy.RZ,
        stage4_rounding=RoundStrategy.RZ,
        m_chunk_size=128,
        return_profile=False,
    ):
        """NVFP4 MMA Accuracy Emulation with M-dimension chunking to avoid OOM"""
        from .utils import NVFP4Utils

        
        assert K % 16 == 0, "K must be multiple of 16 (group size)"
        assert (K // 16) % 4 == 0, "G must be multiple of 4 (mma.k64 blocks)"

        G = K // 16
        num_4_groups = G // 4
        profile = _init_profile_dict(return_profile)
        
        # Pre-process B (weight) - shared across all chunks
        t0 = _profile_stage_start(profile)
        val_b_fp16 = NVFP4Utils.unpack_nvfp4_to_fp16(b_fp4, (N, K))
        s_b = pseudo_quant.swizzled_to_linear_128_4(scale_b, N, G).to(torch.float32)
        _profile_stage_end(profile, "preprocess_b_ms", t0)
        
        # Pre-process A scales - will slice per chunk
        t0 = _profile_stage_start(profile)
        s_a_all = pseudo_quant.swizzled_to_linear_128_4(scale_a, M, G).to(torch.float32)
        val_a_fp16_all = NVFP4Utils.unpack_nvfp4_to_fp16(a_fp4, (M, K))
        _profile_stage_end(profile, "preprocess_a_ms", t0)
        
        # M-dimension chunking to avoid OOM
        summed_results_list = []
        
        for m_start in range(0, M, m_chunk_size):
            m_end = min(m_start + m_chunk_size, M)
            curr_chunk_size = m_end - m_start
            
            # Slice A for current chunk
            val_a_chunk = val_a_fp16_all[m_start:m_end, :]
            s_a_chunk = s_a_all[m_start:m_end, :]
            
            # STEP 1: Intra-Group Partial Sum (Lossless FP16)
            t0 = _profile_stage_start(profile)
            ps1_chunk = MMAEngine.stage1_inner_mma_fp16(val_a_chunk, val_b_fp16)
            _profile_stage_end(profile, "stage1_ms", t0)
            
            # STEP 2: Apply Shared Scales (Lossless F32)
            t0 = _profile_stage_start(profile)
            combined_scales_chunk = s_a_chunk.unsqueeze(1) * s_b.unsqueeze(0)
            scaled_partials_chunk = ps1_chunk.float() * combined_scales_chunk
            _profile_stage_end(profile, "stage2_ms", t0)
            
            # STEP 3: Group -> MMA-K-Block Reduction
            t0 = _profile_stage_start(profile)
            scaled_partials_4grouped = scaled_partials_chunk.view(curr_chunk_size, N, num_4_groups, 4)
            v_list = [scaled_partials_4grouped[..., i] for i in range(4)]
            
            if num_4_groups == 1:
                summed_groups = HardwareCore.hardware_reduction_4to1(
                    v_list, W=W_stage3, output_fp32=True, rounding=stage3_rounding
                )
            else:
                summed_groups = HardwareCore.hardware_reduction_4to1(
                    v_list, W=W_stage3, output_fp32=False, rounding=stage3_rounding
                )
            _profile_stage_end(profile, "stage3_ms", t0)
            
            # STEP 4: Inter-Block Accumulation
            t0 = _profile_stage_start(profile)
            if num_4_groups == 1:
                summed_chunk = summed_groups.squeeze(-1)
            else:
                summed_chunk = HardwareCore.inter_block_accumulate_sequential(
                    summed_groups=summed_groups,
                    W_stage4=W_stage4,
                    stage4_rounding=stage4_rounding,
                )
            _profile_stage_end(profile, "stage4_ms", t0)
            
            summed_results_list.append(summed_chunk)
            
            # Explicit cleanup to free memory
            del ps1_chunk, combined_scales_chunk, scaled_partials_chunk, scaled_partials_4grouped, v_list
            del summed_groups
        
        # Concatenate all chunks
        t0 = _profile_stage_start(profile)
        summed_result = torch.cat(summed_results_list, dim=0)
        _profile_stage_end(profile, "concat_ms", t0)
        
        # Cleanup pre-processed tensors
        del val_a_fp16_all, val_b_fp16, s_a_all, s_b
        
        # STEP 5: Final Scaling & Precision Cast
        t0 = _profile_stage_start(profile)
        alpha_val = alpha_tensor.item()
        final = (summed_result * alpha_val).to(torch.float16)
        _profile_stage_end(profile, "stage5_ms", t0)

        if return_profile:
            profile["total_profiled_ms"] = sum(profile.values())
            return final, profile
        return final

    @staticmethod
    def emulation_scaled_fp4_mm_optimized(
        a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K,
        W_stage3=25, W_stage4=25,
        stage3_rounding=RoundStrategy.RZ,
        stage4_rounding=RoundStrategy.RZ,
        m_chunk_size=128,
        return_profile=False,
    ):
        """
        Optimized PyTorch path with the same numerical semantics as emulation_scaled_fp4_mm.

        Key differences vs baseline:
        1. Vectorized Stage 3 reduction (no Python loop over 4 lanes)
        2. Avoid explicit combined_scales tensor materialization
        3. Pre-allocate output tensor to avoid list append + concat
        """
        from .utils import NVFP4Utils

        assert K % 16 == 0, "K must be multiple of 16 (group size)"
        assert (K // 16) % 4 == 0, "G must be multiple of 4 (mma.k64 blocks)"

        G = K // 16
        num_4_groups = G // 4
        profile = _init_profile_dict(return_profile)

        t0 = _profile_stage_start(profile)
        val_b_fp16 = NVFP4Utils.unpack_nvfp4_to_fp16(b_fp4, (N, K))
        s_b = pseudo_quant.swizzled_to_linear_128_4(scale_b, N, G).to(torch.float32)
        s_b_4 = s_b.view(1, N, num_4_groups, 4)
        _profile_stage_end(profile, "preprocess_b_ms", t0)

        t0 = _profile_stage_start(profile)
        s_a_all = pseudo_quant.swizzled_to_linear_128_4(scale_a, M, G).to(torch.float32)
        val_a_fp16_all = NVFP4Utils.unpack_nvfp4_to_fp16(a_fp4, (M, K))
        _profile_stage_end(profile, "preprocess_a_ms", t0)

        summed_result = torch.empty((M, N), device=val_a_fp16_all.device, dtype=torch.float32)

        for m_start in range(0, M, m_chunk_size):
            m_end = min(m_start + m_chunk_size, M)
            curr_chunk_size = m_end - m_start

            val_a_chunk = val_a_fp16_all[m_start:m_end, :]
            s_a_chunk = s_a_all[m_start:m_end, :]

            t0 = _profile_stage_start(profile)
            ps1_chunk = MMAEngine.stage1_inner_mma_fp16(val_a_chunk, val_b_fp16)
            _profile_stage_end(profile, "stage1_ms", t0)

            t0 = _profile_stage_start(profile)
            ps1_4 = ps1_chunk.view(curr_chunk_size, N, num_4_groups, 4).to(torch.float32)
            s_a_4 = s_a_chunk.view(curr_chunk_size, 1, num_4_groups, 4)

            scaled_partials_4 = ps1_4
            scaled_partials_4.mul_(s_a_4)
            scaled_partials_4.mul_(s_b_4)
            _profile_stage_end(profile, "stage2_ms", t0)

            t0 = _profile_stage_start(profile)
            if num_4_groups == 1:
                summed_groups = HardwareCore.hardware_reduction_4to1_grouped(
                    scaled_partials_4, W=W_stage3, output_fp32=True, rounding=stage3_rounding
                )
            else:
                summed_groups = HardwareCore.hardware_reduction_4to1_grouped(
                    scaled_partials_4, W=W_stage3, output_fp32=False, rounding=stage3_rounding
                )
            _profile_stage_end(profile, "stage3_ms", t0)

            t0 = _profile_stage_start(profile)
            if num_4_groups == 1:
                summed_chunk = summed_groups.squeeze(-1)
            else:
                summed_chunk = HardwareCore.inter_block_accumulate_sequential(
                    summed_groups=summed_groups,
                    W_stage4=W_stage4,
                    stage4_rounding=stage4_rounding,
                )
            _profile_stage_end(profile, "stage4_ms", t0)

            summed_result[m_start:m_end, :] = summed_chunk

        t0 = _profile_stage_start(profile)
        alpha_val = alpha_tensor.item()
        final = (summed_result * alpha_val).to(torch.float16)
        _profile_stage_end(profile, "stage5_ms", t0)

        if return_profile:
            profile["total_profiled_ms"] = sum(profile.values())
            return final, profile
        return final

    @staticmethod
    def emulation_scaled_fp4_mm_triton(
        a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K,
        W_stage3=25, W_stage4=25,
        stage3_rounding=RoundStrategy.RZ,
        stage4_rounding=RoundStrategy.RZ,
        m_chunk_size=128,
        triton_block_size=256,
        triton_use_stage3=True,
        triton_fuse_stage34=False,
        verify_stage3=False,
        verify_stage4=False,
        return_profile=False,
    ):
        """
        Triton-accelerated variant.

        Current scope:
        - Stage 1/2: optimized PyTorch path
        - Stage 3: optional Triton kernel
        - Stage 4: Triton kernel for RZ sequential accumulation
        - Optional Stage3+4 fused Triton path
        """
        from .utils import NVFP4Utils
        from .triton_stage4 import TRITON_STAGE4_AVAILABLE, stage4_accumulate_rz_triton
        from .triton_stage3 import (
            TRITON_STAGE3_AVAILABLE,
            stage3_reduce_4to1_rz_triton,
            stage34_fused_rz_triton,
        )

        use_stage3_triton = triton_use_stage3 and (stage3_rounding == RoundStrategy.RZ)
        use_stage34_fused_triton = triton_fuse_stage34 and (stage3_rounding == RoundStrategy.RZ) and (
            stage4_rounding == RoundStrategy.RZ
        )

        if use_stage3_triton and not TRITON_STAGE3_AVAILABLE:
            raise RuntimeError("Triton stage3 is unavailable in current environment.")
        if use_stage34_fused_triton and not TRITON_STAGE3_AVAILABLE:
            raise RuntimeError("Triton stage3 is unavailable for stage3+4 fusion.")
        if (not use_stage34_fused_triton) and stage4_rounding == RoundStrategy.RZ and not TRITON_STAGE4_AVAILABLE:
            raise RuntimeError("Triton stage4 is unavailable in current environment.")

        assert K % 16 == 0, "K must be multiple of 16 (group size)"
        assert (K // 16) % 4 == 0, "G must be multiple of 4 (mma.k64 blocks)"

        G = K // 16
        num_4_groups = G // 4
        profile = _init_profile_dict(return_profile)

        t0 = _profile_stage_start(profile)
        val_b_fp16 = NVFP4Utils.unpack_nvfp4_to_fp16(b_fp4, (N, K))
        s_b = pseudo_quant.swizzled_to_linear_128_4(scale_b, N, G).to(torch.float32)
        s_b_4 = s_b.view(1, N, num_4_groups, 4)
        _profile_stage_end(profile, "preprocess_b_ms", t0)

        t0 = _profile_stage_start(profile)
        s_a_all = pseudo_quant.swizzled_to_linear_128_4(scale_a, M, G).to(torch.float32)
        val_a_fp16_all = NVFP4Utils.unpack_nvfp4_to_fp16(a_fp4, (M, K))
        _profile_stage_end(profile, "preprocess_a_ms", t0)

        summed_result = torch.empty((M, N), device=val_a_fp16_all.device, dtype=torch.float32)

        for m_start in range(0, M, m_chunk_size):
            m_end = min(m_start + m_chunk_size, M)
            curr_chunk_size = m_end - m_start

            val_a_chunk = val_a_fp16_all[m_start:m_end, :]
            s_a_chunk = s_a_all[m_start:m_end, :]

            t0 = _profile_stage_start(profile)
            ps1_chunk = MMAEngine.stage1_inner_mma_fp16(val_a_chunk, val_b_fp16)
            _profile_stage_end(profile, "stage1_ms", t0)

            t0 = _profile_stage_start(profile)
            ps1_4 = ps1_chunk.view(curr_chunk_size, N, num_4_groups, 4).to(torch.float32)
            s_a_4 = s_a_chunk.view(curr_chunk_size, 1, num_4_groups, 4)
            scaled_partials_4 = ps1_4
            scaled_partials_4.mul_(s_a_4)
            scaled_partials_4.mul_(s_b_4)
            _profile_stage_end(profile, "stage2_ms", t0)

            if num_4_groups > 1 and use_stage34_fused_triton:
                t0 = _profile_stage_start(profile)
                summed_chunk = stage34_fused_rz_triton(
                    scaled_partials_4=scaled_partials_4,
                    w_stage3=W_stage3,
                    w_stage4=W_stage4,
                    block_size=triton_block_size,
                )
                if verify_stage4:
                    ref_groups = HardwareCore.hardware_reduction_4to1_grouped(
                        scaled_partials_4, W=W_stage3, output_fp32=False, rounding=stage3_rounding
                    )
                    ref_chunk = HardwareCore.inter_block_accumulate_sequential(
                        summed_groups=ref_groups,
                        W_stage4=W_stage4,
                        stage4_rounding=stage4_rounding,
                    )
                    _assert_bit_exact_equal(ref_chunk, summed_chunk, where="stage34_fused_triton")
                _profile_stage_end(profile, "stage3_ms", t0)
                # Stage4 is fused into stage3 timer for this mode.
                summed_result[m_start:m_end, :] = summed_chunk
                continue

            t0 = _profile_stage_start(profile)
            if num_4_groups == 1:
                summed_groups = HardwareCore.hardware_reduction_4to1_grouped(
                    scaled_partials_4, W=W_stage3, output_fp32=True, rounding=stage3_rounding
                )
            elif use_stage3_triton:
                summed_groups = stage3_reduce_4to1_rz_triton(
                    scaled_partials_4=scaled_partials_4,
                    w_stage3=W_stage3,
                    block_size=triton_block_size,
                )
                if verify_stage3:
                    ref_groups = HardwareCore.hardware_reduction_4to1_grouped(
                        scaled_partials_4, W=W_stage3, output_fp32=False, rounding=stage3_rounding
                    )
                    _assert_bit_exact_equal(ref_groups, summed_groups, where="stage3_triton")
            else:
                summed_groups = HardwareCore.hardware_reduction_4to1_grouped(
                    scaled_partials_4, W=W_stage3, output_fp32=False, rounding=stage3_rounding
                )
            _profile_stage_end(profile, "stage3_ms", t0)

            t0 = _profile_stage_start(profile)
            if num_4_groups == 1:
                summed_chunk = summed_groups.squeeze(-1)
            elif stage4_rounding == RoundStrategy.RZ:
                summed_chunk = stage4_accumulate_rz_triton(
                    summed_groups=summed_groups,
                    w_stage4=W_stage4,
                    block_size=triton_block_size,
                )
                if verify_stage4:
                    ref_chunk = HardwareCore.inter_block_accumulate_sequential(
                        summed_groups=summed_groups,
                        W_stage4=W_stage4,
                        stage4_rounding=stage4_rounding,
                    )
                    _assert_bit_exact_equal(ref_chunk, summed_chunk, where="stage4_triton")
            else:
                summed_chunk = HardwareCore.inter_block_accumulate_sequential(
                    summed_groups=summed_groups,
                    W_stage4=W_stage4,
                    stage4_rounding=stage4_rounding,
                )
            _profile_stage_end(profile, "stage4_ms", t0)

            summed_result[m_start:m_end, :] = summed_chunk

        t0 = _profile_stage_start(profile)
        alpha_val = alpha_tensor.item()
        final = (summed_result * alpha_val).to(torch.float16)
        _profile_stage_end(profile, "stage5_ms", t0)

        if return_profile:
            profile["total_profiled_ms"] = sum(profile.values())
            return final, profile
        return final

    @staticmethod
    def emulation_scaled_fp4_mm_debug(
        a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, 
        W_stage3=25, W_stage4=25,
        stage3_rounding=RoundStrategy.RZ,
        stage4_rounding=RoundStrategy.RZ
    ):
        """Debug version"""
        from .utils import NVFP4Utils
        
        assert K % 16 == 0, "K must be multiple of 16 (group size)"
        assert (K // 16) % 4 == 0, "G must be multiple of 4 (mma.k64 blocks)"

        # STEP 1-2
        val_a_fp16 = NVFP4Utils.unpack_nvfp4_to_fp16(a_fp4, (M, K))
        val_b_fp16 = NVFP4Utils.unpack_nvfp4_to_fp16(b_fp4, (N, K))
        ps1 = MMAEngine.stage1_inner_mma_fp16(val_a_fp16, val_b_fp16)
        
        G = K // 16
        s_a = pseudo_quant.swizzled_to_linear_128_4(scale_a, M, G).to(torch.float32)
        s_b = pseudo_quant.swizzled_to_linear_128_4(scale_b, N, G).to(torch.float32)
        combined_scales = s_a.unsqueeze(1) * s_b.unsqueeze(0)
        scaled_partials = ps1.float() * combined_scales

        # STEP 3: Stage 3 with W_stage3
        num_4_groups = G // 4
        scaled_partials_4grouped = scaled_partials.view(M, N, num_4_groups, 4)
        v_list = [scaled_partials_4grouped[..., i] for i in range(4)]
        
        if num_4_groups == 1:
            summed_groups = HardwareCore.hardware_reduction_4to1(
                v_list, W=W_stage3, output_fp32=True, rounding=stage3_rounding
            )
        else:
            summed_groups = HardwareCore.hardware_reduction_4to1(
                v_list, W=W_stage3, output_fp32=False, rounding=stage3_rounding
            )
        
        debug_info = {
            "num_blocks": num_4_groups,
            "block_sums": [summed_groups[..., i] for i in range(num_4_groups)],
            "W_stage3": W_stage3,
            "W_stage4": W_stage4,
            "stage3_rounding": stage3_rounding.value,
            "stage4_rounding": stage4_rounding.value,
        }
        
        # STEP 4
        if num_4_groups == 1:
            summed_result = summed_groups.squeeze(-1)
            stage4_sum_raw = None
        else:
            summed_result = HardwareCore.inter_block_accumulate_sequential(
                summed_groups=summed_groups,
                W_stage4=W_stage4,
                stage4_rounding=stage4_rounding,
            )
            stage4_sum_raw = summed_result.clone()
        
        debug_info["stage4_sum_raw"] = stage4_sum_raw
        
        # STEP 5
        alpha_val = alpha_tensor.item()
        final_result = (summed_result * alpha_val).to(torch.float16)
        debug_info["alpha"] = alpha_val
        
        return final_result, debug_info
