"""
Core emulation classes: HardwareCore and MMAEngine
Fully aligned with verify_acc_modeling.py
"""
import torch
import nvfp.pseudo_quant as pseudo_quant


class HardwareCore:
    """负责具体的 RZ 规约硬件模拟逻辑"""
    
    @staticmethod
    def to_float32_rz_bitwise(tensor_f64):
        f32 = tensor_f64.to(torch.float32)
        mask = f32.abs() > tensor_f64.abs()
        res = f32.clone()
        if mask.any():
            res[mask] = torch.nextafter(f32[mask], torch.zeros_like(f32[mask]))
        return res

    @staticmethod
    def hardware_add_wbits(acc_fp32, new_val_wbits, W=25):
        """
        用于 Stage 4：FP32 accumulator + W bits new_val，输出 FP32。
        """
        _, acc_exp = torch.frexp(acc_fp32.abs())
        scale = 2.0**(W - acc_exp)
        acc_aligned = acc_fp32.double() * scale
        new_val_aligned = torch.trunc(new_val_wbits.double() * scale)
        sum_fixed = acc_aligned + new_val_aligned
        sum_wbits = sum_fixed / scale
        return HardwareCore.to_float32_rz_bitwise(sum_wbits)

    @staticmethod
    def hardware_reduction_4to1_pure_rz(v_list, W=25, output_fp32=True):
        """
        完全模拟 RZ 逻辑的 4-to-1 规约。
        """
        stacked = torch.stack(v_list, dim=0)
        max_val = torch.max(stacked.abs(), dim=0)[0]
        _, max_exp = torch.frexp(max_val)
        
        scale = 2.0**(W - max_exp)
        v_aligned_sum = torch.zeros_like(v_list[0], dtype=torch.float64)
        for v in v_list:
            v_aligned_sum += torch.trunc(v.double() * scale)
            
        summed_f64 = v_aligned_sum / scale
        
        if output_fp32:
            return HardwareCore.to_float32_rz_bitwise(summed_f64)
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
    def emulation_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, 
                                W_stage3=25, W_stage4=25):
        """
        NVFP4 MMA Accuracy Emulation (Blackwell Architecture)
        """
        from .utils import NVFP4Utils
        
        assert K % 16 == 0, "K must be multiple of 16 (group size)"
        assert (K // 16) % 4 == 0, "G must be multiple of 4 (mma.k64 blocks)"

        # STEP 1: Intra-Group Partial Sum (Lossless FP16)
        val_a_fp16 = NVFP4Utils.unpack_nvfp4_to_fp16(a_fp4, (M, K))
        val_b_fp16 = NVFP4Utils.unpack_nvfp4_to_fp16(b_fp4, (N, K))
        ps1 = MMAEngine.stage1_inner_mma_fp16(val_a_fp16, val_b_fp16)

        # STEP 2: Apply Shared Scales (Lossless F32)
        G = K // 16
        s_a = pseudo_quant.swizzled_to_linear_128_4(scale_a, M, G).to(torch.float32)
        s_b = pseudo_quant.swizzled_to_linear_128_4(scale_b, N, G).to(torch.float32)
        combined_scales = s_a.unsqueeze(1) * s_b.unsqueeze(0)
        scaled_partials = ps1.float() * combined_scales

        # STEP 3: Group -> MMA-K-Block Reduction (W_stage3-bit Emulation)
        num_4_groups = G // 4
        scaled_partials_4grouped = scaled_partials.view(M, N, num_4_groups, 4)
        v_list = [scaled_partials_4grouped[..., i] for i in range(4)]
        
        if num_4_groups == 1:
            summed_groups = HardwareCore.hardware_reduction_4to1_pure_rz(v_list, W=W_stage3, output_fp32=True)
        else:
            summed_groups = HardwareCore.hardware_reduction_4to1_pure_rz(v_list, W=W_stage3, output_fp32=False)

        # STEP 4: Inter-Block Accumulation (W_stage4-bit Emulation)
        if num_4_groups == 1:
            summed_result = summed_groups.squeeze(-1)
        else:
            acc = HardwareCore.to_float32_rz_bitwise(summed_groups[..., 0])
            for i in range(1, num_4_groups):
                acc = HardwareCore.hardware_add_wbits(acc, summed_groups[..., i], W=W_stage4)
            summed_result = acc

        # STEP 5: Final Scaling & Precision Cast
        alpha_val = alpha_tensor.item()
        return (summed_result * alpha_val).to(torch.float16)

    @staticmethod
    def emulation_scaled_fp4_mm_debug(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, 
                                      W_stage3=25, W_stage4=25):
        """
        Debug version that returns intermediate results for analysis.
        """
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
            summed_groups = HardwareCore.hardware_reduction_4to1_pure_rz(v_list, W=W_stage3, output_fp32=True)
        else:
            summed_groups = HardwareCore.hardware_reduction_4to1_pure_rz(v_list, W=W_stage3, output_fp32=False)
        
        debug_info = {
            "num_blocks": num_4_groups,
            "block_sums": [summed_groups[..., i] for i in range(num_4_groups)],
            "W_stage3": W_stage3,
            "W_stage4": W_stage4,
        }
        
        # STEP 4: Stage 4 with W_stage4
        if num_4_groups == 1:
            summed_result = summed_groups.squeeze(-1)
            stage4_sum_raw = None
        else:
            acc = HardwareCore.to_float32_rz_bitwise(summed_groups[..., 0])
            for i in range(1, num_4_groups):
                acc = HardwareCore.hardware_add_wbits(acc, summed_groups[..., i], W=W_stage4)
            summed_result = acc
            stage4_sum_raw = summed_result.clone()
        
        debug_info["stage4_sum_raw"] = stage4_sum_raw
        
        # STEP 5
        alpha_val = alpha_tensor.item()
        final_result = (summed_result * alpha_val).to(torch.float16)
        debug_info["alpha"] = alpha_val
        
        return final_result, debug_info
