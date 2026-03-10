import torch
import nvfp.ops as ops
import nvfp.pseudo_quant as pseudo_quant
import random
import sys
import time
import os
import traceback

# ===================================================================================
# Class 1: Utils - 负责底层映射与解包
# ===================================================================================
class NVFP4Utils:
    _TABLE_CACHE = {}

    @staticmethod
    def get_fp4_e2m1_table(device="cuda"):
        key = str(device)
        if key not in NVFP4Utils._TABLE_CACHE:
            pos_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
            neg_vals = [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
            NVFP4Utils._TABLE_CACHE[key] = torch.tensor(pos_vals + neg_vals, device=device, dtype=torch.float16)
        return NVFP4Utils._TABLE_CACHE[key]

    @staticmethod
    def unpack_nvfp4_to_fp16(packed_uint8, original_shape):
        device = packed_uint8.device
        table = NVFP4Utils.get_fp4_e2m1_table(device)
        low = packed_uint8 & 0x0F
        high = (packed_uint8 >> 4) & 0x0F
        unpacked = torch.stack([low, high], dim=-1).view(original_shape)
        return table[unpacked.long()]

# ===================================================================================
# Class 2: HardwareCore - 负责具体的 RZ 规约硬件模拟逻辑
# ===================================================================================
class HardwareCore:
    @staticmethod
    def to_float32_rz_bitwise(tensor_f64):
        f32 = tensor_f64.to(torch.float32)
        mask = f32.abs() > tensor_f64.abs()
        res = f32.clone()
        if mask.any():
            res[mask] = torch.nextafter(f32[mask], torch.zeros_like(f32[mask]))
        return res

    @staticmethod
    def to_float16_rz(tensor_f32):
        """
        将 FP32 转换为 FP16，使用 Round towards Zero (RZ)。
        与 PyTorch 默认的 RNE 不同。
        """
        # 方法：先转为 FP16 获取大致值，然后根据符号调整
        f16 = tensor_f32.to(torch.float16)
        
        # 对于 RZ，如果 FP32 的值在 FP16 表示的两个值之间，
        # 选择绝对值较小的那个（向零方向）
        
        # 检查是否有差异
        diff = tensor_f32.float() - f16.float()
        
        # 如果 tensor_f32 > 0 且 diff > 0，说明 f16 是较大的那个值
        # 需要减到更小的值（向零）
        mask_pos = (tensor_f32 > 0) & (diff > 0)
        mask_neg = (tensor_f32 < 0) & (diff < 0)
        
        result = f16.clone()
        if mask_pos.any():
            result[mask_pos] = torch.nextafter(f16[mask_pos], torch.zeros_like(f16[mask_pos]))
        if mask_neg.any():
            result[mask_neg] = torch.nextafter(f16[mask_neg], torch.zeros_like(f16[mask_neg]))
        
        return result

    @staticmethod
    def _apply_wbit_rz_truncation(value, max_abs_val, W):
        """
        对单个值按 W-bit 进行 RZ truncation。
        基于硬件 accumulator 行为：对齐到最大值的 exponent，保留 W bits。
        """
        _, max_exp = torch.frexp(max_abs_val)
        scale = 2.0**(W - max_exp)
        # RZ truncation: trunc towards zero
        truncated = torch.trunc(value.double() * scale) / scale
        return truncated

    @staticmethod
    def hardware_add_rz(acc, new_val, W=25):
        """
        模拟硬件 accumulator 的 running sum（带 W-bit RZ truncation）。
        用于 Stage 4 的 inter-block accumulation。
        
        Hardware 行为（与 hardware_reduction_4to1_pure_rz 一致）：
        1. 根据 max(acc_exp, new_val_exp) 对齐
        2. 将两者都 truncate 到 W bits（对齐到 max_exp）
        3. 累加 truncate 后的值
        4. 结果再进行 FP32 RZ truncation
        """
        # Step 1: 计算 max_exp（浮点加法对齐原则）
        _, acc_exp = torch.frexp(acc.abs())
        _, new_val_exp = torch.frexp(new_val.abs())
        max_exp = torch.maximum(acc_exp, new_val_exp)
        
        # Step 2: 基于 max_exp 计算 scale
        scale = 2.0**(W - max_exp)
        
        # Step 3: 将两者都对齐到 max_exp，并 truncate 到 W bits
        acc_aligned = torch.trunc(acc.double() * scale)  # 已经是整数
        new_val_aligned = torch.trunc(new_val.double() * scale)  # truncate 丢弃低位
        
        # Step 4: 累加（定点数相加）
        sum_fixed = acc_aligned + new_val_aligned
        
        # Step 5: 转回浮点，并进行 FP32 RZ truncation
        sum_f64 = sum_fixed / scale
        return HardwareCore.to_float32_rz_bitwise(sum_f64)

    @staticmethod
    def hardware_add_rne(acc, new_val, W=25):
        """
        模拟硬件 accumulator 的 running sum（带 W-bit RNE truncation）。
        用于 Stage 4 的 inter-block accumulation（测试用）。
        
        Hardware 行为（使用 RNE - Round to Nearest Even）：
        1. 根据 max(acc_exp, new_val_exp) 对齐
        2. 将两者都 round 到 W bits（使用 RNE）
        3. 累加 round 后的值
        4. 结果再进行 FP32 RNE truncation
        """
        # Step 1: 计算 max_exp（浮点加法对齐原则）
        _, acc_exp = torch.frexp(acc.abs())
        _, new_val_exp = torch.frexp(new_val.abs())
        max_exp = torch.maximum(acc_exp, new_val_exp)
        
        # Step 2: 基于 max_exp 计算 scale
        scale = 2.0**(W - max_exp)
        
        # Step 3: 将两者都对齐到 max_exp，并 round 到 W bits（使用 RNE）
        acc_scaled = acc.double() * scale
        new_val_scaled = new_val.double() * scale
        
        # RNE: round half to even (PyTorch round 使用 banker's rounding)
        acc_aligned = torch.round(acc_scaled)
        new_val_aligned = torch.round(new_val_scaled)
        
        # Step 4: 累加（定点数相加）
        sum_fixed = acc_aligned + new_val_aligned
        
        # Step 5: 转回浮点，并进行 FP32 RNE truncation
        sum_f64 = sum_fixed / scale
        return sum_f64.float()

    @staticmethod
    def hardware_reduction_4to1_pure_rz(v_list, W=25, output_fp32=True):
        """
        完全模拟 RZ 逻辑的 4-to-1 规约。
        v_list 包含 4 个 Tensor，规约在这些 Tensor 之间发生。
        
        Args:
            output_fp32: 如果 True，返回 FP32；如果 False，保持 W bits 精度
        """
        stacked = torch.stack(v_list, dim=0) # [4, ...]
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
            # 保持 W bits 精度（不 cast 到 FP32，保持为 float64 中间表示）
            return summed_f64

    @staticmethod
    def hardware_add_wbits(acc_fp32, new_val_wbits, W=25):
        """
        用于 Stage 4：FP32 accumulator + W bits new_val，输出 FP32。
        
        Args:
            acc_fp32: FP32 accumulator
            new_val_wbits: W bits 精度的新值（来自 Stage 3）
            W: Stage 4 的累加位宽
        
        Returns:
            FP32 accumulator（成为下一轮循环的 acc）
        """
        # 使用 acc 的 exponent 作为对齐基准（acc 是 FP32）
        _, acc_exp = torch.frexp(acc_fp32.abs())
        
        # 基于 acc 的 exponent 计算 scale
        scale = 2.0**(W - acc_exp)
        
        # acc 已经是 FP32，转换到定点（不 truncation，因为已经是 FP32）
        acc_aligned = acc_fp32.double() * scale
        
        # new_val 是 W bits，需要 truncation 到 acc 的 exponent
        new_val_aligned = torch.trunc(new_val_wbits.double() * scale)
        
        # 累加
        sum_fixed = acc_aligned + new_val_aligned
        
        # 结果转回浮点，然后 cast 到 FP32（成为下一轮循环的 acc）
        sum_wbits = sum_fixed / scale
        return HardwareCore.to_float32_rz_bitwise(sum_wbits)

# ===================================================================================
# Class 3: MMAEngine - 整合 MMA 仿真流程
# ===================================================================================
class MMAEngine:
    @staticmethod
    def stage1_inner_mma_fp16(val_a, val_b):
        M, K = val_a.shape
        N, _ = val_b.shape
        K_groups = K // 16
        a_grouped = val_a.view(M, K_groups, 16).to(torch.float16)
        b_grouped = val_b.view(N, K_groups, 16).to(torch.float16)
        partial_sum1 = torch.einsum('mgk,ngk->mgn', a_grouped, b_grouped).to(torch.float16)
        return partial_sum1.permute(0, 2, 1) # [M, N, G]

    @staticmethod
    def emulation_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, 
                                W_stage3=25, W_stage4=25):
        """
        NVFP4 MMA Accuracy Emulation (Blackwell Architecture)
        - Group (G16): Unit for shared scales (K=16).
        - MMA-K-Block (G64): Unit for MMA instruction reduction (4 Groups, K=64).
        
        Args:
            W_stage3: Bit width for intra-block (4-to-1) reduction
            W_stage4: Bit width for inter-block accumulation
        """
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
        # 关键修改：Stage 3 输出保持 W bits 精度，不立即 cast 到 FP32
        num_4_groups = G // 4
        scaled_partials_4grouped = scaled_partials.view(M, N, num_4_groups, 4)
        v_list = [scaled_partials_4grouped[..., i] for i in range(4)]
        
        # K=64 时：需要 cast 到 FP32（因为这是最终结果）
        # K>64 时：保持 W bits 精度，进入 Stage 4
        if num_4_groups == 1:
            summed_groups = HardwareCore.hardware_reduction_4to1_pure_rz(v_list, W=W_stage3, output_fp32=True)
        else:
            summed_groups = HardwareCore.hardware_reduction_4to1_pure_rz(v_list, W=W_stage3, output_fp32=False)

        # STEP 4: Inter-Block Accumulation (W_stage4-bit Emulation)
        if num_4_groups == 1:
            summed_result = summed_groups.squeeze(-1)
        else:
            # Stage 4: Block 0 cast to FP32，然后依次累加 W bits Block 1, 2, ...
            # 第一个 block cast 到 FP32
            acc = HardwareCore.to_float32_rz_bitwise(summed_groups[..., 0])
            
            # 后续的 blocks 保持 W bits，逐个累加
            # hardware_add_wbits 返回 FP32，成为下一轮循环的 acc
            for i in range(1, num_4_groups):
                acc = HardwareCore.hardware_add_wbits(acc, summed_groups[..., i], W=W_stage4)
            
            # 最终结果已经是 FP32（由 hardware_add_wbits 返回）
            summed_result = acc

        # summed_result = summed_groups.sum(dim=-1) if num_4_groups > 1 else summed_groups.squeeze(-1)
        
        # STEP 5: Final Scaling & Precision Cast
        # NOTE: 这里的 DType cast 也需要对齐；不过 k=64 都能通过，感觉这里应该没啥问题
        alpha_val = alpha_tensor.item()
        return (summed_result * alpha_val).to(torch.float16)
        # return HardwareCore.to_float16_rz(summed_result * alpha_val)

    @staticmethod
    def emulation_scaled_fp4_mm_debug(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, 
                                      W_stage3=25, W_stage4=25):
        """
        Debug version that returns intermediate results for analysis.
        """
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
            # Block 0 cast to FP32（作为 accumulator 初始值）
            acc = HardwareCore.to_float32_rz_bitwise(summed_groups[..., 0])
            
            # 后续的 blocks 保持 W bits，逐个累加
            # hardware_add_wbits 返回 FP32，成为下一轮循环的 acc
            for i in range(1, num_4_groups):
                acc = HardwareCore.hardware_add_wbits(acc, summed_groups[..., i], W=W_stage4)
            
            # 最终结果已经是 FP32（由 hardware_add_wbits 返回）
            summed_result = acc
            stage4_sum_raw = summed_result.clone()

        
        
        debug_info["stage4_sum_raw"] = stage4_sum_raw
        
        # STEP 5
        alpha_val = alpha_tensor.item()
        final_result = (summed_result * alpha_val).to(torch.float16)
        debug_info["alpha"] = alpha_val
        
        return final_result, debug_info
        # return HardwareCore.to_float16_rz(summed_result * alpha_val), debug_info

    @staticmethod
    def emulation_scaled_fp4_mm_chunk(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, W=27):
        G = K // 16
        s_b = pseudo_quant.swizzled_to_linear_128_4(scale_b, N, G).to(torch.float32)
        val_b_fp16 = NVFP4Utils.unpack_nvfp4_to_fp16(b_fp4, (N, K))
        s_a_all = pseudo_quant.swizzled_to_linear_128_4(scale_a, M, G).to(torch.float32)
        val_a_fp16_all = NVFP4Utils.unpack_nvfp4_to_fp16(a_fp4, (M, K))

        M_CHUNK = 32 
        summed_results_list = []
        num_4_groups = G // 4

        for m_start in range(0, M, M_CHUNK):
            m_end = min(m_start + M_CHUNK, M)
            curr_chunk_size = m_end - m_start
            
            val_a_chunk = val_a_fp16_all[m_start:m_end, :]
            s_a_chunk = s_a_all[m_start:m_end, :]
            
            ps1_chunk = MMAEngine.stage1_inner_mma_fp16(val_a_chunk, val_b_fp16)
            combined_scales_chunk = s_a_chunk.unsqueeze(1) * s_b.unsqueeze(0)
            scaled_partials_chunk = ps1_chunk.float() * combined_scales_chunk # [m_chunk, N, G]
            
            sp_chunk_4grouped = scaled_partials_chunk.view(curr_chunk_size, N, num_4_groups, 4)
            v_list = [sp_chunk_4grouped[..., i] for i in range(4)]
            summed_chunk_groups = HardwareCore.hardware_reduction_4to1_pure_rz(v_list, W=W)
            
            # Stage 4: Running accumulation with RZ truncation
            if num_4_groups == 1:
                final_chunk_sum = summed_chunk_groups.squeeze(-1)
            else:
                acc = summed_chunk_groups[..., 0]
                for i in range(1, num_4_groups):
                    acc = HardwareCore.hardware_add_rz(acc, summed_chunk_groups[..., i], W=25)
                    # acc = HardwareCore.hardware_add_rne(acc, summed_chunk_groups[..., i], W=25)
                final_chunk_sum = acc
            summed_results_list.append(final_chunk_sum)

        summed_result = torch.cat(summed_results_list, dim=0)
        alpha_val = alpha_tensor.item()
        return (summed_result * alpha_val).to(torch.float16)
    
# ===================================================================================
# Class 4: DataGenerator - 负责生成各种分布的测试数据
# ===================================================================================
class DataGenerator:
    @staticmethod
    def get_random_tensor(shape, dist_type, device="cuda", dtype=torch.float16):
        if dist_type == "normal":
            return torch.randn(shape, device=device, dtype=dtype)
        elif dist_type == "uniform":
            return (torch.rand(shape, device=device, dtype=dtype) * 2 - 1)
        elif dist_type == "large":
            return torch.randn(shape, device=device, dtype=dtype) * 100.0
        elif dist_type == "small":
            return torch.randn(shape, device=device, dtype=dtype) * 0.001
        elif dist_type == "outliers":
            t = torch.randn(shape, device=device, dtype=dtype)
            mask = torch.rand(shape, device=device) < 0.01
            t[mask] *= 50.0
            return t
        elif dist_type == "mixed_rows":
            t = torch.randn(shape, device=device, dtype=dtype)
            scale = torch.exp(torch.randn(shape[0], 1, device=device) * 2)
            return t * scale.to(dtype)
        elif dist_type == "abs_large":
             return torch.abs(torch.randn(shape, device=device, dtype=dtype)) * 100.0
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
class DataGenerator_Abs:
    @staticmethod
    def get_random_tensor(shape, dist_type, device="cuda", dtype=torch.float16):
        # DEBUG: 只生成正数，便于分析
        if dist_type == "normal":
            return torch.abs(torch.randn(shape, device=device, dtype=dtype))
        elif dist_type == "uniform":
            return torch.rand(shape, device=device, dtype=dtype)  # 0 to 1, positive only
        elif dist_type == "large":
            return torch.abs(torch.randn(shape, device=device, dtype=dtype)) * 100.0
        elif dist_type == "small":
            return torch.abs(torch.randn(shape, device=device, dtype=dtype)) * 0.001
        elif dist_type == "outliers":
            t = torch.abs(torch.randn(shape, device=device, dtype=dtype))
            mask = torch.rand(shape, device=device) < 0.01
            t[mask] *= 50.0
            return t
        elif dist_type == "mixed_rows":
            t = torch.abs(torch.randn(shape, device=device, dtype=dtype))
            scale = torch.exp(torch.randn(shape[0], 1, device=device) * 2)
            return t * scale.to(dtype)
        elif dist_type == "abs_large":
             return torch.abs(torch.randn(shape, device=device, dtype=dtype)) * 100.0
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
        
# ===================================================================================
# Class 5: TestRunner - 驱动测试逻辑
# ===================================================================================
class TestRunner:
    @staticmethod
    def run_test_case(iter_idx, M, N, K, dist_a, dist_b, sample_count=5, 
                      W_stage3=25, W_stage4=25, debug_mismatch=False):
        FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX = 6.0, 448.0
        a = DataGenerator.get_random_tensor((M, K), dist_a)
        b = DataGenerator.get_random_tensor((N, K), dist_b)

        # a = DataGenerator_Abs.get_random_tensor((M, K), dist_a)
        # b = DataGenerator_Abs.get_random_tensor((N, K), dist_b)

        def get_gs(t):
            amax = torch.abs(t).max().to(torch.float32).item()
            return torch.tensor([FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / (amax if amax > 0 else 1.0)], 
                                device="cuda", dtype=torch.float32)

        a_gs, b_gs = get_gs(a), get_gs(b)
        alpha_val = 1.0 / (a_gs.item() * b_gs.item())
        # alpha_val = 1.0
        alpha_tensor = torch.tensor([alpha_val], device="cuda", dtype=torch.float32)

        # Hardware execution
        a_fp4, scale_a = ops.scaled_fp4_quant(a, a_gs)
        b_fp4, scale_b = ops.scaled_fp4_quant(b, b_gs)
        real_output = ops.cutlass_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, torch.float16)
        
        # Modeling emulation with debug info
        if debug_mismatch:
            modeling_res, debug_info = MMAEngine.emulation_scaled_fp4_mm_debug(
                a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, 
                W_stage3=W_stage3, W_stage4=W_stage4
            )
        else:
            modeling_res = MMAEngine.emulation_scaled_fp4_mm(
                a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, 
                W_stage3=W_stage3, W_stage4=W_stage4
            )
            debug_info = None

        # Comparison
        diff = (real_output.float() - modeling_res.float()).abs()
        matches = (real_output == modeling_res)
        diff[matches] = 0.0
        both_nan = real_output.isnan() & modeling_res.isnan()
        diff[both_nan] = 0.0
        diff[diff.isnan()] = float('inf')

        max_diff = diff.max().item()
        status = "SUCCESS" if max_diff == 0 else "MISMATCH"
        mismatch_details = []

        if status == "MISMATCH":
            mismatch_indices = torch.nonzero(diff != 0, as_tuple=False)
            for idx in mismatch_indices[:sample_count]:
                r, c = idx[0].item(), idx[1].item()
                detail = {
                    "idx": (r, c), "real": real_output[r, c].item(),
                    "model": modeling_res[r, c].item(), "diff": diff[r, c].item()
                }
                # Add debug info if available
                if debug_info is not None and K > 64:
                    detail["debug"] = {
                        "block_sums": [debug_info["block_sums"][b][r, c].item() for b in range(debug_info["num_blocks"])],
                        "alpha": alpha_val,
                        "stage4_sum_raw": debug_info["stage4_sum_raw"][r, c].item() if "stage4_sum_raw" in debug_info else None,
                    }
                    
                    # Note: Getting Real Hardware block sums requires re-quantizing
                    # each block separately, which changes the block-wise scales.
                    # This is complex due to swizzled format, so we skip it for now.
                    # Instead, we can infer from the mismatch pattern.
                mismatch_details.append(detail)
        
        return {
            "status": status, "max_diff": max_diff,
            "M": M, "N": N, "K": K,
            "dist_a": dist_a, "dist_b": dist_b,
            "mismatch_details": mismatch_details
        }

# ===================================================================================
# Main Entry Point
# ===================================================================================
def main():
    seed = int(time.time() * 1e6) ^ int.from_bytes(os.urandom(8), "little")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("seed =", seed)
    
    num_iterations = 10000
    # distributions = ["normal", "uniform", "large", "outliers", "mixed_rows", "abs_large"]
    distributions = ["outliers"]
    dims_m = [128, 256, 1024, 2048]
    dims_n = [128, 256, 1024, 2048, 4096]
    dims_k = [128, 256, 512, 1024]
    # dims_k = [64]
    
    # Stage 3 W is fixed (verified correct for K=64), iterate Stage 4 W
    # W_stage3 = 35
    
    for W_stage3 in range(34, 35):
        for w_stage4 in range(28, 35):  # Test W from 20 to 36
            NUM_SAMPLES_TO_PRINT = 5
            MAX_MISMATCH_TESTS = 20
            mismatch_print_count = 0
            
            print(f"\n{'='*70}")
            print(f"Testing Stage 4 W={w_stage4} (Stage 3 W={W_stage3})")
            print(f"{'='*70}")
            print(f"Starting MMA Modeling Verification ({num_iterations} iterations)...")
            results = []
            
            for i in range(num_iterations):
                M, N, K = random.choice(dims_m), random.choice(dims_n), random.choice(dims_k)
                da, db = random.choice(distributions), random.choice(distributions)
                
                print(f"\rTest {i+1}/{num_iterations}: K={K:4}, A={da:10}, B={db:10} ... ", end="")
                sys.stdout.flush()
                
                try:
                    # Enable debug mode for K > 64 to capture intermediate results
                    debug_mode = (K > 64)
                    res = TestRunner.run_test_case(i, M, N, K, da, db, 
                                                    sample_count=NUM_SAMPLES_TO_PRINT, 
                                                    W_stage3=W_stage3, W_stage4=w_stage4,
                                                    debug_mismatch=debug_mode)
                    results.append(res)
                    print(f"{res['status']}")

                    if res['status'] == "MISMATCH" and mismatch_print_count < MAX_MISMATCH_TESTS:
                        mismatch_print_count += 1
                        print(f"    >>> Mismatch Details:")
                        for d in res['mismatch_details']:
                            print(f"      Pos {d['idx']}: Real={d['real']:.6f} | Model={d['model']:.6f} | Diff={d['diff']:.6f}")
                            # Print detailed debug info if available
                            if 'debug' in d and d['debug'] is not None:
                                dbg = d['debug']
                                print(f"        [Debug] Model Block sums (raw, before alpha):")
                                for b_idx, b_sum in enumerate(dbg['block_sums']):
                                    print(f"          Block {b_idx}: {b_sum:.6f}")
                                print(f"        [Debug] Stage 4 sum (raw): {dbg['stage4_sum_raw']:.6f}")
                                print(f"        [Debug] Alpha: {dbg['alpha']:.10f}")
                                print(f"        [Debug] Expected (raw * alpha): {dbg['stage4_sum_raw'] * dbg['alpha']:.6f}")

                except Exception as e:
                    print(f"\nFATAL ERROR on Test {i+1}: {e}")
                    traceback.print_exc()

            # Summary
            mismatches = [r for r in results if r['status'] == "MISMATCH"]
            print("\n" + "="*50)
            print(f"VERIFICATION SUMMARY | Stage3 W={W_stage3}, Stage4 W={w_stage4}")
            print(f"Total: {num_iterations}, Matches: {len(results)-len(mismatches)}, Mismatches: {len(mismatches)}")
            print("="*50)
            
            # Early exit if perfect match found
            if len(mismatches) == 0:
                print(f"*** PERFECT MATCH FOUND with Stage4 W={w_stage4}! ***")
                break

if __name__ == "__main__":
    main()