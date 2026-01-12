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
    def hardware_reduction_4to1_pure_rz(v_list, W=26):
        """
        完全模拟 RZ 逻辑的 4-to-1 规约。
        v_list 包含 4 个 Tensor，规约在这些 Tensor 之间发生。
        """
        stacked = torch.stack(v_list, dim=0) # [4, ...]
        max_val = torch.max(stacked.abs(), dim=0)[0]
        _, max_exp = torch.frexp(max_val)
        
        scale = 2.0**(W - max_exp)
        v_aligned_sum = torch.zeros_like(v_list[0], dtype=torch.float64)
        for v in v_list:
            v_aligned_sum += torch.trunc(v.double() * scale)
            
        summed_f64 = v_aligned_sum / scale
        return HardwareCore.to_float32_rz_bitwise(summed_f64)

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
    def emulation_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, W=25):
        """
        NVFP4 MMA Accuracy Emulation (Blackwell Architecture)
        - Group (G16): Unit for shared scales (K=16).
        - MMA-K-Block (G64): Unit for MMA instruction reduction (4 Groups, K=64).
        """
        assert K % 16 == 0, "K must be multiple of 16 (group size)"
        assert (K // 16) % 4 == 0, "G must be multiple of 4 (mma.k64 blocks)"

        # STEP 1: Intra-Group Partial Sum (Lossless FP16)
        # Compute dot product for each Group (K=16). Output ps1 shape: [M, N, G].
        val_a_fp16 = NVFP4Utils.unpack_nvfp4_to_fp16(a_fp4, (M, K))
        val_b_fp16 = NVFP4Utils.unpack_nvfp4_to_fp16(b_fp4, (N, K))
        ps1 = MMAEngine.stage1_inner_mma_fp16(val_a_fp16, val_b_fp16) # [M, N, G]

        # STEP 2: Apply Shared Scales (Lossless F32)
        # Apply combined A/B shared scales to Group psums.
        G = K // 16
        s_a = pseudo_quant.swizzled_to_linear_128_4(scale_a, M, G).to(torch.float32)
        s_b = pseudo_quant.swizzled_to_linear_128_4(scale_b, N, G).to(torch.float32)
        combined_scales = s_a.unsqueeze(1) * s_b.unsqueeze(0) # [M, N, G]
        
        # Up to this step, the precision should all be lossless
        scaled_partials = ps1.float() * combined_scales # [M, N, G]

        # STEP 3: Group -> MMA-K-Block Reduction (W-bit Emulation)
        # Model 4-to-1 reduction inside mma.m32n8k64 (K=64) using W-bit.
        num_4_groups = G // 4
        scaled_partials_4grouped = scaled_partials.view(M, N, num_4_groups, 4)
        v_list = [scaled_partials_4grouped[..., i] for i in range(4)]
        summed_groups = HardwareCore.hardware_reduction_4to1_pure_rz(v_list, W=W)
        
        # STEP 4: Inter-Block Accumulation (Global K Reduction)
        # Final summation of MMA-K-Blocks. TODO: Model hardware main accumulator.
        summed_result = summed_groups.sum(dim=-1) if num_4_groups > 1 else summed_groups.squeeze(-1)
        
        # STEP 5: Final Scaling & Precision Cast
        # Apply global alpha and cast result back to torch.float16.
        alpha_val = alpha_tensor.item()
        return (summed_result * alpha_val).to(torch.float16)

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
            
            final_chunk_sum = summed_chunk_groups.sum(dim=-1) if num_4_groups > 1 else summed_chunk_groups.squeeze(-1)
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
            return torch.randn(shape, device=device, dtype=dtype)

# ===================================================================================
# Class 5: TestRunner - 驱动测试逻辑
# ===================================================================================
class TestRunner:
    @staticmethod
    def run_test_case(iter_idx, M, N, K, dist_a, dist_b, sample_count=5, W=25):
        FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX = 6.0, 448.0
        a = DataGenerator.get_random_tensor((M, K), dist_a)
        b = DataGenerator.get_random_tensor((N, K), dist_b)

        def get_gs(t):
            amax = torch.abs(t).max().to(torch.float32).item()
            return torch.tensor([FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / (amax if amax > 0 else 1.0)], 
                                device="cuda", dtype=torch.float32)

        a_gs, b_gs = get_gs(a), get_gs(b)
        alpha_val = 1.0 / (a_gs.item() * b_gs.item())
        alpha_tensor = torch.tensor([alpha_val], device="cuda", dtype=torch.float32)

        # Hardware execution
        a_fp4, scale_a = ops.scaled_fp4_quant(a, a_gs)
        b_fp4, scale_b = ops.scaled_fp4_quant(b, b_gs)
        real_output = ops.cutlass_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, torch.float16)
        
        # Modeling emulation
        modeling_res = MMAEngine.emulation_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, W=W)

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
                mismatch_details.append({
                    "idx": (r, c), "real": real_output[r, c].item(),
                    "model": modeling_res[r, c].item(), "diff": diff[r, c].item()
                })
        
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
    
    num_iterations = 100000
    distributions = ["normal", "uniform", "large", "outliers", "mixed_rows", "abs_large"]
    dims_m = [128, 256, 1024, 2048]
    dims_n = [128, 256, 1024, 2048, 4096]
    # dims_k = [128, 256, 1024, 2048]
    dims_k = [64]

    for w in range(27, 28):
        NUM_SAMPLES_TO_PRINT = 5
        MAX_MISMATCH_TESTS = 20
        mismatch_print_count = 0
        
        print(f"Starting MMA Modeling Verification ({num_iterations} iterations)...")
        results = []
        
        for i in range(num_iterations):
            M, N, K = random.choice(dims_m), random.choice(dims_n), random.choice(dims_k)
            da, db = random.choice(distributions), random.choice(distributions)
            
            print(f"\rTest {i+1}/{num_iterations}: K={K:4}, A={da:10}, B={db:10} ... ", end="")
            sys.stdout.flush()
            
            try:
                res = TestRunner.run_test_case(i, M, N, K, da, db, sample_count=NUM_SAMPLES_TO_PRINT, W=w)
                results.append(res)
                print(f"{res['status']}")

                if res['status'] == "MISMATCH" and mismatch_print_count < MAX_MISMATCH_TESTS:
                    mismatch_print_count += 1
                    print(f"    >>> Mismatch Details:")
                    for d in res['mismatch_details']:
                        print(f"      Pos {d['idx']}: Real={d['real']:.6f} | Model={d['model']:.6f} | Diff={d['diff']:.6f}")

            except Exception as e:
                print(f"\nFATAL ERROR on Test {i+1}: {e}")
                traceback.print_exc()

        # Summary
        mismatches = [r for r in results if r['status'] == "MISMATCH"]
        print("\n" + "="*50)
        print(f"VERIFICATION SUMMARY | W={w}")
        print(f"Total: {num_iterations}, Matches: {len(results)-len(mismatches)}, Mismatches: {len(mismatches)}")
        print("="*50)

if __name__ == "__main__":
    main()