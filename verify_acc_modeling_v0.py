import torch
import nvfp.ops as ops
import nvfp.pseudo_quant as pseudo_quant
import random
import sys

from fprev_improved import nvfp4_gemm_simulation,nvfp4_pseudo_quantize
# ===================================================================================
# Helper: Mapping and Unpacking (Direct from Real Bits)
# ===================================================================================
def get_fp4_e2m1_table(device="cuda"):
    pos_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    neg_vals = [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    return torch.tensor(pos_vals + neg_vals, device=device, dtype=torch.float16)

def unpack_nvfp4_to_fp16(packed_uint8, original_shape):
    device = packed_uint8.device
    table = get_fp4_e2m1_table(device)
    # NVFP4 packing: [low_nibble, high_nibble]
    low = packed_uint8 & 0x0F
    high = (packed_uint8 >> 4) & 0x0F
    unpacked = torch.stack([low, high], dim=-1).view(original_shape)
    return table[unpacked.long()]

def hardware_reduction_4to1_pure_rz(v_list, W=26):
    """
    完全模拟 RZ 逻辑的 4-to-1 规约，不依赖 PyTorch 默认转换。
    """
    stacked = torch.stack(v_list, dim=0)
    max_val = torch.max(stacked.abs(), dim=0)[0]
    _, max_exp = torch.frexp(max_val)
    
    # 1. 对齐窗口
    scale = 2.0**(W - max_exp)
    
    # 2. 定点化累加 (模拟移位对齐时的截断)
    v_aligned_sum = torch.zeros_like(v_list[0], dtype=torch.float64)
    for v in v_list:
        v_aligned_sum += torch.trunc(v.double() * scale)
        
    # 3. 还原到浮点量级
    # 此时 summed_f64 在数值上已经是被截断过的了
    summed_f64 = v_aligned_sum / scale
    
    # 4. 【关键修正】：手动执行 FP32 的 RZ 转换，而不是用 .float()
    # 将 f64 强制转为 f32 时，如果不希望发生 RNE，我们先用位掩码抹除低位
    # 对于 FP32，尾数是 23 位。如果 W 超过了 24 (1个隐藏位 + 23个显式位)，
    # 我们需要确保超出 23 位的部分被截断。
    
    return to_float32_rz_bitwise(summed_f64)

def to_float32_rz_bitwise(tensor_f64):
    """
    模拟将高精度浮点数截断为 FP32 (RZ 模式)
    """
    # 这里的逻辑是：我们要保留 FP32 能表示的前 23 位尾数，抹除后面所有位
    # 为方便起见，先转成标准的 f32，然后通过位运算拉回 RZ 结果
    f32 = tensor_f64.to(torch.float32)
    
    # 找出那些因为 RNE 进位导致绝对值变大的点
    # 如果 f32 后的绝对值比原始 f64 绝对值大，说明它“进位”了，不符合 RZ
    mask = f32.abs() > tensor_f64.abs()
    
    # 对这些点，向 0 的方向找上一个可表示的 FP32 数
    res = f32.clone()
    if mask.any():
        res[mask] = torch.nextafter(f32[mask], torch.zeros_like(f32[mask]))
        
    return res


def emulation_scaled_fp4_mm_chunk(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, W=27):
    """
    分块版本的仿真函数。
    修正：增加了对 K > 64 时每 4 个 Group 进行一次硬件规约，随后进行全局累加的逻辑。
    """
    # 1. 预先处理 B 的相关变量 (B 在 M 轴循环中是共享的)
    G = K // 16
    num_4_groups = G // 4
    s_b = pseudo_quant.swizzled_to_linear_128_4(scale_b, N, G).to(torch.float32) # [N, G]
    val_b_fp16 = unpack_nvfp4_to_fp16(b_fp4, (N, K)) # [N, K]
    
    # 2. 准备 A 的线性化数据 (全量准备，循环内切片)
    s_a_all = pseudo_quant.swizzled_to_linear_128_4(scale_a, M, G).to(torch.float32) # [M, G]
    val_a_fp16_all = unpack_nvfp4_to_fp16(a_fp4, (M, K)) # [M, K]

    # 3. 对 M 维度进行分批处理 (Chunking)
    M_CHUNK = 32 
    summed_results_list = []

    for m_start in range(0, M, M_CHUNK):
        m_end = min(m_start + M_CHUNK, M)
        curr_chunk_size = m_end - m_start
        
        # --- 分批取出当前的 A 相关变量 ---
        val_a_chunk = val_a_fp16_all[m_start:m_end, :]  # [curr_chunk_size, K]
        s_a_chunk = s_a_all[m_start:m_end, :]           # [curr_chunk_size, G]
        
        # --- 执行 Stage 1: Inner MMA (Lossless FP16) ---
        # [m_chunk, K] @ [N, K].T -> [m_chunk, N, G]
        ps1_chunk = stage1_inner_mma_fp16(val_a_chunk, val_b_fp16)
        
        # --- 执行 Stage 2: Scaling ---
        combined_scales_chunk = s_a_chunk.unsqueeze(1) * s_b.unsqueeze(0) # [m_chunk, N, G]
        scaled_partials_chunk = ps1_chunk.float() * combined_scales_chunk # [m_chunk, N, G]
        
        # --- 执行 Stage 2: Hardware Reduction (每 4 个 group 进行规约) ---
        # 1. 拆分为 (m_chunk, N, G//4, 4)
        scaled_partials_4grouped = scaled_partials_chunk.view(curr_chunk_size, N, num_4_groups, 4)
        
        # 2. 准备 v_list (4路输入)
        v_list = [scaled_partials_4grouped[..., i] for i in range(4)]
        
        # 3. 硬件规约 (W-bit RZ) -> [m_chunk, N, G//4]
        summed_groups_chunk = hardware_reduction_4to1_pure_rz(v_list, W=W)

        # 4. 如果 G//4 > 1，说明 K > 64，对剩余的规约组进行普通累加
        if num_4_groups > 1:
            summed_chunk = summed_groups_chunk.sum(dim=-1)
        else:
            summed_chunk = summed_groups_chunk.squeeze(-1)
        
        # 存入列表
        summed_results_list.append(summed_chunk)
        
        # 显式清理当前块的中间变量
        del ps1_chunk, combined_scales_chunk, scaled_partials_chunk, summed_groups_chunk

    # 4. 拼接所有分块结果
    summed_result_all = torch.cat(summed_results_list, dim=0)
    
    # 5. 应用最终 Global Alpha 并转为 FP16
    alpha_val = alpha_tensor.item()
    final_res = (summed_result_all * alpha_val).to(torch.float16)

    # 彻底清理显存
    del val_a_fp16_all, val_b_fp16, s_a_all, s_b, summed_results_list
    return final_res

# ===================================================================================
# Stage 1 & 2 Modeling Logic
# ===================================================================================
def stage1_inner_mma_fp16(val_a, val_b):
    M, K = val_a.shape
    N, _ = val_b.shape
    K_groups = K // 16
    a_grouped = val_a.view(M, K_groups, 16).to(torch.float16)
    b_grouped = val_b.view(N, K_groups, 16).to(torch.float16)
    # [M, G, 16] x [N, G, 16] -> [M, N, G]
    partial_sum1 = torch.einsum('mgk,ngk->mgn', a_grouped, b_grouped).to(torch.float16)
    return partial_sum1.permute(0, 2, 1) # Return [M, N, G]

def emulation_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, W=25):
    # 1. Unpack bits to FP16
    val_a_fp16 = unpack_nvfp4_to_fp16(a_fp4, (M, K))
    val_b_fp16 = unpack_nvfp4_to_fp16(b_fp4, (N, K))

    # 2. Stage 1: Inner MMA (Lossless FP16)
    ps1 = stage1_inner_mma_fp16(val_a_fp16, val_b_fp16)

    # 3. Stage 2: Scaling & Summation (Modeled)
    # Restore swizzled shared scales to linear float32
    s_a = pseudo_quant.swizzled_to_linear_128_4(scale_a, M, K // 16).to(torch.float32)
    s_b = pseudo_quant.swizzled_to_linear_128_4(scale_b, N, K // 16).to(torch.float32)
    combined_scales = s_a.unsqueeze(1) * s_b.unsqueeze(0) # [M, N, G]
    
    # Core modeling: scaled sums in float32 to maintain intermediate magnitude
    
    # NOTE: 直到这一步，整个计算过程应该都是没有精度损失的，bit-level accurate 的
    scaled_partials = ps1.float() * combined_scales # [M, N, K // 16]
    # scaled_partials = ps1.double() * combined_scales.double()

    G = K // 16
    num_4_groups = G // 4
    scaled_partials_4grouped = scaled_partials.view(M, N, num_4_groups, 4)
    v_list = [scaled_partials_4grouped[..., i] for i in range(4)]
    # 4 groups of 4 partials each
    summed_groups = hardware_reduction_4to1_pure_rz(v_list, W=W)

    # group64 accumulation
    summed_result = summed_groups.sum(dim=-1) if num_4_groups > 1 else summed_groups.squeeze(-1)
    # summed_result = scaled_partials.sum(dim=-1)

    # v_list = [scaled_partials[..., i] for i in range(4)]
    # summed_result = hardware_reduction_4to1_pure_rz(v_list, W=W)
    
    # Final scaling by alpha (alpha is a CUDA tensor)
    alpha_val = alpha_tensor.item()
    return (summed_result * alpha_val).to(torch.float16)

# ===================================================================================
# Data Distribution Generators
# ===================================================================================
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
# Test Case Runner
# ===================================================================================
def run_test_case(iter_idx, M, N, K, dist_a, dist_b, sample_count=5, W=25):
    FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX = 6.0, 448.0
    a = get_random_tensor((M, K), dist_a)
    b = get_random_tensor((N, K), dist_b)

    # 1. Global Scale Prep (Standard NVFP flow)
    def get_gs(t):
        amax = torch.abs(t).max().to(torch.float32).item()
        return torch.tensor([FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / (amax if amax > 0 else 1.0)], 
                            device="cuda", dtype=torch.float32)

    a_gs = get_gs(a)
    b_gs = get_gs(b)
    # Alpha must be a CUDA tensor for the ops kernel
    alpha_val = 1.0 / (a_gs.item() * b_gs.item())
    alpha_tensor = torch.tensor([alpha_val], device="cuda", dtype=torch.float32)

    # 2. Hardware Execution
    a_fp4, scale_a = ops.scaled_fp4_quant(a, a_gs)
    b_fp4, scale_b = ops.scaled_fp4_quant(b, b_gs)
    
    # Run Real Kernel
    real_output = ops.cutlass_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, torch.float16)
    # 3. Modeling Emulation (Stage 1 & 2)
    modeling_res = emulation_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, W=W)
    # modeling_res = emulation_scaled_fp4_mm_chunk(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, W=W)

    # 4. Comparison
    # diff = (real_output - modeling_res).abs()
    diff = (real_output.float() - modeling_res.float()).abs()
    
    # If both are equal (including both inf, both -inf), diff should be 0
    matches = (real_output == modeling_res)
    diff[matches] = 0.0
    
    # Handle NaNs if both are NaN
    both_nan = real_output.isnan() & modeling_res.isnan()
    diff[both_nan] = 0.0
    
    # Treat any remaining NaNs (e.g. NaN vs Number) as mismatch (inf)
    diff[diff.isnan()] = float('inf')

    max_diff = diff.max().item()
    
    status = "SUCCESS" if max_diff == 0 else "MISMATCH"
    mismatch_details = []

    if status == "MISMATCH":
        # Find indices where diff is non-zero
        mismatch_indices = torch.nonzero(diff != 0, as_tuple=False)
        # Collect details for the first N samples
        for idx in mismatch_indices[:sample_count]:
            r, c = idx[0].item(), idx[1].item()
            mismatch_details.append({
                "idx": (r, c),
                "real": real_output[r, c].item(),
                "model": modeling_res[r, c].item(),
                "diff": diff[r, c].item()
            })
        # torch.save(a, f"tensor_a.pt")
        # torch.save(b, f"tensor_b.pt")
        # exit(0)
    
    return {
        "status": status,
        "max_diff": max_diff,
        "M": M, "N": N, "K": K,
        "dist_a": dist_a, "dist_b": dist_b,
        "mismatch_details": mismatch_details
    }

# ===================================================================================
# Main Entry
# ===================================================================================
def main():
    import time, os
    seed = int(time.time() * 1e6) ^ int.from_bytes(os.urandom(8), "little")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("seed =", seed)
    
    num_iterations = 100000
    distributions = ["normal", "uniform", "large", "outliers", "mixed_rows", "abs_large"]
    # distributions = ["outliers"]
    # distributions = ["normal", "large"]
    # distributions = ["normal"]
    
    dims_m = [128, 256, 1024, 2048, 4096]
    # dims_m = [128]
    dims_n = [128, 256, 1024, 2048, 4096]
    # dims_n = [128]
    # dims_k = [64, 256, 1024, 4096]
    dims_k = [128]
    # dims_k = [64]

    for w in range(27, 28):

        # Error reporting config
        NUM_SAMPLES_TO_PRINT = 5  # N: Number of mismatch values to print per test
        MAX_MISMATCH_TESTS = 20    # M: Max number of mismatch tests to print details for
        mismatch_print_count = 0
        
        print(f"Starting MMA Modeling Verification ({num_iterations} iterations)...")
        results = []
        
        for i in range(num_iterations):
            M, N, K = random.choice(dims_m), random.choice(dims_n), random.choice(dims_k)
            da, db = random.choice(distributions), random.choice(distributions)

            print(f"\rTest {i+1}/{num_iterations}: M={M:4}, N={N:4}, K={K:4}, A={da:10}, B={db:10} ... ", end="")
            sys.stdout.flush()
            
            try:
                res = run_test_case(i, M, N, K, da, db, sample_count=NUM_SAMPLES_TO_PRINT, W=w)
                results.append(res)
                print(f"{res['status']}")

                if res['status'] == "MISMATCH" and mismatch_print_count < MAX_MISMATCH_TESTS:
                    mismatch_print_count += 1
                    print(f"    >>> Mismatch Details (First {len(res['mismatch_details'])}):")
                    for d in res['mismatch_details']:
                        print(f"      Pos {d['idx']}: Real={d['real']:.6f} | Model={d['model']:.6f} | Diff={d['diff']:.6f}")

            except Exception as e:
                print(f"\nFATAL ERROR on Test {i+1}: {e}")
                import traceback
                traceback.print_exc() # 打印完整堆栈
                # sys.exit(1) # 立即退出程序

        # --- Summary ---
        successes = [r for r in results if r['status'] == "SUCCESS"]
        mismatches = [r for r in results if r['status'] == "MISMATCH"]

        print("\n" + "="*50)
        print(f"VERIFICATION SUMMARY")
        print(f"Total Tests:      {num_iterations}")
        print(f"Perfect Matches:  {len(successes)}")
        print(f"Mismatches:       {len(mismatches)}")
        print(f"Modeling W:      {w}")
        print("="*50)

        if mismatches:
            worst = max(mismatches, key=lambda x: x['max_diff'])
            print(f"Worst Mismatch Found:")
            print(f"  Max Abs Diff: {worst['max_diff']:.8f}")
            print(f"  Config:       M={worst['M']}, N={worst['N']}, K={worst['K']}")
            print(f"  Dists:        A={worst['dist_a']}, B={worst['dist_b']}")

if __name__ == "__main__":
    main()