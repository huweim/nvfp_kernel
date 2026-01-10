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

def hardware_reduction_4to1(v_list, W=26):
    """
    模拟 4 个数同时输入加法规约树，仅在最终输出时执行一次位宽截断。
    v_list: [v0, v1, v2, v3] 每个都是 [M, N] 的 tensor
    """
    # 1. 将 4 个数堆叠，找到全局最大绝对值
    stacked = torch.stack(v_list, dim=0) # [4, M, N]
    max_val = torch.max(stacked.abs(), dim=0)[0]
    
    # 2. 获取全局最大指数作为对齐基准
    _, max_exp = torch.frexp(max_val)
    
    # 3. 确定物理窗口：所有数都对齐到 max_exp，保留 W 位
    # 模拟硬件内部一个位宽为 W 的固定点加法空间
    scale = 2.0**(W - max_exp)
    
    # 4. 对齐并截断（模拟数据进入加法器对齐器时丢失低位）
    # 硬件通常在对齐移位时直接丢弃掉出窗口的 bit
    v_aligned_sum = torch.zeros_like(v_list[0], dtype=torch.float64)
    for v in v_list:
        v_aligned_sum += torch.trunc(v.double() * scale)
        # v_aligned_sum += torch.round(v.double() * scale)
        
    # 5. 统一缩放回原量级（单次舍入/截断发生在这里）
    summed_result = (v_aligned_sum / scale).float()
    
    return summed_result

def hardware_accumulator_add(a, b, W=25):
    """
    模拟一个具有固定物理位宽 W 的浮点/定点累加器加法。
    W 是从最高有效位开始向下覆盖的总长度。
    """
    # 找到这两个数中绝对值最大的，确定对齐基准指数
    max_val = torch.max(a.abs(), b.abs())
    
    # 获取基准指数 (torch.frexp 返回的 mantissa 在 [0.5, 1) 之间)
    _, max_exp = torch.frexp(max_val)
    
    # 物理窗口缩放：将数值放大到最高位对齐 W 的量级
    # 举例：如果 max_val 约等于 2^20，W=25，则我们要保留到 2^(20-25) = 2^-5 的位置
    scale = 2.0**(W - max_exp)
    
    # 模拟对齐时的截断 (Truncation/RZ)。硬件为了节省面积，在位移对齐时通常直接截断。
    # 这里使用 double 确保中间计算不引入 Python/Torch 的额外 float32 误差
    a_aligned = torch.trunc(a.double() * scale) / scale
    b_aligned = torch.trunc(b.double() * scale) / scale
    
    return (a_aligned + b_aligned).float()

def to_fp16_limited_sticky(tensor_f32, sticky_width=9):
    """
    严谨模拟硬件 RNE (Round to Nearest Even)。
    
    参数:
        tensor_f32: 输入的 FP32 张量
        sticky_width: 硬件实际检查的 Sticky 位数（从 Guard 位往后数）。
                      FP32 转 FP16 理论最大值是 12。
                      如果设为 9，则忽略最后 3 位 (12-9=3)，这往往是硬件 Mismatch 的根源。
    """
    if tensor_f32.dtype != torch.float32:
        tensor_f32 = tensor_f32.float()

    # 1. 获取底层比特
    i32 = tensor_f32.view(torch.int32)

    # 2. 定义位位置
    # FP32: [S(1)] [E(8)] [M(23)]
    # FP16: [S(1)] [E(5)] [M(10)] -> 需要舍弃 13 位尾数
    LSB_BIT = 1 << 13      # FP16 尾数的最后一位在 FP32 中的位置
    GUARD_BIT = 1 << 12    # 决定舍入的第一位
    
    # 3. 构造有限的 Sticky 掩码
    # 标准 Sticky 掩码是 (1 << 12) - 1，即后 12 位
    # 我们根据 sticky_width 截断它
    full_sticky_mask = (1 << 12) - 1
    ignored_bits = 12 - sticky_width
    sticky_mask = full_sticky_mask & ~((1 << ignored_bits) - 1)

    # 4. 提取状态
    lsb = (i32 & LSB_BIT) != 0
    guard = (i32 & GUARD_BIT) != 0
    sticky = (i32 & sticky_mask) != 0

    # 5. RNE 进位判定 (Round Up 逻辑)
    # 进位条件：Guard 为 1 且 (Sticky 为 1 或 LSB 为 1)
    round_up = guard & (lsb | sticky)

    # 6. 执行进位并清理尾数
    # 如果 add_unit 导致尾数溢出，i32 加法会自动进位到指数位，这与硬件逻辑一致
    add_unit = round_up.to(torch.int32) << 13
    rounded_i32 = (i32 + add_unit) & ~full_sticky_mask & ~GUARD_BIT

    # 7. 转回 FP16
    return rounded_i32.view(torch.float32).to(torch.float16)

def stage1_inner_mma_fp16(val_a, val_b):
    M, K = val_a.shape
    N, _ = val_b.shape
    K_groups = K // 16
    a_grouped = val_a.view(M, K_groups, 16).to(torch.float16)
    b_grouped = val_b.view(N, K_groups, 16).to(torch.float16)
    # [M, G, 16] x [N, G, 16] -> [M, N, G]
    partial_sum1 = torch.einsum('mgk,ngk->mgn', a_grouped, b_grouped).to(torch.float16)
    return partial_sum1.permute(0, 2, 1) # Return [M, N, G]

def run_modeling_emulation_chunk(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K):
    # 1. 预先处理 B 的线性化，因为 B 在整个循环中是共享的
    K_groups = K // 16
    s_b = pseudo_quant.swizzled_to_linear_128_4(scale_b, N, K_groups).to(torch.float32) # [N, G]
    val_b_fp16 = unpack_nvfp4_to_fp16(b_fp4, (N, K)) # [N, K]
    
    # 2. 对 M 维度进行分批处理
    # 设定 M 轴的 Chunk 大小，可以根据显存动态调整，例如 32 或 64
    M_CHUNK = 32 
    summed_results_list = []
    
    # 将 scale_a 线性化
    s_a_all = pseudo_quant.swizzled_to_linear_128_4(scale_a, M, K_groups).to(torch.float32) # [M, G]
    # 反量化 A
    val_a_fp16_all = unpack_nvfp4_to_fp16(a_fp4, (M, K)) # [M, K]

    for m_start in range(0, M, M_CHUNK):
        m_end = min(m_start + M_CHUNK, M)
        
        # --- 分批取出当前的 A 相关变量 ---
        val_a_chunk = val_a_fp16_all[m_start:m_end, :]  # [m_chunk, K]
        s_a_chunk = s_a_all[m_start:m_end, :]           # [m_chunk, G]
        
        # --- 执行 Stage 1: Inner MMA (Lossless FP16) ---
        # 输入: [m_chunk, K], [N, K] -> 输出: [m_chunk, N, G]
        ps1_chunk = stage1_inner_mma_fp16(val_a_chunk, val_b_fp16)
        
        # --- 执行 Stage 2: Scaling & Summation ---
        # s_a_chunk: [m_chunk, 1, G] * s_b: [1, N, G] -> [m_chunk, N, G]
        combined_scales_chunk = s_a_chunk.unsqueeze(1) * s_b.unsqueeze(0)
        
        # 计算带缩放的中间值并求和
        # [m_chunk, N, G].float() * [m_chunk, N, G] -> [m_chunk, N]
        scaled_partials_chunk = ps1_chunk.float() * combined_scales_chunk

        # 4 group 之间 reduction 的精度高于 float32
        # scaled_partials_chunk = scaled_partials_chunk.reshape(M_CHUNK, N, -1, 4) # [m_chunk, N, G/4, 4]
        # scaled_partials_chunk = scaled_partials_chunk.double().sum(dim=-1)  # Sum over 4 elements -> [m_chunk, N, G/4]
        # scaled_partials_chunk = scaled_partials_chunk.float()


        summed_chunk = scaled_partials_chunk.sum(dim=-1)
        
        # 存入列表
        summed_results_list.append(summed_chunk)
        
        # 显式清理当前循环的临时大张量
        del ps1_chunk, combined_scales_chunk, scaled_partials_chunk

    # 3. 拼接并应用最终 alpha
    # [M, N]
    summed_result = torch.cat(summed_results_list, dim=0)
    
    alpha_val = alpha_tensor.item()

    # final_res = (summed_result * alpha_val).to(torch.float16)

    final_res = to_float16_rz_bitwise(summed_result * alpha_val)

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

def run_modeling_emulation(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, W=25):
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
    
    # print(s_a[416], s_b[8], combined_scales[416,8], ps1[416,8], alpha_tensor)
    # exit(0)
    # Core modeling: scaled sums in float32 to maintain intermediate magnitude
    
    # NOTE: 直到这一步，整个计算过程应该都是没有精度损失的，bit-level accurate 的
    scaled_partials = ps1.float() * combined_scales
    # scaled_partials = ps1.double() * combined_scales.double()

    # print(scaled_partials[416,8], alpha_tensor)

    # summed_result = scaled_partials.sum(dim=-1)

    # K=64 时，G=4，我们手动执行加法树
    # 假设硬件执行树状规约：(v0 + v1) + (v2 + v3)
    # v0 = scaled_partials[..., 0]
    # v1 = scaled_partials[..., 1]
    # v2 = scaled_partials[..., 2]
    # v3 = scaled_partials[..., 3]
    # W = 26  # 模拟的硬件累加器位宽 [25, 24, 23, 22, 21, 20, 19] 可
    # # 第一层加法
    # s01 = hardware_accumulator_add(v0, v1, W=W)
    # s23 = hardware_accumulator_add(v2, v3, W=W)
    # # 第二层加法（最终求和）
    # summed_result = hardware_accumulator_add(s01, s23, W=W)

    v_list = [scaled_partials[..., i] for i in range(4)]
    # summed_result = hardware_reduction_4to1(v_list, W=W)
    summed_result = hardware_reduction_4to1_pure_rz(v_list, W=W)
    
    # summed_result_manual = (scaled_partials[:,:,0] + scaled_partials[:,:,3]) + (scaled_partials[:,:,1] + scaled_partials[:,:,2])
    # print(summed_result[416,8], summed_result_manual[416,8], summed_result.shape, scaled_partials.shape)
    # exit(0)
    
    # Final scaling by alpha (alpha is a CUDA tensor)
    alpha_val = alpha_tensor.item()
    # print(bin((summed_result * alpha_val)[416,8].view(torch.uint32)))
    # print('manual', bin((summed_result * alpha_val)[416,8].to(torch.float16).view(torch.uint16)))
    # print(bin((scaled_partials.sum(dim=-1) * alpha_val)[416,8].to(torch.float16).view(torch.uint16)))
    # print(bin(to_fp16_limited_sticky((summed_result * alpha_val)[416,8], sticky_width=11).view(torch.uint16)))
    # exit(0)
    return (summed_result * alpha_val).to(torch.float16)
    # return to_fp16_limited_sticky(summed_result * alpha_val, sticky_width=11)
    # return to_float16_rz_bitwise(summed_result * alpha_val)
    # return to_float16_rz(summed_result * alpha_val)

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

    # a_load = torch.load('tensor_a.pt')
    # b_load = torch.load('tensor_b.pt')
    # M, K = a_load.shape
    # N, _ = b_load.shape
    
    # row_416 = a_load[416]
    # col_8 = b_load[8]
    # a_ = torch.zeros_like(a_load)
    # b_ = torch.zeros_like(b_load)
    # a_[416] = row_416
    # b_[8] = col_8

    # a = a_load
    # b = b_load

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
    modeling_res = run_modeling_emulation(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, W=W)

    # FPREV_IMPROVED

    # scale_b = pseudo_quant.swizzled_to_linear_128_4(scale_b, N, K).to(torch.float32) # [N, K]
    # b_fp4 = unpack_nvfp4_to_fp16(b_fp4, (N, K)) # [N, K]
    # scale_a = pseudo_quant.swizzled_to_linear_128_4(scale_a, M, K).to(torch.float32) # [M, K]
    # a_fp4 = unpack_nvfp4_to_fp16(a_fp4, (M, K)) # [M, K]

    # sim_A_fp4, sim_A_scale, sim_A_global = nvfp4_pseudo_quantize(a)
    # sim_B_fp4, sim_B_scale, sim_B_global = nvfp4_pseudo_quantize(b)
    # modeling_res = nvfp4_gemm_simulation(sim_A_fp4, sim_B_fp4, sim_A_scale, sim_B_scale, (1.0 / sim_A_global / sim_B_global).item())


    # 4. Comparison
    diff = (real_output - modeling_res).abs()
    
    # [Fix] Handle inf - inf = nan
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
    
    num_iterations = 10000
    # distributions = ["normal", "uniform", "large", "outliers", "mixed_rows", "abs_large"]
    distributions = ["outliers"]
    # distributions = ["normal", "large"]
    # distributions = ["normal"]
    
    dims_m = [128, 256, 1024, 2048]
    # dims_m = [128]
    dims_n = [128, 256, 1024, 2048]
    # dims_n = [128]
    # dims_k = [64, 256, 1024, 4096]
    dims_k = [64]

    for w in range(25, 32):

        # Error reporting config
        NUM_SAMPLES_TO_PRINT = 5  # N: Number of mismatch values to print per test
        MAX_MISMATCH_TESTS = 20    # M: Max number of mismatch tests to print details for
        mismatch_print_count = 0
        
        print(f"Starting MMA Modeling Verification ({num_iterations} iterations)...")
        results = []
        
        for i in range(num_iterations):
            M, N, K = random.choice(dims_m), random.choice(dims_n), random.choice(dims_k)
            da, db = random.choice(distributions), random.choice(distributions)
            
            print(f"\rTest {i+1}/{num_iterations}: K={K:4}, A={da:10}, B={db:10} ... ", end="")
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
                print(f"ERROR: {e}")

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