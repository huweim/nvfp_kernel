import torch
import nvfp.ops as ops
import nvfp.pseudo_quant as pseudo_quant
import random
import sys

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

def run_modeling_emulation(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K):
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
    scaled_partials = ps1.float() * combined_scales
    summed_result = scaled_partials.sum(dim=-1)
    
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
def run_test_case(iter_idx, M, N, K, dist_a, dist_b, sample_count=5):
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
    modeling_res = run_modeling_emulation(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K)

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
    torch.manual_seed(42)
    random.seed(42)
    
    num_iterations = 1000
    distributions = ["normal", "uniform", "large", "outliers", "mixed_rows", "abs_large"]
    
    dims_m = [128, 256, 1024, 2048]
    dims_n = [128, 256, 1024, 2048]
    dims_k = [64, 256, 1024, 4096]
    # dims_k = [64]

    # Error reporting config
    NUM_SAMPLES_TO_PRINT = 5  # N: Number of mismatch values to print per test
    MAX_MISMATCH_TESTS = 3    # M: Max number of mismatch tests to print details for
    mismatch_print_count = 0
    
    print(f"Starting MMA Modeling Verification ({num_iterations} iterations)...")
    results = []
    
    for i in range(num_iterations):
        M, N, K = random.choice(dims_m), random.choice(dims_n), random.choice(dims_k)
        da, db = random.choice(distributions), random.choice(distributions)
        
        print(f"\rTest {i+1}/{num_iterations}: K={K:4}, A={da:10}, B={db:10} ... ", end="")
        sys.stdout.flush()
        
        try:
            res = run_test_case(i, M, N, K, da, db, sample_count=NUM_SAMPLES_TO_PRINT)
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
    print("="*50)

    if mismatches:
        worst = max(mismatches, key=lambda x: x['max_diff'])
        print(f"Worst Mismatch Found:")
        print(f"  Max Abs Diff: {worst['max_diff']:.8f}")
        print(f"  Config:       M={worst['M']}, N={worst['N']}, K={worst['K']}")
        print(f"  Dists:        A={worst['dist_a']}, B={worst['dist_b']}")

if __name__ == "__main__":
    main()