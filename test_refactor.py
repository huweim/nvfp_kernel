"""
Test script for refactored emulation code.
Fully aligned with verify_acc_modeling.py
"""
import sys
import time
import os
import random
import traceback

import torch
from nvfp import ops

from emulation import HardwareCore, MMAEngine, NVFP4Utils, DataGenerator, DataGenerator_Abs


# Constants from verify_acc_modeling.py
FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX = 6.0, 448.0


def get_global_scale(t):
    """Compute global scale for quantization."""
    amax = torch.abs(t).max().to(torch.float32).item()
    return torch.tensor([FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / (amax if amax > 0 else 1.0)], 
                        device="cuda", dtype=torch.float32)


class TestRunner:
    """
    测试执行器 - 与 verify_acc_modeling.py 完全一致
    """
    @staticmethod
    def run_test_case(iter_idx, M, N, K, dist_a, dist_b, sample_count=5,
                      W_stage3=36, W_stage4=36):
        """Run a single test case - aligned with verify_acc_modeling.py"""
        # Generate data
        a = DataGenerator.get_random_tensor((M, K), dist_a)
        b = DataGenerator.get_random_tensor((N, K), dist_b)

        # Quantize
        a_gs, b_gs = get_global_scale(a), get_global_scale(b)
        alpha_val = 1.0 / (a_gs.item() * b_gs.item())
        alpha_tensor = torch.tensor([alpha_val], device="cuda", dtype=torch.float32)

        # Hardware execution
        a_fp4, scale_a = ops.scaled_fp4_quant(a, a_gs)
        b_fp4, scale_b = ops.scaled_fp4_quant(b, b_gs)
        real_output = ops.cutlass_scaled_fp4_mm(a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, torch.float16)

        # Modeling emulation
        modeling_res = MMAEngine.emulation_scaled_fp4_mm(
            a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K,
            W_stage3=W_stage3, W_stage4=W_stage4
        )

        # Comparison - aligned with verify_acc_modeling.py
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
                mismatch_details.append(detail)
        
        return {
            "status": status, "max_diff": max_diff,
            "M": M, "N": N, "K": K,
            "dist_a": dist_a, "dist_b": dist_b,
            "mismatch_details": mismatch_details
        }


def main():
    """
    Main test entry - 与 verify_acc_modeling.py 的 main() 完全一致
    """
    # Set seed like original
    seed = int(time.time() * 1e6) ^ int.from_bytes(os.urandom(8), "little")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("seed =", seed)
    
    # Test configuration from original
    num_iterations = 10000
    distributions = ["normal", "uniform", "large", "outliers", "mixed_rows", "abs_large"]
    dims_m = [128, 256, 1024, 2048]
    dims_n = [128, 256, 1024, 2048, 4096]
    dims_k = [128, 256, 512, 1024]
    
    # RTX 5090 preset: W_stage3=36, W_stage4=36
    W_stage3 = 36
    w_stage4 = 36
    
    NUM_SAMPLES_TO_PRINT = 5
    MAX_MISMATCH_TESTS = 20
    mismatch_print_count = 0
    
    print(f"\n{'='*70}")
    print(f"Testing Stage 4 W={w_stage4} (Stage 3 W={W_stage3})")
    print(f"{'='*70}")
    print(f"Starting MMA Modeling Verification ({num_iterations} iterations)...")
    print(f"Distributions: {distributions}")
    print(f"Dims M: {dims_m}")
    print(f"Dims N: {dims_n}")
    print(f"Dims K: {dims_k}")
    print("="*70)
    
    results = []
    
    for i in range(num_iterations):
        M, N, K = random.choice(dims_m), random.choice(dims_n), random.choice(dims_k)
        da, db = random.choice(distributions), random.choice(distributions)
        
        print(f"\rTest {i+1}/{num_iterations}: K={K:4}, A={da:10}, B={db:10} ... ", end="")
        sys.stdout.flush()
        
        try:
            res = TestRunner.run_test_case(i, M, N, K, da, db,
                                            sample_count=NUM_SAMPLES_TO_PRINT,
                                            W_stage3=W_stage3, W_stage4=w_stage4)
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
    print("\n" + "="*70)
    print(f"VERIFICATION SUMMARY | Stage3 W={W_stage3}, Stage4 W={w_stage4}")
    print(f"Total: {num_iterations}, Matches: {len(results)-len(mismatches)}, Mismatches: {len(mismatches)}")
    print("="*70)
    
    if len(mismatches) == 0:
        print(f"*** PERFECT MATCH FOUND with Stage4 W={w_stage4}! ***")
        return 0
    else:
        print(f"Accuracy: {(len(results)-len(mismatches))/len(results)*100:.2f}%")
        return 1


if __name__ == "__main__":
    sys.exit(main())
