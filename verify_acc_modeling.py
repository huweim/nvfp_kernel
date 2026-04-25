"""
Bit-exact verification of NVFP4 emulation against real CUTLASS FP4 GEMM.

This script uses the canonical MMAEngine from emulation.core so that it stays
in sync with the library implementation.  Any drift between this script and the
emulation package is a bug.
"""
import random
import sys
import time
import os
import traceback

import torch
import nvfp.ops as ops
import nvfp.pseudo_quant as pseudo_quant

from emulation import MMAEngine, DataGenerator


# ===================================================================================
# TestRunner - drives the verification loop
# ===================================================================================
class TestRunner:
    @staticmethod
    def run_test_case(iter_idx, M, N, K, dist_a, dist_b, sample_count=5, W=25):
        FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX = 6.0, 448.0
        a = DataGenerator.get_random_tensor((M, K), dist_a)
        b = DataGenerator.get_random_tensor((N, K), dist_b)

        def get_gs(t):
            amax = torch.abs(t).max().to(torch.float32).item()
            return torch.tensor(
                [FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / (amax if amax > 0 else 1.0)],
                device="cuda",
                dtype=torch.float32,
            )

        a_gs, b_gs = get_gs(a), get_gs(b)
        alpha_val = 1.0 / (a_gs.item() * b_gs.item())
        alpha_tensor = torch.tensor([alpha_val], device="cuda", dtype=torch.float32)

        # Hardware execution (real CUTLASS FP4 GEMM)
        a_fp4, scale_a = ops.scaled_fp4_quant(a, a_gs)
        b_fp4, scale_b = ops.scaled_fp4_quant(b, b_gs)
        real_output = ops.cutlass_scaled_fp4_mm(
            a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, torch.float16
        )

        # Modeling emulation (canonical MMAEngine)
        modeling_res = MMAEngine.emulation_scaled_fp4_mm(
            a_fp4,
            b_fp4,
            scale_a,
            scale_b,
            alpha_tensor,
            M,
            N,
            K,
            W_stage3=W,
            W_stage4=W,
        )

        # Comparison
        diff = (real_output.float() - modeling_res.float()).abs()
        matches = real_output == modeling_res
        diff[matches] = 0.0
        both_nan = real_output.isnan() & modeling_res.isnan()
        diff[both_nan] = 0.0
        diff[diff.isnan()] = float("inf")

        max_diff = diff.max().item()
        status = "SUCCESS" if max_diff == 0 else "MISMATCH"
        mismatch_details = []

        if status == "MISMATCH":
            mismatch_indices = torch.nonzero(diff != 0, as_tuple=False)
            for idx in mismatch_indices[:sample_count]:
                r, c = idx[0].item(), idx[1].item()
                mismatch_details.append(
                    {
                        "idx": (r, c),
                        "real": real_output[r, c].item(),
                        "model": modeling_res[r, c].item(),
                        "diff": diff[r, c].item(),
                    }
                )

        return {
            "status": status,
            "max_diff": max_diff,
            "M": M,
            "N": N,
            "K": K,
            "dist_a": dist_a,
            "dist_b": dist_b,
            "mismatch_details": mismatch_details,
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
                res = TestRunner.run_test_case(
                    i, M, N, K, da, db, sample_count=NUM_SAMPLES_TO_PRINT, W=w
                )
                results.append(res)
                print(f"{res['status']}")

                if res["status"] == "MISMATCH" and mismatch_print_count < MAX_MISMATCH_TESTS:
                    mismatch_print_count += 1
                    print(f"    >>> Mismatch Details:")
                    for d in res["mismatch_details"]:
                        print(
                            f"      Pos {d['idx']}: Real={d['real']:.6f} | Model={d['model']:.6f} | Diff={d['diff']:.6f}"
                        )

            except Exception as e:
                print(f"\nFATAL ERROR on Test {i+1}: {e}")
                traceback.print_exc()

        # Summary
        mismatches = [r for r in results if r["status"] == "MISMATCH"]
        print("\n" + "=" * 50)
        print(f"VERIFICATION SUMMARY | W={w}")
        print(
            f"Total: {num_iterations}, Matches: {len(results)-len(mismatches)}, Mismatches: {len(mismatches)}"
        )
        print("=" * 50)


if __name__ == "__main__":
    main()
