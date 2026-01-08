import torch
import nvfp.ops as ops
import nvfp.pseudo_quant as pseudo_quant
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(23)
torch.cuda.manual_seed(23)
torch.cuda.manual_seed_all(23)

# Input tensors
a = torch.randn(128, 128, dtype=torch.float16, device="cuda")

# Create tensor b with the last dimension having first 32 elements as 1 and last 32 as 0
b = torch.randn(128, 128, dtype=torch.float16, device="cuda")


# Quantize weight to NVFP4
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0
b_amax = torch.abs(b).max().to(torch.float32)
b_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / b_amax

# Quantize weight
b_fp4, scale_b_fp4 = ops.scaled_fp4_quant(b, b_global_scale)

# Quantize activation
a_amax = torch.abs(a).max().to(torch.float32)
a_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax
a_fp4, scale_a_fp4 = ops.scaled_fp4_quant(a, a_global_scale)

# Compute GEMM with quantized tensors
alpha = 1.0 / (a_global_scale * b_global_scale)
output = ops.cutlass_scaled_fp4_mm(
    a_fp4, b_fp4, scale_a_fp4, scale_b_fp4, alpha, torch.float16
)

print("Real quantization output shape:", output.shape)
print("Real quantization output sample:")
print(output[:3, :3])

pseudu_output = (
    pseudo_quant.nvfp4_pseudo_quantize(a) @ pseudo_quant.nvfp4_pseudo_quantize(b).T
).half()

print("\nPseudo quantization output shape:", pseudu_output.shape)
print("Pseudo quantization output sample:")
print(pseudu_output[:3, :3])

# Calculate differences
diff = output - pseudu_output
non_zero_mask = diff != 0
total_errors = non_zero_mask.float().sum()

print(f"\nTotal number of errors: {total_errors}")

# Extract exact rows and columns with errors
if total_errors > 0:
    # Get indices of all errors
    error_indices = torch.nonzero(non_zero_mask, as_tuple=False)

    print(f"\nError analysis:")
    print(f"First 10 error locations (row, col):")
    for i in range(min(10, len(error_indices))):
        row, col = error_indices[i]
        real_val = output[row, col].item()
        pseudo_val = pseudu_output[row, col].item()
        error_val = diff[row, col].item()
        print(
            f"  ({row.item():3d}, {col.item():3d}): real={real_val:.6f}, pseudo={pseudo_val:.6f}, diff={error_val:.6f}"
        )

    # Group errors by row
    error_rows = error_indices[:, 0].unique()
    print(
        f"\nErrors found in {len(error_rows)} rows out of {output.shape[0]} total rows"
    )

    # Show rows with most errors
    error_counts_per_row = torch.bincount(
        error_indices[:, 0], minlength=output.shape[0]
    )
    top_error_rows = torch.topk(error_counts_per_row, min(5, len(error_rows)))

    print(f"\nTop 5 rows with most errors:")
    for i, (row_idx, count) in enumerate(
        zip(top_error_rows.indices, top_error_rows.values)
    ):
        if count > 0:
            print(f"  Row {row_idx.item()}: {count.item()} errors")
            # Show first few error columns in this row
            row_errors = error_indices[error_indices[:, 0] == row_idx]
            print(f"    First 5 error columns: {row_errors[:5, 1].tolist()}")

    # Group errors by column
    error_cols = error_indices[:, 1].unique()
    error_counts_per_col = torch.bincount(
        error_indices[:, 1], minlength=output.shape[1]
    )
    top_error_cols = torch.topk(error_counts_per_col, min(5, len(error_cols)))

    print(f"\nTop 5 columns with most errors:")
    for i, (col_idx, count) in enumerate(
        zip(top_error_cols.indices, top_error_cols.values)
    ):
        if count > 0:
            print(f"  Column {col_idx.item()}: {count.item()} errors")

    # Detailed output of all problematic rows and columns
    print(f"\n=== DETAILED ERROR PATTERN ANALYSIS ===")

    # All rows with errors (sorted by error count)
    all_error_rows = []
    for row_idx in range(output.shape[0]):
        error_count = (error_indices[:, 0] == row_idx).sum().item()
        if error_count > 0:
            all_error_rows.append((row_idx, error_count))
    all_error_rows.sort(key=lambda x: x[1], reverse=True)

    print(f"\nAll rows with errors (sorted by error count):")
    for row_idx, count in all_error_rows:
        # Get all error columns in this row
        row_errors = error_indices[error_indices[:, 0] == row_idx]
        error_cols = sorted(row_errors[:, 1].tolist())
        print(f"  Row {row_idx:3d}: {count:3d} errors -> columns: {error_cols}")

    # All columns with errors (sorted by error count)
    all_error_cols = []
    for col_idx in range(output.shape[1]):
        error_count = (error_indices[:, 1] == col_idx).sum().item()
        if error_count > 0:
            all_error_cols.append((col_idx, error_count))
    all_error_cols.sort(key=lambda x: x[1], reverse=True)

    print(f"\nAll columns with errors (sorted by error count):")
    for col_idx, count in all_error_cols:
        # Get all error rows in this column
        col_errors = error_indices[error_indices[:, 1] == col_idx]
        error_rows = sorted(col_errors[:, 0].tolist())
        print(f"  Column {col_idx:3d}: {count:3d} errors -> rows: {error_rows}")

    # Check for patterns (e.g., every 16th row/column due to block size)
    print(f"\n=== PATTERN DETECTION ===")
    block_size = 16
    error_row_mods = {}
    error_col_mods = {}

    for row_idx, count in all_error_rows:
        mod = row_idx % block_size
        error_row_mods[mod] = error_row_mods.get(mod, 0) + count

    for col_idx, count in all_error_cols:
        mod = col_idx % block_size
        error_col_mods[mod] = error_col_mods.get(mod, 0) + count

    print(f"Error distribution by row modulo {block_size}:")
    for mod in sorted(error_row_mods.keys()):
        print(f"  Row % {block_size} = {mod:2d}: {error_row_mods[mod]} total errors")

    print(f"Error distribution by column modulo {block_size}:")
    for mod in sorted(error_col_mods.keys()):
        print(f"  Col % {block_size} = {mod:2d}: {error_col_mods[mod]} total errors")

    # Statistical analysis of errors
    error_values = diff[non_zero_mask]
    print(f"\nError statistics:")
    print(f"  Mean absolute error: {torch.abs(error_values).mean().item():.8f}")
    print(f"  Max absolute error: {torch.abs(error_values).max().item():.8f}")
    print(f"  Min absolute error: {torch.abs(error_values).min().item():.8f}")
    print(f"  Error std deviation: {error_values.std().item():.8f}")

    # Show specific problematic region (e.g., 5x5 region around the largest error)
    max_error_idx = torch.argmax(torch.abs(error_values))
    max_error_location = error_indices[max_error_idx]
    max_row, max_col = max_error_location[0].item(), max_error_location[1].item()

    print(
        f"\nLargest error at ({max_row}, {max_col}): {diff[max_row, max_col].item():.8f}"
    )
    print(f"5x5 region around largest error:")

    # Extract 5x5 region around the largest error (with boundary checking)
    row_start = max(0, max_row - 2)
    row_end = min(output.shape[0], max_row + 3)
    col_start = max(0, max_col - 2)
    col_end = min(output.shape[1], max_col + 3)

    print("Real quantization:")
    print(output[row_start:row_end, col_start:col_end])
    print("\nPseudo quantization:")
    print(pseudu_output[row_start:row_end, col_start:col_end])
    print("\nDifference:")
    print(diff[row_start:row_end, col_start:col_end])

    # Create visualization of error locations
    plt.figure(figsize=(20, 12))

    # Create error heatmap
    error_map = torch.zeros_like(diff)
    error_map[diff != 0] = 1

    # Plot 1: Error locations (binary) with highlighted rows/columns
    plt.subplot(2, 3, 1)
    plt.imshow(error_map.cpu().numpy(), cmap="Reds", aspect="auto")
    plt.title("Error Locations\n(Red = Error, White = No Error)", fontsize=12)
    plt.xlabel("Column Index", fontsize=10)
    plt.ylabel("Row Index", fontsize=10)
    plt.colorbar(label="Error Present")

    # Highlight rows with most errors
    for i, row_idx in enumerate(top_error_rows.indices[:5]):
        if top_error_rows.values[i] > 0:
            plt.axhline(
                y=row_idx.item(), color="blue", linestyle="--", alpha=0.7, linewidth=1
            )
            plt.text(
                -5,
                row_idx.item(),
                f"R{row_idx.item()}",
                color="blue",
                fontsize=8,
                va="center",
            )

    # Highlight columns with most errors
    for i, col_idx in enumerate(top_error_cols.indices[:5]):
        if top_error_cols.values[i] > 0:
            plt.axvline(
                x=col_idx.item(), color="green", linestyle="--", alpha=0.7, linewidth=1
            )
            plt.text(
                col_idx.item(),
                -5,
                f"C{col_idx.item()}",
                color="green",
                fontsize=8,
                ha="center",
                rotation=90,
            )

    # Plot 2: Error magnitude
    plt.subplot(2, 3, 2)
    error_magnitude = torch.abs(diff)
    error_magnitude[error_magnitude == 0] = np.nan  # Don't show zeros
    im = plt.imshow(error_magnitude.cpu().numpy(), cmap="viridis", aspect="auto")
    plt.title("Error Magnitude\n(Log Scale)", fontsize=12)
    plt.xlabel("Column Index", fontsize=10)
    plt.ylabel("Row Index", fontsize=10)
    plt.colorbar(label="Absolute Error")

    # Plot 3: Real quantization output
    plt.subplot(2, 3, 3)
    plt.imshow(output.cpu().numpy(), cmap="coolwarm", aspect="auto")
    plt.title("Real Quantization Output", fontsize=12)
    plt.xlabel("Column Index", fontsize=10)
    plt.ylabel("Row Index", fontsize=10)
    plt.colorbar(label="Value")

    # Plot 4: Pseudo quantization output
    plt.subplot(2, 3, 4)
    plt.imshow(pseudu_output.cpu().numpy(), cmap="coolwarm", aspect="auto")
    plt.title("Pseudo Quantization Output", fontsize=12)
    plt.xlabel("Column Index", fontsize=10)
    plt.ylabel("Row Index", fontsize=10)
    plt.colorbar(label="Value")

    # Plot 5: Difference between outputs
    plt.subplot(2, 3, 5)
    diff_display = diff.cpu().numpy()
    plt.imshow(diff_display, cmap="RdBu_r", aspect="auto")
    plt.title("Difference (Real - Pseudo)", fontsize=12)
    plt.xlabel("Column Index", fontsize=10)
    plt.ylabel("Row Index", fontsize=10)
    plt.colorbar(label="Difference")

    # Plot 6: Zoomed region around largest error
    plt.subplot(2, 3, 6)
    zoom_size = 20
    row_start = max(0, max_row - zoom_size // 2)
    row_end = min(output.shape[0], max_row + zoom_size // 2)
    col_start = max(0, max_col - zoom_size // 2)
    col_end = min(output.shape[1], max_col + zoom_size // 2)

    zoom_diff = diff[row_start:row_end, col_start:col_end]
    plt.imshow(zoom_diff.cpu().numpy(), cmap="RdBu_r", aspect="auto")
    plt.title(
        f"Zoom Around Largest Error\nLocation: ({max_row}, {max_col})", fontsize=12
    )
    plt.xlabel("Column Index", fontsize=10)
    plt.ylabel("Row Index", fontsize=10)
    plt.colorbar(label="Difference")

    # Mark the largest error location
    zoom_max_row = max_row - row_start
    zoom_max_col = max_col - col_start
    plt.plot(zoom_max_col, zoom_max_row, "r*", markersize=15, label="Largest Error")
    plt.legend()

    plt.tight_layout()
    plt.savefig("quantization_error_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nVisualization saved to 'quantization_error_analysis.png'")

    # Save error locations for further analysis
    torch.save(
        {
            "error_indices": error_indices,
            "error_values": error_values,
            "real_output": output,
            "pseudo_output": pseudu_output,
            "difference": diff,
        },
        "error_analysis.pt",
    )
    print(f"Error analysis data saved to 'error_analysis.pt'")

    # ---------------------------------------------------
    # Step 1: Recover A from real C using pseudo B^T inverse
    # ---------------------------------------------------

    # Use high precision for inversion and recovery
    a_fp64 = a.to(torch.float64)
    b_fp64 = b.to(torch.float64)
    output_fp64 = output.to(torch.float64)  # real quantized C

    # Pseudo-quantized B (same as used in pseudu_output)
    B_pseudo = pseudo_quant.nvfp4_pseudo_quantize(b).to(
        torch.float64
    )  # shape: [128, 128]
    B_pseudo_T = B_pseudo.t()  # [128, 128]

    # Check condition number to assess invertibility
    cond_B = torch.linalg.cond(B_pseudo_T)
    print(f"\nCondition number of pseudo B^T: {cond_B:.2e}")
    if cond_B > 1e12:
        print("⚠️ Warning: pseudo B^T is ill-conditioned! Inversion may be unstable.")

    # Compute inverse of B_pseudo^T in float64
    try:
        inv_B_pseudo_T = torch.linalg.inv(B_pseudo_T)  # (B^T)^{-1}
    except torch._C._LinAlgError:
        print("Failed to invert pseudo B^T. Using pseudo-inverse instead.")
        inv_B_pseudo_T = torch.linalg.pinv(B_pseudo_T)

    # Recover A from real C: A_recovered = C_real @ (B_pseudo^T)^{-1}
    A_recovered = output_fp64 @ inv_B_pseudo_T  # [128, 128]

    # Pseudo-quantized A (ground truth for comparison)
    A_pseudo = pseudo_quant.nvfp4_pseudo_quantize(a).to(torch.float64)

    # Compute difference
    A_diff = A_recovered - A_pseudo  # [128, 128]

    # Analyze per-row error (L2 norm or max abs)
    row_errors_l2 = torch.norm(A_diff, dim=1)  # L2 norm per row
    row_errors_max = torch.max(torch.abs(A_diff), dim=1).values  # max abs per row

    # Find rows with large discrepancies
    top_k = 10
    top_l2_rows = torch.topk(row_errors_l2, top_k)
    top_max_rows = torch.topk(row_errors_max, top_k)

    print("\n=== A Recovery Error Analysis (vs Pseudo A) ===")
    print("Top 10 rows with largest L2 error:")
    for i in range(top_k):
        row_idx = top_l2_rows.indices[i].item()
        err = top_l2_rows.values[i].item()
        print(f"  Row {row_idx:3d}: L2 error = {err:.6f}")

    print("\nTop 10 rows with largest max-absolute error:")
    for i in range(top_k):
        row_idx = top_max_rows.indices[i].item()
        err = top_max_rows.values[i].item()
        print(f"  Row {row_idx:3d}: max |error| = {err:.6f}")

    N = 5
    group_size = 16
    assert a.shape[-1] % group_size == 0, "Last dim must be divisible by group_size"

    A_diff_abs = torch.abs(A_recovered - A_pseudo)  # [128, 128]
    # Reshape to [128, num_groups, group_size]
    num_groups = A_diff_abs.shape[-1] // group_size
    A_diff_abs_groups = A_diff_abs.view(A_diff_abs.shape[0], num_groups, group_size)
    group_max_error = A_diff_abs_groups.max(dim=-1).values  # [128, num_groups]

    row_group_max, argmax_group = group_max_error.max(dim=-1)  # [128]
    top_rows = torch.topk(row_group_max, N).indices  # top N rows by worst group

    def get_scale(tensor_value):
        FLOAT8_E4M3_EPS = torch.finfo(torch.float8_e4m3fn).tiny
        org_shape = tensor_value.shape
        group_size = 16
        if group_size > 0:
            assert org_shape[-1] % group_size == 0
            tensor_value = tensor_value.reshape(-1, group_size)

        max_val = tensor_value.abs().amax(dim=1, keepdim=True)
        scales = max_val / FLOAT4_E2M1_MAX
        # avoid divide a too small value
        global_scale = scales.max() / FLOAT8_E4M3_MAX
        scales = (
            (scales / global_scale)
            .clamp(min=FLOAT8_E4M3_EPS)
            .to(torch.float8_e4m3fn)
            .to(tensor_value.dtype)
        ) * global_scale
        return scales

    def value_div_scale(tensor_value, scales):
        org_shape = tensor_value.shape
        group_size = 16
        tensor_value = tensor_value.reshape(-1, group_size)
        return (
            (tensor_value / scales.double()).reshape(org_shape).to(tensor_value.dtype)
        )

    scale = get_scale(a)
    a_cpu = value_div_scale(a, scale).cpu().float()
    a_pseudo_cpu = (
        value_div_scale(pseudo_quant.nvfp4_pseudo_quantize(a), scale).cpu().float()
    )
    A_recovered_cpu = value_div_scale(A_recovered, scale).cpu().float()
    A_pseudo_cpu = value_div_scale(A_pseudo, scale).cpu().float()

    print(f"\n=== Top {N} Rows: Worst Group (size={group_size}) Analysis ===")
    for i in range(N):
        row = top_rows[i].item()
        group_idx = argmax_group[row].item()
        col_start = group_idx * group_size
        col_end = col_start + group_size

        orig_group = a_cpu[row, col_start:col_end]
        pseudo_group = A_pseudo_cpu[row, col_start:col_end]
        recovered_group = A_recovered_cpu[row, col_start:col_end]
        diff_group = recovered_group - pseudo_group

        print(
            f"\nRow {row:3d}, Group {group_idx:2d} (cols {col_start:3d}–{col_end-1:3d}):"
        )
        print(f"  orig     = {orig_group.tolist()}")
        print(f"  pseudo   = {pseudo_group.tolist()}")
        print(f"  recovered= {recovered_group.tolist()}")
        print(f"  diff     = {diff_group.tolist()}")
        print(f"  max|diff|= {diff_group.abs().max().item():.6f}")

    # Optional: visualize A recovery error
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(row_errors_l2.cpu().numpy())
    plt.title("L2 Error per Row in Recovered A")
    plt.xlabel("Row Index")
    plt.ylabel("L2 Norm of Error")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(row_errors_max.cpu().numpy())
    plt.title("Max Absolute Error per Row in Recovered A")
    plt.xlabel("Row Index")
    plt.ylabel("Max |Error|")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("A_recovery_error_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nA recovery error plot saved to 'A_recovery_error_analysis.png'")

else:
    print("No errors found! Real and pseudo quantization produce identical results.")
