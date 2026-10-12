# nvfp_kernel — NVFP4 GEMM: real (CUTLASS sm_120a) and bit-exact emulation (ETON)

- **Real path.** `nvfp.ops.cutlass_scaled_fp4_mm`: the CUTLASS NVFP4 GEMM and quantization kernels, extracted from
  [vLLM](https://github.com/vllm-project/vllm). Builds only for sm_120a (RTX 5090 / RTX PRO 6000).
- **Emulation path.** `emulation/`: ETON, which reproduces the real GEMM bit for bit on any CUDA GPU with Triton
  (H100, A100, …). It takes the same arguments and operand layouts (packed e2m1, 128x4-swizzled ue4m3 scales, fp32
  alpha).

## Build (into an existing venv with torch 2.8; see the top-level README)
```bash
NVFP_BUILD_PROFILE=real           uv pip install --no-build-isolation -e .   # RTX 5090: CUTLASS kernels + emulation
NVFP_BUILD_PROFILE=emulation_only uv pip install --no-build-isolation -e .   # H100 / A100: quantization helpers + emulation
```
Editable: `nvfp` / `emulation` are imported from this directory and `scaled_fp4_ops*.so` is built here (git-ignored).

## Implementations (`emulation/registry.py`, the single table)
| name | what | use |
|---|---|---|
| `beam_eton_fused` | ETON-Fused: one Triton kernel (int8 tensor-core group sums, exact int32-limb accumulation) | default |
| `beam_fusednode` | same numerics, staged kernels | test oracle for ETON-Fused |
| `beam_base` | ETON-Base: unfused PyTorch W3/W4 pipeline (Tab. IV "B.") | latency baseline; not exact on block-range / cancellation inputs |
| `mmasim_v2` | MMA-Sim v2 reference model | comparison only; needs `pip install mmasim` |

```python
from emulation import EmulationKernel
out = EmulationKernel()(a_fp4, b_fp4, scale_a, scale_b, alpha, torch.float16)   # == ops.cutlass_scaled_fp4_mm
```
`accum_bits` (default 36, the probed sm_120a value) sets the accumulator width for what-if studies.

## Runtime switches
- `NVFP_GEMM_BACKEND`: `auto` | `real` | `emulation`. Selects what `nvfp.ops.cutlass_scaled_fp4_mm` runs. `auto` uses
  the real kernel when it was built.
- `NVFP_EMULATION_IMPL`: a registry name, used by the `emulation` backend.
- `NVFP_EMULATION_W3` / `NVFP_EMULATION_W4`: accumulator bits; they must be equal.
- `ETON_GEMM_HIST=<prefix>`: dump an (M, N, K) call histogram per process (used for the MMA-Sim E2E estimate).
- `ETON_FUSED_AUTOTUNE=1`: autotune ETON-Fused instead of using the fixed per-M configs.

## Test and profile
```bash
python -m pytest tests/test_impls.py -q      # every impl vs real (vs beam_fusednode where there is no real kernel)
python gemm_profile.py --m 512 --n 2048 --k-values 512,1024,2048,4096,8192 --iters 100 --warmup 20
python gemm_profile.py ... --m-chunk-size 512   # reproduces the paper's Tab. IV ETON-Base column
```

- **MXFP4.** `kernel_mxfp4/` holds the CUTLASS MXFP4 op; `emulation/mxfp4.py` holds ETON-MXFP4.
- **Older implementations.** `beam_234fusion*`, `beam_naive_triton` and the parameter search are at git tag
  `hpca2027-revision-1012`.
