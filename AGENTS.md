# nvfp_kernel — notes for agents

See `README.md` for build, implementations and switches. This file lists the rules to follow when changing code here.

## Layout
```
nvfp/ops.py            backend dispatch (NVFP_GEMM_BACKEND); _ops_real.py (CUTLASS) / _ops_emulation.py (registry)
nvfp/_ext.py           loads the compiled scaled_fp4_ops extension
nvfp/pseudo_quant.py   fake-quant used by QuantLinear(mode="pseudo")
emulation/registry.py  IMPLS, DEFAULT_IMPL, HW_ACCUM_BITS, run()  <- the only implementation table
emulation/kernel.py    EmulationKernel: input validation + registry.run
emulation/core.py      MMAEngine entry points (beam_base, fusednode, eton_fused) + HardwareCore W-bit adders
emulation/triton_eton_fused.py  ETON-Fused kernel, FusedTemplate (from_probe, NVFP4_SM120)
emulation/triton_fusednode.py   staged fused-node kernels (oracle)
emulation/mxfp4.py     ETON-MXFP4;  emulation/mmasim_backend.py  MMA-Sim v2 adapter;  gemm_hist.py  call histogram
gemm_profile.py        Tab. IV harness (build_inputs / build_method_callables / benchmark_callable, used by
                       ../../scripts/table4_*.py and gemm_latency_llm.py)
tests/test_impls.py    regression gate
```

## Rules
- **Interface.** `EmulationKernel(a, b, scale_a, scale_b, alpha, out_dtype)` must keep matching
  `ops.cutlass_scaled_fp4_mm` in shapes, dtypes, padding and swizzling. See `_validate_inputs`.
- **Adding or renaming an impl.** Change only `emulation/registry.py`. Both dispatch points (`EmulationKernel` and
  `_ops_emulation`) and the CLIs (`--emulation-impl` choices, `gemm_profile --methods`) read it.
- **Before committing a numerics change.** Run `python -m pytest tests/test_impls.py` on the 5090, which compares
  against the real kernel. On other GPUs the test compares against `beam_fusednode`.
- **Template changes.** The template comes from probes (`probe_triton/`). Never change it to make a case pass without
  a probe that measures the new behaviour.
- **Speed knobs.** `m_chunk_size` and the ETON-Fused tile configs must not change any result bit.
- **Removed code.** The old variants (`beam_234fusion*`, `beam_naive_triton`, `*_profile`, `config.py`, `search.py`)
  are at tag `hpca2027-revision-1012`.
