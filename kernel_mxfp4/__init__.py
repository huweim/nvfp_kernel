"""JIT-built CUTLASS sm_120a MXFP4 GEMM (reference for ETON-MXFP4). Separate from scaled_fp4_ops (NVFP4)."""
import os

_mod = None


def load(verbose=False):
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load as _load
        here = os.path.dirname(os.path.abspath(__file__))
        cut = os.path.join(here, "..", "third_party", "cutlass")
        _mod = _load(name="mxfp4_ops_sm120", sources=[os.path.join(here, "mxfp4_scaled_mm_sm120.cu")],
                     extra_include_paths=[os.path.join(cut, "include"), os.path.join(cut, "tools", "util", "include")],
                     extra_cflags=["-O3", "-std=c++17"],
                     extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_120a,code=sm_120a",
                                        "--expt-relaxed-constexpr"],
                     verbose=verbose)
    return _mod


def cutlass_scaled_mxfp4_mm(a, b, sfa, sfb, alpha, out_dtype):
    import torch
    return load().cutlass_scaled_mxfp4_mm(a, b, sfa, sfb, alpha, out_dtype == torch.float16)
