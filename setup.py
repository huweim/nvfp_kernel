import glob
import os

from setuptools import find_packages, setup

current_dir = os.path.dirname(os.path.abspath(__file__))
build_profile = os.getenv("NVFP_BUILD_PROFILE", "real").strip().lower()

if build_profile == "emulation":
    build_profile = "emulation_only"

if build_profile not in {"emulation_only", "real"}:
    raise ValueError(
        f"Invalid NVFP_BUILD_PROFILE={build_profile!r}; expected 'emulation_only' or 'real'."
    )

ext_modules = []
cmdclass = {}

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    include_dirs = [os.path.join(current_dir, "kernel")]
    define_macros = []
    extra_compile_args = {}

    if build_profile == "real":
        sources = [
            os.path.join(current_dir, "binding.cpp"),
            *sorted(glob.glob(os.path.join(current_dir, "kernel", "*.cu"))),
        ]
        include_dirs.extend(
            [
                os.path.join(current_dir, "third_party", "cutlass", "include"),
                os.path.join(
                    current_dir, "third_party", "cutlass", "tools", "util", "include"
                ),
            ]
        )
        define_macros.append(("NVFP_WITH_REAL_GEMM", "1"))
        extra_compile_args = {
            "cxx": ["-O3", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0"],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                "-gencode=arch=compute_120a,code=sm_120a",
            ],
        }
    else:
        sources = [
            os.path.join(current_dir, "binding.cpp"),
            os.path.join(current_dir, "kernel", "reciprocal_kernels.cu"),
        ]

    ext_modules = [
        CUDAExtension(
            name="scaled_fp4_ops",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    cmdclass["build_ext"] = BuildExtension
except ImportError:
    pass

setup(
    name="nvfp",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[
        "torch>=2.8.0",
    ],
    python_requires=">=3.8",
)
