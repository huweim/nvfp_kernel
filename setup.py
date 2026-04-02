import os

from setuptools import find_packages, setup

current_dir = os.path.dirname(os.path.abspath(__file__))

ext_modules = []
cmdclass = {}

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    ext_modules = [
        CUDAExtension(
            name="scaled_fp4_ops",
            sources=[
                os.path.join(current_dir, "binding.cpp"),
                os.path.join(current_dir, "kernel", "reciprocal_kernels.cu"),
            ],
            include_dirs=[os.path.join(current_dir, "kernel")],
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
