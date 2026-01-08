# nvfp_kernel
It's the nvfp kernel extracted from vLLM ([GitHub repository](https://github.com/vllm-project/vllm)).

## Setup
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python setup.py install
```

## Correctness Test
```bash
python example.py
```
There's still a small gap between the pseudo quantization and the real quantization.

## Speed Test
```bash
python speed_test.py
```
Here's the speed test result on RTX 5090. The settings can be seen in `speed_test.py`.

| Size (K=N) | BF16 (ms) | FP4 (ms) | Speedup (FP4/BF16) | BF16 TFLOPS | FP4 TFLOPS |
|------------|-----------|----------|---------------------|--------------|------------|
| 64         | 0.013     | 0.040    | 0.34                | 2.52         | 2.52       |
| 128        | 0.007     | 0.040    | 0.16                | 20.41        | 20.41      |
| 256        | 0.011     | 0.039    | 0.29                | 47.70        | 47.70      |
| 512        | 0.031     | 0.040    | 0.77                | 69.77        | 69.77      |
| 1024       | 0.059     | 0.046    | 1.29                | 144.48       | 144.48     |
| 2048       | 0.215     | 0.078    | 2.77                | 159.68       | 159.68     |
| 4096       | 0.773     | 0.164    | 4.71                | 177.83       | 177.83     |
| 8192       | 2.440     | 0.562    | 4.34                | 225.27       | 225.27     |

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SubSir/nvfp_kernel)
