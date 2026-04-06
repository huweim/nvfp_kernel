# 4096 配置下 Emulation 五路对比报告

## 1. 测试目标
对比以下 5 条路径在 `M=N=K=4096` 下的整体时延和 Stage Breakdown：

1. `baseline`
2. `optimized`
3. `Stage4 Triton only`（`impl=triton --triton-disable-stage3`）
4. `Stage3+4 Triton`（非融合）
5. `Stage3+4 Triton + Fusion`

---

## 2. 复现脚本
已保存脚本：

- [`benchmark_4096_variants.sh`](/home/wmhu/emulation_workspace/nvfp_kernel/benchmark_4096_variants.sh)

脚本内容等价于：

```bash
source /home/wmhu/anaconda3/etc/profile.d/conda.sh
conda activate llm_inference

python test_speed.py --m 4096 --n 4096 --k 4096 --iters 15 --warmup 5 --impl baseline --skip-correctness-check --profile-stages --profile-iters 2 --profile-warmup 1 --seed 12345
python test_speed.py --m 4096 --n 4096 --k 4096 --iters 15 --warmup 5 --impl optimized --skip-correctness-check --profile-stages --profile-iters 2 --profile-warmup 1 --seed 12345
python test_speed.py --m 4096 --n 4096 --k 4096 --iters 15 --warmup 5 --impl triton --triton-disable-stage3 --skip-correctness-check --profile-stages --profile-iters 2 --profile-warmup 1 --seed 12345
python test_speed.py --m 4096 --n 4096 --k 4096 --iters 15 --warmup 5 --impl triton --skip-correctness-check --profile-stages --profile-iters 2 --profile-warmup 1 --seed 12345
python test_speed.py --m 4096 --n 4096 --k 4096 --iters 15 --warmup 5 --impl triton --triton-fuse-stage34 --skip-correctness-check --profile-stages --profile-iters 2 --profile-warmup 1 --seed 12345
```

---

## 3. 测试配置
- GPU: NVIDIA A100-PCIE-40GB
- Env: `llm_inference` (`torch 2.8.0+cu128`, `triton 3.4.0`)
- 固定参数：`W3=34`, `W4=28`, `seed=12345`, `m_chunk_size=128`
- 计时参数：`warmup=5`, `iters=15`

---

## 4. 总体结果（Overall Latency）

| Variant | Avg Latency (ms) | Speedup vs Baseline |
|---|---:|---:|
| baseline | 1239.809 | 1.000x |
| optimized | 1035.323 | 1.198x |
| Stage4 Triton only | 790.354 | 1.569x |
| Stage3+4 Triton | 370.932 | 3.342x |
| Stage3+4 Triton + Fusion | 350.125 | 3.541x |

结论（4096）：
- 相比 baseline，当前最佳 `Stage3+4 Triton + Fusion` 达到 **3.54x**。
- 在 Stage4 已 Triton 化前提下，Stage3 Triton 化把总时延从 `790.354 -> 370.932 ms`，约 **2.13x**。
- Stage3+4 融合在非融合 Triton 基础上再提升约 **1.06x**（`370.932 -> 350.125 ms`）。

---

## 5. Stage Breakdown（ms）

| Variant | Stage1 | Stage2 | Stage3 | Stage4 |
|---|---:|---:|---:|---:|
| baseline | 16.030 | 403.756 | 612.524 | 461.842 |
| optimized | 16.029 | 88.312 | 644.334 | 472.098 |
| Stage4 Triton only | 16.064 | 88.993 | 647.780 | 43.845 |
| Stage3+4 Triton | 16.433 | 88.577 | 228.828 | 43.876 |
| Stage3+4 Triton + Fusion | 16.422 | 88.550 | 249.573 | 0.000 |

说明：
- 融合路径下 `Stage4` 被并入融合核，所以该列为 `0.000`，对应耗时计入 `Stage3`。
- “Stage3 Triton 化收益”应在 `Stage4 Triton only` 与 `Stage3+4 Triton` 之间比较：
  - Stage3: `647.780 -> 228.828 ms`，约 **2.83x**。

---

## 6. 一致性确认（bit-identical）
在 `4096` 下做了显式检查（`iters=1, warmup=1`）：

```bash
python test_speed.py --m 4096 --n 4096 --k 4096 --iters 1 --warmup 1 --impl triton --seed 12345
python test_speed.py --m 4096 --n 4096 --k 4096 --iters 1 --warmup 1 --impl triton --triton-fuse-stage34 --seed 12345
```

两条路径均输出：
- `Correctness check passed: triton output is bit-identical to baseline.`

---

## 7. 参数可调性说明
- `W_stage3/W_stage4` 仍可调（函数参数保持不变）。
- rounding 仍可配置；当不是 `RZ` 时，Triton 加速路径会自动回退到参考实现以保持语义正确。

