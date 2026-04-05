#!/usr/bin/env bash
set -euo pipefail

source /home/wmhu/anaconda3/etc/profile.d/conda.sh
conda activate llm_inference

# Common config
M=4096
N=4096
K=4096
ITERS=15
WARMUP=5
SEED=12345
COMMON_ARGS="--m ${M} --n ${N} --k ${K} --iters ${ITERS} --warmup ${WARMUP} --skip-correctness-check --profile-stages --profile-iters 2 --profile-warmup 1 --seed ${SEED}"

echo "[1/5] baseline"
python test_speed.py ${COMMON_ARGS} --impl baseline

echo "[2/5] optimized"
python test_speed.py ${COMMON_ARGS} --impl optimized

echo "[3/5] stage4 triton only"
python test_speed.py ${COMMON_ARGS} --impl triton --triton-disable-stage3

echo "[4/5] stage3+4 triton (non-fused)"
python test_speed.py ${COMMON_ARGS} --impl triton

echo "[5/5] stage3+4 triton + fusion"
python test_speed.py ${COMMON_ARGS} --impl triton --triton-fuse-stage34

echo "Done."
