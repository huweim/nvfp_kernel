python test_speed.py --m 2048 --n 2048 --k 2048 --iters 15 --warmup 5 --impl triton    --skip-correctness-check --profile-stages --profile-iters 2 --profile-warmup 1 --seed 12345
python test_speed.py --m 2048 --n 2048 --k 2048 --iters 15 --warmup 5 --impl triton    --triton-fuse-stage34 --skip-correctness-check --profile-stages --profile-iters 2 --profile-warmup 1 --seed 12345
python test_speed.py --m 2048 --n 2048 --k 2048 --iters 15 --warmup 5 --impl optimized --skip-correctness-check --profile-stages --profile-iters 2 --profile-warmup 1 --seed 12345
python test_speed.py --m 2048 --n 2048 --k 2048 --iters 15 --warmup 5 --impl baseline  --skip-correctness-check --profile-stages --profile-iters 2 --profile-warmup 1 --seed 12345
