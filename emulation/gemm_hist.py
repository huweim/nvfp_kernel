"""Opt-in GEMM-shape histogram: ETON_GEMM_HIST=<path prefix> counts every emulated NVFP4 GEMM call by (M, N, K) and
writes <prefix>.<pid>.json (every 2000 calls and at exit). Used to estimate the end-to-end time of other GEMM backends
(e.g. MMA-Sim) from a measured run: sum over calls of t_backend(M, N, K). No effect when the variable is unset."""
import atexit
import json
import os
from collections import Counter

_PREFIX = os.environ.get("ETON_GEMM_HIST")
_HIST = Counter()
_N = [0]


def _dump():
    if _PREFIX and _HIST:
        path = f"{_PREFIX}.{os.getpid()}.json"
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"calls": sum(_HIST.values()),
                       "hist": [{"M": m, "N": n, "K": k, "count": c} for (m, n, k), c in sorted(_HIST.items())]}, f)
        os.replace(tmp, path)


if _PREFIX:
    atexit.register(_dump)


def record(m, n, k):
    if _PREFIX:
        _HIST[(int(m), int(n), int(k))] += 1
        _N[0] += 1
        if _N[0] % 2000 == 0:
            _dump()
