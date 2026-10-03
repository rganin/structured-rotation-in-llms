"""Phase 0 environment/calibration benchmarks (plan.md Section 5).

Measures achievable HBM bandwidth (D2D copy), GEMV, BF16 GEMM TFLOPS,
and peak memory, on this device. Prints a JSON blob and writes results/phase0.json.
"""
import json
import time
import platform
import subprocess
from pathlib import Path

import torch

DEV = "cuda"
OUT = Path(__file__).resolve().parents[2] / "results"
OUT.mkdir(exist_ok=True)


def timeit(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bandwidth():
    nbytes = 512 * 1024 * 1024  # 512 MB per buffer
    n = nbytes // 4
    a = torch.empty(n, device=DEV, dtype=torch.float32)
    b = torch.empty_like(a)
    dt = timeit(lambda: b.copy_(a))
    # copy reads + writes
    gbps = 2 * nbytes / dt / 1e9
    return {"copy_GBps": gbps, "bytes_per_copy": 2 * nbytes, "seconds": dt}


def gemv():
    n, m = 4096, 4096
    w = torch.randn(n, m, device=DEV, dtype=torch.bfloat16)
    x = torch.randn(m, device=DEV, dtype=torch.bfloat16)
    dt = timeit(lambda: torch.mv(w, x))
    nbytes = w.numel() * 2 + x.numel() * 2 + n * 2
    return {"gemv_ms": dt * 1e3, "GBps": nbytes / dt / 1e9, "shape": [n, m]}


def gemm(shape):
    m, k, n = shape
    a = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(k, n, device=DEV, dtype=torch.bfloat16)
    dt = timeit(lambda: a @ b)
    flops = 2 * m * k * n
    return {"shape": shape, "TFLOPs": flops / dt / 1e12, "ms": dt * 1e3}


def main():
    assert torch.cuda.is_available(), "no CUDA device"
    props = torch.cuda.get_device_properties(0)
    res = {
        "device_name": torch.cuda.get_device_name(0),
        "cc": f"{props.major}.{props.minor}",
        "total_mem_GB": props.total_memory / 1e9,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "bandwidth": bandwidth(),
        "gemv": gemv(),
        "gemm": [gemm(s) for s in [(512, 512, 512), (1024, 1024, 1024),
                                   (2048, 2048, 2048), (4096, 4096, 4096)]],
    }
    print(json.dumps(res, indent=2))
    (OUT / "phase0.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
