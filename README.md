# Quaternion / Structured-Rotation Efficiency Experiments

Research code asking one question: **does quaternion algebra buy anything real for LLMs?**
The program tests quaternion-structured **weights** (O1), non-abelian quaternion **rotations in the
sequence mixer** (O2), and learned structured **rotations for low-bit quantization** (O3), plus a
final Qwen3.8-27B latency baseline (O4). Full spec: [`plan.md`](plan.md); chronological decisions
and deviations: [`LOG.md`](LOG.md); results: [`reports/`](reports/) and [`results/`](results/).

## Bottom line

| Objective | Answer | Confidence |
|---|---|---|
| **O1** quaternion weights | Pretrained: **no**. From-scratch full quaternion transformer: **yes, iso-params** | High / Medium |
| **O2** quaternion sequence mixer | **Promising but narrow** — only family that learns non-abelian state tracking; no LM/MQAR benefit | Medium |
| **O3** learned rotation for low-bit | **No** — fixed Hadamard is the strong default | Medium |
| **O4** Qwen3.8-27B baseline | Not started | — |

See [`reports/REPORT.md`](reports/REPORT.md) for the full write-up and claims audit.

## Requirements

- Python 3.12, PyTorch 2.12+ (CUDA build for GPU experiments; CPU works for the algebra tests).
- A CUDA GPU for training/eval. Reference device: NVIDIA GB10 (DGX Spark), 130.6 GB unified memory,
  torch 2.12.0+cu130.
- Datasets/models are pulled from Hugging Face (WikiText-2/103, Qwen3-0.6B) into `checkpoints/`.

## Layout

```
plan.md              execution spec and assumptions register
LOG.md               decisions, deviations, blockers
configs/             one config per experiment arm
src/quat/            quaternion ops (Hamilton product, L/R matrices, projections)
src/models/          small LMs and mixer/state-tracking variants
src/eval/            benchmarks, projection diagnostics, rotation quantization, MQAR
kernels/             Triton/CUDA decode kernels (empty so far)
tests/               float64 algebra unit tests
results/             jsonl/json per run
reports/             per-phase reports + consolidated REPORT.md
data/  checkpoints/  large artifacts (gitignored)
```

## Usage

Run the algebra unit tests (no GPU needed):

```bash
python3 tests/test_algebra.py
```

Phase 0 environment/bandwidth/GEMM calibration (writes `results/phase0.json`):

```bash
python3 src/eval/bench_phase0.py
```

Examples of the experiment entry points (each takes `--help`):

```bash
# O1: pretrained weight-projection diagnostic + end-to-end ppl
python3 src/eval/proj_diag.py --help
python3 src/eval/substitute_ppl.py --help

# O3: rotation for W4A4/W4A16 quantization
python3 src/eval/rotation_quant.py --help
python3 src/eval/learn_rotation.py --help

# O1/O2: from-scratch LM and mixer screening
python3 src/models/train_lm.py --help
python3 src/models/train_mixer.py --help
python3 src/models/state_tracking.py --help
python3 src/eval/mqar.py --help
```

Scripts insert `src/` onto `sys.path`, so no package install is required.

## Notes

- Comparisons are **iso-params** (training) or **iso-bytes** (inference), per `plan.md` Rule 3.
- Quaternion convention: last dim 4 = `(a, b, c, d)` for `a + bi + cj + dk`; products and
  cumulative rotations are done in FP32/FP64.
- `data/` and `checkpoints/` are gitignored; regenerate data with the tokenizer/data prep in the
  relevant scripts.
