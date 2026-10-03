# Experiment Results Log

One file for all measured results. Each experiment has a stable ID, config, seed(s), and the
comparison baseline. Negative/inconclusive results are recorded plainly (plan Rule 1).

Legend: **CONFIRMED** / **REFUTED** / **INCONCLUSIVE** / **PENDING**.

Device: NVIDIA GB10 (DGX Spark), CC 12.1, 130.6 GB unified, torch 2.12.0+cu130.
Measured peak D2D bandwidth **219.6 GB/s**, BF16 GEMM peak **90.5 TFLOPS** (4096³).

---

## E0 — Phase 0 calibration (`results/phase0.json`)

| Metric | Assumed (A1) | Measured | Verdict |
|---|---|---|---|
| D2D copy BW | ~273 GB/s | 219.6 GB/s | A1 optimistic ~20% |
| GEMV achieved | — | 191.6 GB/s | memory-bound |
| BF16 GEMM 512³/1024³/2048³/4096³ | — | 39.3 / 64.9 / 88.5 / 90.5 TFLOPS | training budget 30 TFLOPS safe |
| Unified memory | 128 GB | 130.6 GB | confirmed |

---

## E1 — Quaternion algebra (plan 1.1) — `tests/test_algebra.py`

16/16 tests pass (float64). Full detail in `reports/algebra_checks.md`.

| Assumption | Statement | Verdict |
|---|---|---|
| A2 | block-quat layer = 4× fewer params, same structure | CONFIRMED |
| A3 | activation-aware closed form is optimal & efficient | CONFIRMED |
| A5 | isotropic weights retain ~¼ energy in quat subspace | CONFIRMED (26.5%) |
| A6 | `H⊗I₄` left/right preserve block-quat | CONFIRMED |
| A6 (2nd clause) | `W·D` does **not** preserve structure | **REFUTED** — both `D·W` and `W·D` preserve |
| A9 | quat-rotation LA == scalar-decay LA on rotated k/q | CONFIRMED (rel 5e-8) |

**Implication:** A9 is a green light — selective quaternion-rotation attention can be trained with
existing chunked linear-attention kernels (no custom kernel needed to test quality).

---

## E2 — Synthetic state tracking (plan 2.1, GATE G2) — assumption #1

Status: **DONE — G2 PASSES for quaternion (assumption #1 supported at recurrence level)**.
Script `src/models/state_tracking.py`; data `results/statetracking_*.jsonl`;
full write-up `reports/part2_synthetic.md`.

Shell: `h_t = A_t h_{t-1}`, `h_0` learned, MLP readout. Task = group prefix product.
Train length ≤ 8, evaluate at length up to 256 (**length generalization**). d_emb 128,
state_dim 8, 12k steps, Adam 3e-3. Success = last-position accuracy > 0.9 at L=256.

| Group | Variant | Transition family | Success seeds | Best L256 |
|---|---|---|---|---|
| A5 | scalar | scalar decay | 0/3 | 0.02 |
| A5 | diagonal | diagonal decay | 0/3 | 0.02 |
| A5 | abelian | complex rotation (commutative) | 0/3 | 0.04 |
| A5 | householder | rank-1 reflection (GDN-like) | 0/3 | 0.02 |
| A5 | deltaproduct | 2 reflections | 1/8 | 1.00 |
| A5 | **quat** | **unit-quaternion blocks (L(q))** | **9/10** | **1.000** |
| A5 | orth | dense SO(8) (upper bound) | 0/8 | 0.09 |
| S3 | **quat** | unit quaternion | **2/2** | **1.000** |
| S3 | orth / householder / abelian | — | 0 | ≤0.25 |
| S5 | quat / orth / deltaproduct | — | 0/2 each | 0.02 |

Key finding: the unit-quaternion transition is the **only** family that reliably learns and
length-generalizes on the non-abelian, non-solvable group A5. Dense SO(8) *contains* a faithful
A5 action but does not find it (optimization/inductive-bias effect, not expressivity); this must be
stated honestly. S5 is out of reach for left-only quaternion blocks (no faithful 4-dim real rep),
consistent with plan A10.

**G2: PASS** → quaternion arms get full priority in Track B.


---

## E3 — Weight-projection diagnostic (plan 1.2 / 5.3) — O1

Status: **DONE — negative for quaternion weights.** Qwen3-0.6B (28L, hidden 1024, GQA),
WikiText-2 train calibration (16k tokens), 5 depths × 7 projection types.
`src/eval/proj_diag.py`, `results/proj_diag_qwen06.jsonl`, `src/eval/substitute_ppl.py`.

Per-layer activation-weighted error `||(W-W꜀)X||/||WX||`:

| proj | Q_frob | **Q_act** | LR (act-aware) | PHM4 | INT4 g128 | G1 (Q_act) |
|---|---|---|---|---|---|---|
| q_proj | 0.82 | 0.248 | 0.171 | 0.78 | 0.099 | 0/5 |
| k_proj | 0.82 | 0.244 | 0.179 | 0.81 | 0.093 | 0/5 |
| v_proj | 0.89 | 0.423 | 0.376 | 0.90 | 0.153 | 2/5 |
| o_proj | 0.86 | 0.242 | 0.282 | 0.85 | 0.124 | **5/5** |
| gate_proj | 0.80 | 0.221 | 0.156 | 0.73 | 0.078 | 0/5 |
| up_proj | 0.87 | 0.368 | 0.279 | 0.87 | 0.129 | 0/5 |
| down_proj | 0.86 | 0.159 | 0.327 | 0.85 | 0.130 | **5/5** |

- **A5 CONFIRMED exactly:** retained Frobenius energy under block-quaternion projection = **0.250**
  for *every* layer → pretrained dense weights are isotropic w.r.t. quaternion structure.
- **A4 CONFIRMED:** dense INT4 (g128) has the lowest error at equal bytes in 6/7 projection types.
  Quaternion only beats activation-aware low-rank on the *output* projections (o_proj, down_proj).
- **PHM4 ≈ Q_frob** (both useless); activation awareness is what rescues quaternion.

End-to-end WikiText-2 ppl when **all** layers of one type are substituted (equal bytes):

| Method | o_proj ppl | down_proj ppl |
|---|---|---|
| original | 23.80 | 23.80 |
| **Q_act** | 2.7e7 (collapse) | 3.9e6 (collapse) |
| LR act-aware | 1053.5 | 372.4 |
| **INT4 g128** | **24.47** | **26.46** |

**Verdict O1:** quaternion-structured **weights are not competitive**. G1 fails (ppl criterion).
From-scratch Track A still runs (a model can adapt to structure), but no pretrained-weight path.


---

## E4 — Rotation for quantization (plan 1.3) — assumption #3 / O3

Status: **DONE — negative for learned/QB rotations.** Qwen3-0.6B, W4A4 runtime rotation
(`(xR)(WR)^T`), WikiText-2 ppl (40k tokens). `src/eval/rotation_quant.py`,
`src/eval/learn_rotation.py`, `results/rotation_quant.jsonl`.

Function preservation with orthogonal R verified (max|Δ| ≤ 8e-7). Quantizers: per-token absmax
A4, per-output-row g128 RTN W4 (crude; no GPTQ/clipping/folding).

| Method | R | ppl |
|---|---|---|
| fp32 reference | — | 23.05 |
| W4A16 (no rotation) | — | 32.46 |
| W4A4 (no rotation) | I | 146349.7 |
| W4A4 + random orthogonal | random | 81.4 |
| **W4A4 + fixed Hadamard** | H | **63.0** |
| W4A4 + learned dense (layerwise MSE) | Cayley | 1158.3 |
| W4A4 + quaternion-butterfly (QB) | QB (init = H) | 68.3 |

- **Premise confirmed:** rotations are *essential* for W4A4 (146k → 63) and Hadamard beats a random
  orthogonal rotation.
- **Assumption #3 not supported:** QB, which is *initialized exactly at Hadamard*, does not improve
  on it (68.3 vs 63.0) — optimization slightly increases its training objective (7.70 → 8.13).
  A naive layerwise-MSE learned dense rotation overfits calibration and is far worse (1158).
- Plan's soft gate fires: record negative; do not pursue QB on Qwen3.8 unless G5 also fires.
- **Caveat:** absolute W4A4 ppl is far from fp16 because this quantizer is crude (no GPTQ, no
  clipping, no folding). The learned-rotation comparison uses a layerwise MSE proxy rather than
  SpinQuant's end-to-end KD, so "learned rotations cannot help" is not established in general —
  only that a fixed Hadamard is a strong, hard-to-beat default *in this setup*.

**Verdict O3:** no evidence that quaternion-butterfly rotations beat fixed Hadamard.


---

## E5 — Track A screening (plan 2.4) — O1

Status: **DONE — small positive at iso-params, but 4× FFN FLOPs.** From-scratch Llama-style LM
(d=512, 6 layers, 8 heads, seq 512, WikiText-103 GPT-2 tokens, **30M tokens**, AdamW 1e-3,
3 seeds for the key arms). `src/models/train_lm.py`, `results/trackA.jsonl`.

| Arm | FFN | hidden | non-emb params | stored FFN bytes | val loss (mean ± std, n=3) |
|---|---|---|---|---|---|
| A0 | dense | 1408 | 19.27M | 1× | 4.404 (1 seed) |
| **A1** | **quat** | 1408 | **9.54M** | **¼** | **4.519 ± 0.031** |
| A2 | dense | 352 | **9.54M** | 1× | 4.576 ± 0.011 |
| A3 | lowrank | 1408 | 9.55M | 1× | 4.632 (1 seed) |
| A6 | quat | 5632 | 19.27M | 1× (=A0) | 4.364 (1 seed) |

- **A1 vs A2 (iso-params, paired seeds):** quat wins all 3 seeds; paired Δ = **−0.057 nats**
  (t ≈ 1.95). Borderline by Rule 4 (vs 2×σ); a small parameter-efficiency advantage, but A1 runs
  **4× the FFN FLOPs** of A2 because it materializes a 1408-wide matmul (A2 only 352-wide).
- **A3 low-rank is clearly worst** (−0.06 vs quat at iso-params).
- **A6 vs A0 (iso-params, wide quat vs dense):** wide quat is better (4.364 vs 4.404) at the same
  parameter count and stored bytes, but again 4× FFN FLOPs.
- **Inference read:** a quat model at *dense shapes* (A1) gets 4.519 at ¼ FFN bytes vs dense FP16
  A0's 4.404. Dense INT4 at A0 shapes also stores ~¼ bytes and E3 shows INT4 beats quaternion
  projections, so quaternion weights still show **no clear inference-bytes advantage**.

Caveats: only 30M tokens (plan wanted 1B), one model size, seed noise comparable to the effect.
Verdict O1 (from scratch): **marginal, compute-costly, not yet a win.**


---

## E7 — Track B mixer screening (plan 2.5) — O2 / assumption #1

Status: **DONE — synthetic win does not transfer to plain LM (inconclusive/no benefit).**
`src/models/train_mixer.py` (exact quadratic LA; verified against sequential recurrence to 1e-15),
`src/eval/mqar.py`, `results/trackB.jsonl`, `results/mqar.jsonl`,
`reports/part2_trackB.md`.

Iso-params (~19.27M non-emb), d512/6L/8H, GPT-2 WikiText-103, 20M tokens:

| Mixer | state transition | seed0 val | seed1 val | mean |
|---|---|---|---|---|
| **scalar** | S←exp(g)S + kv (abelian, no rot) | **4.5644** | 4.5841 | 4.5743 |
| **quat** | non-abelian unit-quat rotation + decay | 4.5980 | **4.5299** | 4.5640 |
| complex | abelian data-dep. rotation + decay | 4.6384 | — | — |
| attn | softmax attention | 4.6168 | — | — |

- Quat and scalar **swap order across seeds**; difference (−0.010 mean) is far below seed noise
  (~0.03–0.05). **Rule 4 → inconclusive; no demonstrated LM benefit** for non-abelian rotation.
- Abelian complex rotation is *worse* than no rotation (4.638 vs 4.564, seed0).
- MQAR (16/64 pairs): weak for all mixers, **no quat advantage** (16-pair acc 0.097 vs 0.097 scalar
  when the decay gate is initialized open; chance 0.008).
- Not run: GDN baseline (`fla` gate bug on this build), hybrids B5–B7, ≥1σ confirmation at larger
  scale, force_hidden FFN control.

**Verdict O2:** the quaternion rotation is a real inductive bias for **state-tracking** (E2, G2
pass) but, at this scale on natural-language LM and MQAR, it is **neutral-to-slightly-negative**.
No inference win (the state is the same size; the running frame adds Dk numbers/head at decode).


---

## Final answers (updated as experiments complete)

- **O1 (quaternion weights worth pursuing?):** **No for pretrained weights** (E3: exactly isotropic
  0.250 energy, INT4 dominates, end-to-end collapse). **Yes for a from-scratch full quaternion
  transformer** (E5b: quat attention+FFN at iso-params beats dense by 0.112 nats over 3 seeds,
  t≈6.5, at +28% training wall-time; E5 FFN-only was only borderline). No inference-*bytes* win
  over dense INT4 at equal bytes.
- **O2 (quaternion/non-abelian rotation in sequence mixer?):** **Synthetic yes, LM no.**
  E2: the only family that reliably solves A5 state tracking and length-generalizes (9/10 seeds,
  100% at L256, trained ≤8); dense SO(8)/abelian/Householder fail. But E7 Track B at LM scale:
  quat ≈ scalar within seed noise (no gain), complex rotation worse, MQAR no advantage, same
  state size. It is a state-tracking inductive bias, not a general LM win.
- **O3 (learned QB-rotation better than fixed Hadamard for low-bit?):** **No evidence.** E4:
  rotations are essential (146k→63) and Hadamard beats random, but QB initialized at Hadamard does
  not improve on it (68.3 vs 63.0); a naive learned dense rotation overfits (1158).
- **O4 (Qwen3.8-27B latency baseline + structural test):** **Not started.** Downloads permitted;
  pending.

### On "quantum-related algebra"
No classical speedup comes from quantum structure (plan Section 1 is explicit). The value found is
purely in the **non-commutative rotation / SU(2) structure** of unit quaternions (O2), not in
anything quantum-mechanical. There is no evidence for quantum speedups; there is evidence that
non-abelian transition structure helps state tracking.


## E6 — Qwen3.8-27B baseline + diagnostics (plan 5.0–5.3) — O4

Status: **PENDING** (downloads permitted; not started).

## E8 — Hard state tracking: distractors + depth (plan 2.1 extension) — O2

Status: **INCONCLUSIVE (task too hard at this budget).** `src/models/track_hard.py`.

Multi-layer transition recurrence with random distractor tokens interleaved; target is the running
product of group tokens only (distractors must be ignored).

| Group | Variant | layers | noise | L256 acc (seed0) |
|---|---|---|---|---|
| S3 | quat | 1 | 0 | **1.000** |
| S3 | quat | 1 | 4 | 0.168 |
| S3 | quat / scalar | 2 | 0 | 0.12–0.19 |
| S3 | scalar / quat | 2 | 4 | 0.15–0.20 |
| A5 | scalar/complex/quat | 2 | 4 | ≤0.03 |

- Single-layer, no distractors reproduces the E2 win (quat 1.000).
- Adding distractors or a second layer makes **every** variant fail at 4k–8k steps: no variant
  learns to map distractor tokens to the identity transition. This does not discriminate and is
  not evidence against quaternion (the task is under-trained / the shell lacks a selectivity gate).
- The A.4 linear-attention shell (E7 `track_mixer.py`) also fails all variants on A5 — its state is
  a sum of rank-1 writes, so a data-dependent query and a varying key cannot both hold a constant
  readout; the transition recurrence is the correct object for group tracking.

**Net O2 (final):** the non-abelian quaternion transition is a real, narrow inductive bias for
single-layer recurrent state tracking (A5: 9/10 seeds vs 0/8 dense SO(8), 1/8 DeltaProduct,
0/7 Householder). It does **not** transfer to plain LM/recall (E7) and is not realized by the
linear-attention shell as implemented. No inference/state-size win.

## E5b — Full quaternion transformer (Tay et al. 2019 replication) — O1

Status: **POSITIVE (iso-params), reopens the from-scratch weight question.** arXiv:1906.04393
(quaternion FFN + quaternion attention, component-wise). We had only tested quaternion FFN before;
adding **quaternion attention** (Q,K,V,O via `QuatLinear`, standard softmax + RoPE) changes the
conclusion. Same harness as E5, 30M tokens, 3 seeds, iso-params (manual hidden resize).

| Arm | attn | ffn | hidden | non-emb | val mean ± std (n=3) |
|---|---|---|---|---|---|
| A0 dense | dense | dense | 1408 | 19.27M | 4.445 ± 0.036 |
| **full quat** | **quat** | **quat** | **7680** | **19.27M** | **4.333 ± 0.015** |
| full quat (75% params) | quat | quat | 1408 | 4.82M | 4.550 (1 seed) |
| attn quat only | quat | dense | 1920 | 19.27M | 4.426 (1 seed) |
| ffn quat only | dense | quat | 5632 | 19.27M | 4.364 (1 seed) |

- **Paired Δ (full quat − dense) = −0.112 nats on all 3 seeds**, t≈6.5 → significant by Rule 4.
- Training wall time only **+28%** despite 4× FFN FLOPs (hidden 7680 vs 1408) — the small-model
  training is not FLOP-bound. At decode (batch 1, bandwidth-bound) the stored bytes/params are
  identical to dense, so the extra compute is largely hidden.
- At the paper's **75% parameter reduction** (non-emb 4.82M) quality is 4.550 — only +0.105 vs the
  4×-larger dense model, reproducing the paper's "comparable at 75% compression" claim.
- Adding quaternion attention on top of quaternion FFN improved seed0 from 4.364 → 4.321.

**Revised O1 (from scratch):** the *full* quaternion transformer is a **real iso-parameter win**
(the earlier "marginal" verdict came from testing only quaternion FFN). Caveats: single model
size/corpus, 4× FFN FLOPs, and the win is at iso-params not iso-FLOPs.
