# REPORT — Quaternion / Structured-Rotation Efficiency Experiments

Device: NVIDIA GB10 (DGX Spark), 130.6 GB unified, torch 2.12.0+cu130. Measured peak D2D
bandwidth **219.6 GB/s**, BF16 GEMM peak **90.5 TFLOPS**. All results in
`reports/EXPERIMENTS.md` and `results/`; per-phase reports in `reports/`.

This is a partial execution: the plan is a ~25–30 day program; this run covered the cheap,
decisive front (Phase 0, algebra, synthetic gate G2, pretrained weight diagnostic, rotation
quantization, and Track A screening). Items not run are listed at the end.

---

## Bottom line

| Objective | Answer | Confidence |
|---|---|---|
| **O1** quaternion **weights** | **No** for pretrained; **yes, iso-params** for a full quaternion transformer from scratch (E5b) | High (pretrained) / Medium (scratch) |
| **O2** quaternion **sequence mixer** | **Promising** — only family that learns non-abelian state tracking | Medium |
| **O3** **learned rotation** for low-bit | **No** — fixed Hadamard is the strong default | Medium |
| **O4** Qwen3.8-27B baseline | Not started | — |

**Quantum-related algebra:** no classical speedup from quantum structure (as the plan states).
The useful ingredient is the **non-commutative SU(2) rotation** of unit quaternions — and it helps
in exactly one place (O2), not in weight compression or quantization.

---

## O1 — Quaternion weights

### Pretrained (plan 1.2): negative
Qwen3-0.6B, WikiText-2 calibration, all 7 projection types × 5 depths.
- **A5 confirmed exactly**: projecting any pretrained weight onto the block-quaternion subspace
  retains **0.250** of its energy — pretrained weights are isotropic w.r.t. quaternion structure.
- Plain Frobenius quaternion projection is useless (err ≈ 0.86). Activation-aware (A.3) projection
  is competitive with activation-aware low-rank **only on output projections** `o_proj`/`down_proj`.
- Dense **INT4 (g128) wins at equal bytes in 6/7 types**.
- End-to-end (replace all layers of a type at equal bytes): quaternion collapses ppl (1e6–1e7);
  INT4 stays within +0.7–2.7 ppl of 23.8.
- **Gate G1 failed.** Pretrained weight-structure track closed.

### From scratch (plan 2.4 + Tay et al. 2019): positive at iso-params
Llama-style d512/6L, 30M tokens, 3 seeds, iso-parameter (E5, E5b):
- Quaternion FFN only: 4.519 ± 0.031 vs dense-equal-params 4.576 ± 0.011 (Δ −0.057, t≈1.95, borderline).
- **Full quaternion transformer (quaternion attention + FFN), hidden 7680 at iso-params:
  4.333 ± 0.015 vs dense 4.445 ± 0.036 → Δ = −0.112 nats, t≈6.5, on all 3 seeds.** Only +28%
  training wall-time despite 4× FFN FLOPs. This replicates Tay et al. 2019 (arXiv:1906.04393) and
  reverses the earlier FFN-only "marginal" verdict.
- At 75% parameter reduction the full quaternion model is 4.550 (only +0.105 vs the 4×-larger dense).
- Dense **INT4** still dominates quaternion at equal *bytes* for pretrained weight conversion, and
  low-rank (4.632) is worst.

**O1 verdict:** pretrained-weight conversion is closed; but a **from-scratch full quaternion
transformer is a genuine iso-parameter win**. Whether it survives at 1B tokens / 100M params, and
whether the 4× FLOPs cost is acceptable at large batch, is untested.

---

## O2 — Quaternion / non-abelian rotation in the sequence mixer

### Algebra (plan 1.1) — all prerequisites verified
16/16 float64 tests. Key: **A9 confirmed** — selective quaternion-rotation linear attention equals
ordinary scalar-decay linear attention on rotated k/q, so it trains with **standard chunked kernels**
(no custom kernel required to test quality). A6 partially refuted (both `D·W` and `W·D` preserve
quaternion structure; the spec's "W·D does not" is false).

### Synthetic state tracking (plan 2.1, **Gate G2**) — positive
Group prefix-product task, transition-RNN shell, train length ≤ 8, test to 256:
- **A5 (smallest non-solvable group): quaternion 9/10 seeds reach 100% at length 256.**
  Dense SO(8) 0/8 (best 0.09), abelian 0, Householder 0, DeltaProduct 1/8.
- S3: quat 2/2. S5: all fail (expected — no faithful 4-dim real rep).
- Honest framing: SO(8) *contains* a faithful A5 action, so this is an **inductive-bias/optimization**
  win for quaternion structure, not a strict expressivity theorem.
- **Gate G2 PASSED.**

### Track B at LM scale (plan 2.5) — inconclusive / no benefit
Exact parallel linear attention, iso-params ~19.27M, 20M tokens:
- scalar 4.5644/4.5841, **quat 4.5980/4.5299** (mean Δ −0.010, seed noise ~0.03–0.05 → **inconclusive**).
  Abelian complex rotation is *worse* than no rotation (4.638).
- MQAR: no advantage for quat (16-pair acc 0.097 vs 0.097 scalar).
- No state-size or inference advantage (same state; the running frame adds d_k numbers/head).

### Where non-commutativity matters (plan extension)
- Single-layer transition recurrence **does** reproduce the win: S3 1.000, A5 9/10 seeds (E8).
- Adding **distractor tokens** or depth makes every variant fail at this budget (E8) — the shell
  lacks a selectivity gate to map distractors to the identity transition.
- The **linear-attention (A.4) shell itself does not state-track** even with quaternion rotations
  (E7 `track_mixer.py`, all variants ≈ chance): its state is a sum of rank-1 writes, so a
  data-dependent query cannot hold the constant readout that group tracking needs. The transition
  recurrence is the correct object.

**O2 verdict (final):** the non-abelian quaternion transition is a real but **narrow** inductive
bias for single-layer recurrent **state tracking** (A5: 9/10 vs 0/8 dense SO(8), 1/8 DeltaProduct,
0/7 Householder). It does **not** transfer to plain LM or recall, is not realized by the A.4
linear-attention shell as implemented, and gives no inference/state-size benefit. Not a general LM
architecture win.


---

## O3 — Learned structured rotation for quantization

Qwen3-0.6B, W4A4 (per-token absmax A4, g128 RTN W4), WikiText-2 ppl:
- No rotation: 146350 → random orthogonal: 81.4 → **fixed Hadamard: 63.0**. Rotations are essential
  and Hadamard beats a random rotation.
- **Quaternion-butterfly (QB), initialized exactly at Hadamard: 68.3** — no improvement.
- Learned dense (layerwise-MSE proxy): 1158 (overfits).

**O3 verdict:** no evidence that a structured/learned quaternion-butterfly rotation beats a fixed
Hadamard. The plan's soft gate fires: deprioritize for Qwen3.8 unless G5 fires. Caveat: the
quantizer is crude (no GPTQ/clipping/folding) and the learned baseline uses a proxy objective.

---

## O4 — Qwen3.8-27B

Not started. Downloads are permitted; this is the largest remaining item (baseline latency/quality
table for BF16/FP8/NVFP4 ± MTP, byte-budget profile, 27B projection diagnostic).

---

## Claims audit (plan Section 3)

| ID | Claim | Status |
|---|---|---|
| A1 | Spark ~273 GB/s, ~30 TFLOPS | **partly refuted**: 220 GB/s measured; 30 TFLOPS safe |
| A2 | quat layer = 4× fewer params | confirmed |
| A3 | activation-aware closed form | confirmed |
| A4 | INT4 dense beats quat at iso-bytes | **confirmed** (pretrained) |
| A5 | pretrained weights far from quat subspace | **confirmed, exactly 0.250** |
| A6 | `H⊗I₄` preserves; `W·D` does not | **partly refuted** (both preserve) |
| A9 | quat rotated LA ≡ scalar-decay LA on rotated k/q | **confirmed** |
| A10 | unit quats cover A5 word problem | **confirmed** (9/10); S5 not representable |
| A11 | diagonal can't state-track; Householder/delta partly can | confirmed (Householder failed here; DeltaProduct 1/8) |
| A13/A14/A15 | 27B NVFP4 / rotation headroom / structured scaling | untested / A15 consistent with low-rank being worst |

## Recommended next experiments (in priority order)
1. **Track B at LM scale** — does the O2 state-tracking advantage improve language modeling or MQAR
   at equal state size? This is the only place a real quaternion win remains.
2. **A proper from-scratch Track A run** (1B tokens, LR sweep, 100M confirmation) to test the weak
   O1 positive.
3. **O4** Qwen3.8-27B baseline if a deployment-latency question dominates.
4. Quantization: only revisit if a real W4A4 gap is found on 27B (G5), then compare QB vs SpinQuant
   with a proper end-to-end objective.
