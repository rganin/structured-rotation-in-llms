# LOG — Quaternion/Structured-Rotation Experiments

Chronological decisions, deviations, blockers, actual vs estimated time.

## 2026-09-27 — Session 1

### Environment established (Phase 0 partial) — actual ~0.2 d
- Device: NVIDIA GB10 (DGX Spark), compute capability 12.1 (Blackwell), 130.6 GB unified memory,
  20 CPU cores, 2.0 TB free disk.
- `torch 2.12.0+cu130`, CUDA 13.0, aarch64. Internet reachable (HF + PyPI 200).

### Measured Phase 0 numbers (`results/phase0.json`)
| Metric | Assumed (A1) | Measured |
|---|---|---|
| D2D copy bandwidth | ~273 GB/s | **219.6 GB/s** |
| GEMV achieved | — | 191.6 GB/s (4096×4096) |
| BF16 GEMM 512³ | — | 39.3 TFLOPS |
| BF16 GEMM 1024³ | — | 64.9 TFLOPS |
| BF16 GEMM 2048³ | — | 88.5 TFLOPS |
| BF16 GEMM 4096³ | — | 90.5 TFLOPS |
| Unified memory | 128 GB | 130.6 GB |

**Decision (Rule 2/5):** A1 bandwidth is optimistic by ~20% (220 vs 273 GB/s). Timeline math
`hours = 6ND/(TFLOPS·3600·1e12)` should use ~30–50 TFLOPS for realistic *training* (BF16 GEMM
peaks at 90, but include optimizer/activation costs). 30 TFLOPS training assumption is safe, not
conservative. No timeline rescale needed yet.

### Task 1.1 — Quaternion algebra unit tests — actual ~0.15 d
`src/quat/ops.py`, `tests/test_algebra.py` (16/16 pass, float64). Findings on the
**[DERIVED]** assumptions:

- **A2 CONFIRMED.** Block-quaternion layer = 4× fewer params; naive build + dense matmul correct.
- **A3 CONFIRMED.** Activation-aware closed form recovers an exact quaternion W from noiseless
  data (rel err 3.6e-9) and beats plain Frobenius projection on the activation-weighted objective.
- **A5 CONFIRMED.** Isotropic Gaussian weights retain 0.256 ≈ ¼ of energy under block-quaternion
  projection.
- **A9 CONFIRMED.** The [DERIVED] equivalence *selective quaternion rotation linear attention ==
  ordinary scalar-decay causal linear attention on rotated k/q* (`k̃_s = P̄_s⊗k_s`, `q̃_t = P̄_t⊗q_t`)
  holds numerically for random unit rotations and decay, rel err 5e-8 (fp64 roundoff). This is the
  enabler for training with existing chunked kernels.
- **A6 PARTIALLY REFUTED.** Both `(H⊗I₄)W`, `W(H⊗I₄)`, **and `D·W` AND `W·D`** remain
  block-quaternion when D is block-diagonal of left-multiplication unit-quaternion blocks. The spec
  claims `W·D` does *not*; that is **false**: `L(q_ij)L(p_j) = L(q_ij⊗p_j)` is still a left-mult
  matrix. The block-quaternion matrices form a subalgebra closed under multiplication, so any
  quaternion-block-diagonal factor on either side preserves structure. **Correct the spec text in
  Section 1.1 / A6.** (Non-commutativity does not break closure — only the *value* changes.)

### Bugs caught that would have corrupted later phases
- `Lmat` was initially the transpose of the true left-multiplication matrix (would silently give
  `L(q⊗p)` instead of `L(p⊗q)`).
- `conj` on a flattened quaternion vector flips the real part of every block after the first; always
  apply `conj` in `(..., 4)` layout.
- Cumulative-rotation scan must be initialized to the identity quaternion `(1,0,0,0)`, not `ones`
  (`1+i+j+k`, norm 2), or the frame is scaled and `q⁻¹ = q̄` fails.

## 2026-09-28 — Session 2

### Task 2.0/2.1 — implementation + synthetic state tracking (GATE G2) — actual ~0.5 d
- Implemented `src/models/state_tracking.py` (group prefix products S3/A5/S5) with transition
  families scalar/diagonal/abelian/householder/deltaproduct/quat/orth/dense.
- Tuning notes: a 4-dim state with a linear head cannot linearly separate 60 A5 classes; widened to
  state_dim 8 + LayerNorm + MLP head. Training length matters: at train_len 32 no family fits A5;
  at train_len ≤ 8 the structured solution is found. This is itself a finding (curriculum).
- **G2 PASSES.** Quaternion transition 9/10 seeds solve A5 to length 256 (trained ≤8); dense SO(8)
  0/8; DeltaProduct 1/8; scalar/diagonal/abelian/householder 0. S3: quat 2/2. S5: all fail.
- Honest framing: SO(8) *could* represent A5, so this is an **inductive-bias/optimization** win for
  quaternion structure, not a strict expressivity win. Recorded in `reports/part2_synthetic.md`.
- Deviation from plan: this used a transition-RNN shell (`h_t=A_t h_{t-1}`) rather than the literal
  A.4 linear-attention shell. Justified because A9 (E1) proves the A.4 shell with quaternion
  rotation is equivalent to standard scalar-decay LA on rotated k/q; the transition family is the
  object of interest for A10/A11. The literal A.4 shell remains a Track B item.

### Next
- E3: weight-projection diagnostic (plan 1.2) on a small pretrained model — addresses O1.
- E4: quaternion-butterfly rotation quantization (plan 1.3) — assumption #3.

## 2026-09-28 — Session 3 (Part I weight diagnostic, plan 1.2) — actual ~0.4 d

- Downloaded Qwen/Qwen3-0.6B (28L, h1024, GQA, tied emb). Implemented
  `src/eval/proj_diag.py` (Q_frob, Q_act, LR, PHM4, INT4, Q+INT4) and `src/eval/substitute_ppl.py`.
- Bugs fixed: PHM-4 ALS block indexing/residual (now recovers exact quaternion to 7e-8);
  activation-aware Cholesky made robust; SVD-LLM low-rank corrected (Vh vs V).
- **Results:** retained quaternion energy = 0.250 exactly on every layer (A5 confirmed); INT4
  dominates at iso-bytes; Q_act competitive only on o_proj/down_proj at layer level, and collapses
  end-to-end (ppl 1e6–1e7) vs INT4 (+0.7–2.7 ppl). **G1 FAILED** on the ppl criterion.
- Report: `reports/part1.md`. Decision: close pretrained quaternion-weight track (skip 1.4).

### Next
- 1.3 rotation-for-quantization (assumption #3): Hadamard vs learned dense vs quaternion-butterfly
  at W4A16/A4 on a small model; function-preservation first.

## 2026-09-28 — Session 4 (rotation quantization, plan 1.3) — actual ~0.5 d
- Implemented `src/eval/rotation_quant.py` (runtime W4A4/W4A16 with orthogonal R; function
  preservation max|Δ|≤8e-7) and `src/eval/learn_rotation.py` (Cayley dense + quaternion-butterfly).
- Results (Qwen3-0.6B, WikiText-2 ppl): fp32 23.05; w4a16 32.46; w4a4 no-rot 146350;
  +random 81.4; +Hadamard 63.0; +learned dense 1158; **+QB 68.3**. QB starts *at* Hadamard and does
  not improve. Soft gate fires: negative for assumption #3.

## 2026-09-28 — Session 5 (Track A from scratch, plan 2.4) — actual ~0.5 d
- Implemented `src/models/train_lm.py` (dense/quat/lowrank/phm FFN), GPT-2 tokenized WikiText-103.
- 30M tokens, iso-params: A1 quat 4.519±0.031 vs A2 dense-equal-params 4.576±0.011 (3 seeds,
  paired Δ −0.057, t≈1.95) but A1 uses 4× FFN FLOPs; A3 lowrank 4.632; A6 wide-quat 4.364 vs
  A0 dense 4.404 at equal params (4× FLOPs). No inference-bytes win vs dense INT4.

## Consolidated
- `reports/REPORT.md` answers O1–O4. Net: quaternion weights no; quaternion sequence mixer
  promising (G2 passed); QB rotation no; 27B not started.

### Deviations / scope cuts (Rule 5)
- Synthetic gate used a transition-RNN shell, not the literal A.4 linear-attention shell (A9
  justifies equivalence; shell test deferred to Track B).
- Track A: 30M tokens instead of 1B; no LR sweep, no 100M confirmation (G3 not evaluated).
- Track B, Phase K, Part III not run.
- Data: WikiText-103 instead of FineWeb-Edu for the quick Track A screen.

## 2026-09-28 — Session 6 (Track B mixers, plan 2.5) — actual ~0.5 d
- Implemented `src/models/train_mixer.py` (exact parallel LA: scalar/complex/quat/fixed/attn) with
  param-matched FFN; verified vs sequential recurrence <=3.6e-15. `src/eval/mqar.py`.
- `fla` chunk_gla gate is ignored on this build (output invariant to g) -> implemented mixers myself;
  GDN baseline deferred.
- 20M tokens, iso-params: scalar 4.564/4.584, quat 4.598/4.530 (order flips; mean diff -0.01,
  seed noise ~0.04 -> inconclusive). complex 4.638 (worse than no rotation). MQAR no quat advantage.
- **O2 conclusion: synthetic state-tracking win does not transfer to plain LM/recall; no inference
  benefit.** New architecture = specialized state-tracking tool, not a general LM improvement.
- Report: `reports/part2_trackB.md`; REPORT.md O2 updated.

## 2026-09-28 — Session 7 (where non-commutativity matters) — actual ~0.3 d
- Added `src/models/track_mixer.py` (A.4 LA shell on group task) and `src/models/track_hard.py`
  (multi-layer + distractor transition recurrence).
- Findings: A.4 linear-attention shell cannot state-track groups (all mixers chance); single-layer
  transition recurrence reproduces the A5 quat win (9/10), but distractors/depth break all variants
  at this budget (no selectivity gate). O2 is a narrow state-tracking win, not a general LM win.

## 2026-09-28 — Session 8 (Tay et al. 2019 arXiv:1906.04393 mapping) — actual ~0.4 d
- Read the quaternion-NLP paper. Its quaternion FFN = our QuatLinear; its key untested idea is
  **quaternion attention** (Q,K,V,O via Hamilton-product linear maps) and full-vs-partial.
- Implemented `--attn_kind` in `src/models/train_lm.py` (quaternion attention projections).
- **Positive result:** full quaternion transformer at iso-params (attn+FFN quat, hidden 7680,
  19.27M non-emb) = 4.333±0.015 vs dense 4.445±0.036 over 3 seeds (paired Δ −0.112, t≈6.5).
  +28% training wall-time (small-model training not FLOP-bound). At 75% params (4.82M non-emb)
  = 4.550. This contradicts the earlier FFN-only "marginal" verdict and supports the paper.
- Fresh untested ideas flagged: Hamilton-product/component-softmax attention; quaternion
  embeddings at small scale (tied emb = 57% of our params); full-vs-partial sweep with seeds.
