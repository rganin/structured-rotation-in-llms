# Quaternion / Structured-Rotation Efficiency Experiments — Execution Spec

**Target machine:** single NVIDIA DGX Spark (aarch64, Blackwell-class GPU, 128 GB unified memory)
**Spec date:** 2026-09-27
**Audience:** an autonomous coding/experiment agent. Read Section 0 first.
**Primary goal:** reduce LLM **inference latency** (batch-1 decode first), then **memory footprint**.
**Secondary goal:** find out whether quaternion algebra gives any real advantage over standard structures, at the pretrained-model level, at the small from-scratch level, and on Qwen3.8-27B.

---

## 0. Rules for the executing agent

1. **Do not fabricate results.** Report negative and inconclusive results plainly. A negative result is a valid deliverable.
2. **Measure, don't trust.** Every number in this spec marked *(estimate)* or *(assumed)* must be replaced by a measurement in Phase 0. Every claim tagged **[DERIVED]** was worked out by the spec author and has not been checked; write a unit test for it before building on it.
3. **Iso-bytes and iso-params comparisons only.** Any "win" must be reported against a baseline with the same stored bytes (inference) or same parameter count (training). Also report FLOPs and measured throughput.
4. **Seed noise.** A difference smaller than 2× the seed standard deviation (measured in Phase 2.2) is *inconclusive*, not a win.
5. **Time boxing.** If a task exceeds 1.5× its estimate, log a short decision note (continue / cut scope / skip) in `LOG.md` and proceed. If a setup blocker exceeds 1 day, document it, take the fallback in Section 12, and move on.
6. **Gates.** Sections marked **GATE** define go/no-go criteria. Follow them; do not run gated-off experiments unless a spare-time note says so.
7. **Never modify original model files.** Work on copies; keep checkpoints under `checkpoints/`, results under `results/`.
8. **Log everything**: git hash, config, seed, tokens seen, wall time, tokens/s, peak memory, parameter counts (total and non-embedding), stored bytes at FP16 and INT4 equivalents.
9. **Single GPU.** GPU jobs run serially. Overlap CPU-side work (coding, data prep, analysis) with GPU jobs.

---

## 1. Objectives and non-goals

**Objectives**
- O1. Decide whether quaternion-structured **weights** are worth pursuing (pretrained diagnostics, then from-scratch).
- O2. Decide whether quaternion / non-abelian **rotations in the sequence mixer** (linear attention / recurrent state) improve quality per byte of state or per parameter.
- O3. Decide whether **learned structured rotations** (quaternion butterfly ∘ Hadamard) improve low-bit quantization over fixed Hadamard rotations.
- O4. Establish a fair **latency baseline for Qwen3.8-27B** on the Spark, and test whichever of O1–O3 survive.

**Non-goals**
- Training anything above ~150M parameters from scratch.
- Reproducing frontier-model quality.
- Quantum-computing speedups (superposition/entanglement give no classical speedup; only the linear-algebra structure is used).

---

## 2. Environment and global conventions

- Use NVIDIA-provided containers for PyTorch/CUDA on the Spark. Expect aarch64 wheel gaps (flash-attn, Triton, fla kernels). Budget time for this in Phase 0.
- Python project layout:
  ```
  repo/
    LOG.md                # decisions, deviations, blockers
    configs/              # one YAML per experiment arm
    src/quat/             # quaternion ops, layers, rotations, tests
    src/models/           # small LM + variants
    src/eval/             # ppl, lm-eval wrappers, synthetic tasks
    kernels/              # Triton/CUDA decode kernels
    results/              # jsonl per run + summary CSVs
    reports/              # plots, REPORT.md
  ```
- Precision: BF16 training with FP32 master weights; quaternion products, cumulative rotations, and normalizations in FP32.
- Seeds: default `{0,1,2}`; data order fixed per seed and **identical across arms** for the same seed.
- Reference codebases (verify current URLs and licenses before use): `karpathy/nanoGPT` or `KellerJordan/modded-nanogpt` (transformer baseline), `fla-org/flash-linear-attention` (GLA, Gated DeltaNet, kernels), `spcl/QuaRot`, `facebookresearch/SpinQuant` (rotation quantization), `EleutherAI/lm-evaluation-harness` (downstream evals).

---

## 3. Assumptions register

| ID | Assumption | Confidence / source | How it is tested | Phase |
|----|-----------|---------------------|------------------|-------|
| A1 | Spark has ~273 GB/s memory bandwidth and 128 GB unified memory; achieved BF16 training throughput ~30 TFLOPS | Author's recollection; **assumed** | Bandwidth copy test + GEMM benchmark + 5-min training benchmark | 0 |
| A2 | A quaternion linear layer = real linear layer whose 4×4 blocks are left-multiplication matrices of a quaternion → 4× fewer parameters, same FLOPs if naive (16 mult per block); the 8-mult variant does not speed up GPUs | High (standard math) | Unit tests, kernel benchmarks | 1.1, K |
| A3 | Batch-1 decode is memory-bandwidth-bound, so bytes read per token is the latency metric | High | Roofline check in Phase 0 and Phase 5 | 0, 5 |
| A4 | At iso-bytes, INT4/VQ dense weights beat quaternion-structured weights on quality | Medium; consistent with "Compute Better Spent" (Qiu et al. 2024) | Sections 1.2, 2.4 | 1, 2 |
| A5 | Pretrained dense weights are far from the block-quaternion subspace (isotropic weights would keep ~¼ of energy) | Medium | Projection diagnostic | 1.2 |
| A6 | **[DERIVED]** A real Hadamard/orthogonal matrix applied *across* blocks (H ⊗ I₄) commutes with block-quaternion structure; left-multiplying a block-quaternion matrix by unit-quaternion blocks preserves structure, right-multiplying does not | Derived; unchecked | Numerical unit test | 1.1 |
| A7 | A 4-channel quaternion rotation alone cannot spread outliers globally; only Kronecker/butterfly structure across blocks can | High | Outlier statistics after rotation | 1.3 |
| A8 | Unit quaternions give an orthogonal, associative, **non-commutative** transition (3 free params per 4-block) | High | Unit tests | 2.1 |
| A9 | **[DERIVED]** Unit-quaternion transition with scalar decay is equivalent to ordinary scalar-decay linear attention applied to q/k rotated by the cumulative quaternion product (q̃ₜ = P̄ₜ⊗qₜ, k̃ₛ = P̄ₛ⊗kₛ). This makes it trainable with existing chunked linear-attention kernels | Derived; unchecked | Unit test vs sequential recurrence (Appendix A.4) | 2.1 |
| A10 | Unit quaternions cover the binary icosahedral group, so an A5 word problem is exactly representable with 4-dim blocks; S5 may need extra structure (e.g. left+right multiplication / reflections) | Group theory, medium; unchecked | A5/S5 synthetic tasks | 2.1 |
| A11 | Diagonal transitions cannot do state tracking (they commute); Householder/delta-rule transitions (DeltaNet, DeltaProduct, RWKV-7) partially can | Literature, medium | Baselines in 2.1 | 2.1 |
| A12 | Qwen3.8-27B: dense ~27–28B params, 64 layers = 16 × [3 × (Gated DeltaNet → FFN) + 1 × (gated full attention → FFN)], 262K native context, Apache 2.0, MTP speculative decoding supported | Web sources, Aug–Sep 2026; **verify from the model config and model card** | Read `config.json`, inspect modules | 5.0 |
| A13 | NVFP4 W4A4 on all linear layers of Qwen3.8-27B, including Gated DeltaNet gate projections, matches BF16 within seed noise | One preprint (arXiv 2609.04098, "MINIMA"); **unreproduced** | Reproduce or reimplement | 5.1 |
| A14 | Learned rotations help low-bit quantization mainly when W4A4/W4A8 shows a quality gap; if NVFP4 already matches BF16, rotation headroom is small | Author's reasoning | Gate G5 | 5 |
| A15 | Structured weight layers (Monarch, TT, Kronecker) often scale worse than dense at equal compute | Literature (Qiu et al. 2024), medium | Track A arms | 2.4 |
| A16 | Qwen3 small dense models use QK-norm and tied embeddings at small sizes, which affects rotation folding | Recollection; **verify** | Inspect config/modeling code | 1.3 |

---

## 4. Master timeline

Time estimates assume one Spark, an experienced agent, and A1's 30 TFLOPS (rescale after Phase 0 with `hours = 6·N·D / (TFLOPS·3600·1e12)`, N = total params including the tied output matmul, D = tokens). "Cond." = conditional on a gate.

| # | Task | Est. time | Depends on | GPU? |
|---|------|-----------|-----------|------|
| 0 | Environment, benchmarks, calibration | 1 d | — | yes |
| **Part I — Pretrained-model experiments** | | | | |
| 1.1 | Quaternion math unit tests (A2, A6, A9 prerequisites) | 0.25 d | 0 | no |
| 1.2 | Weight-projection diagnostic (quaternion vs low-rank vs PHM vs INT4) | 1 d | 0, 1.1 | light |
| 1.3 | Rotation-for-quantization: Hadamard baseline vs learned quaternion-butterfly | 3 d | 0, 1.1 | yes |
| 1.4 | Layer-wise distillation of promising layers (**cond. G1**) | 2 d | 1.2 | yes |
| **Part II — Small models from scratch** | | | | |
| 2.0 | Implement layers/models + tests | 1.5 d | 1.1 | no |
| 2.1 | Synthetic state-tracking experiments (**GATE G2**) | 1.5 d | 2.0 | light |
| 2.2 | Data prep + baseline noise floor (3 seeds @ ~58M) | 1.5 d | 2.0 | yes |
| 2.3 | LR sweeps (per arm) | folded into 2.4/2.5 | 2.2 | yes |
| 2.4 | Track A screening (weight structure), 7 arms | 1.7 d | 2.2 | yes |
| 2.5 | Track B screening (sequence mixer), 8 arms | 2.4 d | 2.1, 2.2 | yes |
| 2.6 | 100M confirmation, top-3 + baseline × 2 seeds (**GATE G3**) | 3.7 d | 2.4, 2.5 | yes |
| 2.7 | Third seed if borderline (**cond.**) | 1.8 d | 2.6 | yes |
| 2.8 | Part II analysis and interim report | 0.5 d | 2.6 | no |
| **Phase K — Kernels** (**GATE G4**) | | | | |
| K.1 | Fused quaternion decode kernel vs INT4/FP16 at iso-bytes | 2 d | 2.6 | yes |
| **Part III — Qwen3.8-27B** | | | | |
| 5.0 | Model acquisition, environment, architecture verification | 0.5–1 d | 0 | yes |
| 5.1 | Latency/quality baseline: BF16, FP8, NVFP4, ± MTP | 1.5 d | 5.0 | yes |
| 5.2 | Byte-budget profile per layer type | 0.5 d | 5.0 | light |
| 5.3 | Weight-projection diagnostic on 27B, incl. GDN projections | 1 d | 5.0, 1.2 | yes |
| 5.4 | Learned-rotation quantization (**cond. G5**) | 3.5 d | 5.1, 1.3 | yes |
| 5.5 | Identity-initialized quaternion rotation in GDN, tiny-budget distillation (**cond. G6, stretch**) | 2–4 d | 2.6 | yes |
| 6 | Final report | 1 d | all | no |

**Totals (sequential, single GPU):** core path ≈ 25 days; with all conditional items ≈ 30–32 days. Coding time for 2.0 and K.1 can overlap GPU runs, so the calendar time can be shortened by ~3–4 days.

**All items sorted by estimated cost (cheapest first)**, for triage if time runs short: 1.1 (0.25 d) → 2.8 & 5.2 (0.5 d) → 5.0 (0.5–1 d) → 0 & 1.2 & 5.3 (1 d) → 2.0 & 2.1 & 2.2 & 5.1 (1.5 d) → 2.4 & 2.7 & K.1 (1.7–2 d) → 1.4 (2 d) → 2.5 (2.4 d) → 1.3 & 5.4 (3–3.5 d) → 2.6 (3.7 d). If cutting, cut in this order: 5.5, 2.7, 1.4, 5.4, Track A arms A5–A6, K.1.

---

## 5. Phase 0 — Environment, benchmarks, calibration (≈ 1 day)

Tasks
1. Build the container/environment; confirm PyTorch sees the GPU; confirm `torch.compile` works; try Triton, flash-attn, and fla kernels and note which work.
2. **Bandwidth test:** large device-to-device copy and a GEMV loop; record achieved GB/s (test A1, A3).
3. **GEMM benchmark:** BF16 matmul at sizes typical of the small models (512–2048 hidden); record achievable TFLOPS.
4. **Training throughput benchmark:** 5-minute run of the baseline transformer (Section 7.2), record tokens/s, MFU, peak memory.
5. Compute `TFLOPS_eff` and rewrite the "Est. time" column of the master timeline in `LOG.md`.

Deliverable: `reports/phase0.md` with measured bandwidth, TFLOPS, tokens/s, software support matrix, and the rescaled timeline.

---

## 6. Part I — Experiments on pretrained models

**Models.** Primary: a small standard GQA transformer such as **Qwen3-1.7B** (0.6B for fast iteration). Reference-implementation compatibility: a Llama-architecture model (e.g. SmolLM2-1.7B) because QuaRot/SpinQuant code targets Llama-style layouts. Verify availability and config on Hugging Face. Optionally add any small hybrid Qwen (3.5/3.6/3.8 family) model if one exists; check the Hugging Face org page.

**Calibration data:** 128–512 sequences × 2048 tokens from FineWeb-Edu (a shard not used later for validation) or WikiText-2 train. Evaluation: WikiText-2 test perplexity at seq 2048; lm-eval-harness 0-shot on `arc_easy`, `hellaswag`, `piqa`, `winogrande`.

### 1.1 Quaternion math unit tests (≈ 0.25 d)

Implement `src/quat/ops.py` with the Hamilton product, left/right multiplication matrices, conjugate, normalization, and block-quaternion linear layer (Appendix A). Tests (float64, tolerance 1e-10):
- `L(p)L(q) = L(p⊗q)`; `R(q)x = x⊗q`; `L(p)R(q) = R(q)L(p)`.
- Unit q: `L(q)` orthogonal; `L(q)⁻¹ = L(q̄)`.
- **[DERIVED, A6]** For real H (n/4 × n/4) and block-quaternion W: `(H ⊗ I₄) W` and `W (H ⊗ I₄)` remain block-quaternion. For unit-quaternion diagonal blocks D: `D W` remains block-quaternion; `W D` does **not** in general. Record the truth in `reports/algebra_checks.md` and correct any wrong statement in this spec.
- Optional facts (low priority, from memory, do not build on them without verification): the Hurwitz-quaternion / D4 / E8 lattice relations mentioned in Appendix C.

### 1.2 Weight-projection diagnostic (≈ 1 d)

For every linear layer type (attention q/k/v/o, FFN gate/up/down) and for depths {early, middle, late}:

Approximations at **equal stored bytes** (FP16 quaternion = ¼ bytes of dense FP16 = INT4 dense ignoring scales):
- **Q**: block-quaternion. Two variants: (a) plain Frobenius projection (Appendix A.2), (b) **activation-aware closed form** (Appendix A.3, **[DERIVED]**).
- **LR**: activation-aware low-rank (SVD of W·C^{1/2}, C = XXᵀ/T, SVD-LLM/ASVD style) with rank r = mn / (4(m+n)).
- **PHM-4**: parameterized hypercomplex multiplication, `W = Σᵢ Aᵢ ⊗ Sᵢ`, 4 learned 4×4 Aᵢ, fitted by alternating least squares or Adam on the activation-weighted objective.
- **INT4**: round-to-nearest, group size 128 (and GPTQ if time allows).
- **Q+INT4** (¹⁄₁₆ bytes) as an extra row.

Metric per layer: `err = ‖(W − Ŵ)X‖_F / ‖WX‖_F` on calibration activations X. Also report the plain Frobenius retained-energy fraction of the naive projection (compare against the isotropic expectation of ~25%, test A5).

End-to-end: substitute **all** layers of a single type at a time with each approximation (no fine-tuning) and report WikiText-2 perplexity. Report a table: rows = layer type × depth, columns = method.

**GATE G1:** Quaternion is "competitive" on a layer type if `err_Q(b) ≤ 1.1 × min(err_LR, err_PHM4)` and its perplexity delta is within 2× that of INT4 on that layer type. If it is competitive on **no** layer type, stop the pretrained weight-structure track (skip 1.4); Part II Track A still runs, because a from-scratch model can adapt to structure that a pretrained one cannot. If competitive on some, run 1.4 for those types only.

### 1.3 Rotation for quantization: Hadamard vs learned quaternion butterfly (≈ 3 d)

Goal: test O3 on a pretrained standard transformer at W4A4 / W4A8 (+ KV4/KV8 if supported by the reference code).

Prerequisites (A16): read the model code and note (a) RMSNorm scale folding (fold into following linears), (b) **tied embeddings** — untie by copying before rotating, (c) QK-norm and RoPE — do not rotate q/k heads except through the online Hadamard handling described in the QuaRot paper; verify numerically.

Steps
1. Verify **function preservation in FP32**: after inserting the residual-stream rotation R₁ (embedding/output projections get R₁, input projections Rᵀ₁, etc.), logits equal the original within 1e-4.
2. Baselines, all measured for perplexity + downstream + decode tokens/s where the kernels exist:
   - FP16 reference.
   - RTN W4A16 (g128), RTN W4A4 without rotation (expected poor).
   - QuaRot-style fixed Hadamard rotations, RTN and (if time) GPTQ weights.
   - SpinQuant-style learned dense R₁ (Cayley SGD) as the strong learned baseline.
3. **Candidate: quaternion-butterfly rotation ("QB-rotation")** for the residual stream, dimension d, n = d/4 quaternion blocks. Stages s = 1..S (S = log₂ n, padded if d/4 is not a power of two — use a Hadamard-friendly factorization or apply per power-of-two sub-block):
   `R = Π_s [ G_s(θ_s) · Diag(L(p_{s,1}), …, L(p_{s,n})) ] · H`
   where `p_{s,i}` are unit quaternions (4 normalized params each), `G_s(θ)` is a Givens rotation (angle θ per block-pair, pairing block i with i⊕2^{s-1}, applied identically to the 4 components), and H is the fixed Walsh–Hadamard (optional). Every factor is orthogonal, so R is orthogonal; cost is O(d log d).
   Optimize the parameters on calibration data with the same objective and budget as the SpinQuant baseline (loss = KD/CE after quantization, straight-through estimator).
4. Report perplexity and downstream at each bit-width, the online-rotation overhead in decode latency, and outlier statistics (max/median activation magnitude per channel group) before/after each rotation.

Hypothesis to test, not assume: QB-rotation matches or beats fixed Hadamard at W4A4 with less freedom than dense learned R. Risk: the family is too restrictive to beat plain Hadamard. Both outcomes are fine.

**GATE (soft):** if QB-rotation does not beat fixed-Hadamard by more than seed/optimization noise (repeat calibration with 3 different calibration subsets), record the negative result and do not pursue it on Qwen3.8 unless G5 also fires.

### 1.4 Layer-wise distillation of promising layers (cond. G1, ≈ 2 d)

For layer types that passed G1: initialize block-quaternion weights from the activation-aware solution (A.3), then optionally fine-tune with output-matching loss on calibration activations (per layer, then jointly for a few hundred steps). Compare perplexity/downstream against INT4 (RTN/GPTQ) at the same bytes. Report the full-model result with **only** the passing layer types replaced.

---

## 7. Part II — Small models from scratch

### 7.1 Dataset and evaluation (recommendation)

- **Training data: FineWeb-Edu, `sample-10BT` config** (Hugging Face `HuggingFaceFW/fineweb-edu`). It is quality-filtered web text, difficult enough to separate architectures (TinyStories is too easy for that purpose; use it only for smoke tests).
- **Token budgets:** ~1B tokens for the ~58M-parameter screening runs, ~2B tokens for ~100M confirmation (≈ 20 tokens/param). The 10BT sample suffices.
- **Tokenizer:** GPT-2 BPE (`tiktoken`), vocab padded to 50304, **fixed for all arms**. Note: at ~58M total params the tied embedding table (~26M) is about half the model. Keep embeddings identical (unstructured) in every arm and report **non-embedding parameters** as the comparison axis.
- **Validation:** 20M held-out tokens from a shard never trained on; report val loss (nats/token) at fixed token counts. Also report WikiText-103 validation perplexity (out-of-distribution) and, at 100M, lm-eval 0-shot `arc_easy`, `hellaswag`, `piqa` (noisy at this scale; report with confidence intervals).
- **Recall probe (Track B):** multi-query associative recall (MQAR) synthetic task, with 64–512 key–value pairs, sequence length 512–2048, trained jointly as a small auxiliary eval (separate tiny models are acceptable). Linear-attention variants are known to fail here; a hybrid layout should recover it.

### 7.2 Baseline architecture

Llama-style decoder: RMSNorm, RoPE, SwiGLU, tied embeddings, no biases. Suggested screening config (tune to the parameter budget within ±3%): `d_model=512, n_layers=10, n_heads=8 (head_dim 64), SwiGLU hidden 1408, seq_len 2048` → ≈ 58M total (≈ 32M non-embedding). 100M config: `d_model=768, n_layers=12` scaled to hit ~100M total (verify count).

Training defaults: AdamW (β=0.9/0.95, wd 0.1), peak LR 1e-3 (swept per arm, Section 7.6), cosine to 10%, warmup ~200 steps, grad-clip 1.0, batch 262,144 tokens (128 × 2048; ≈ 3.8k steps for 1B tokens), BF16 autocast, `torch.compile`. Initialize quaternion/structured layers so the output variance matches the dense initialization.

### 7.3 Phase 2.0 — Implementation (≈ 1.5 d, no GPU)

Implement, unit-test (against a naive dense reference), and expose via config flags:
- `QuatLinear` (naive: build dense W from 4·(m/4)(n/4) parameters, then a dense matmul; correctness first, speed second).
- `PHMLinear(n=4)`, `LowRankLinear`, and (optional) `MonarchLinear` or block tensor-train linear.
- Sequence-mixer layers from `fla` (GLA with scalar decay, Gated DeltaNet), plus the new **selective-rotation linear attention** variants (Appendix A.4).
- A shared training harness with logging (Rule 8) and deterministic data loaders.

### 7.4 Phase 2.1 — Synthetic state-tracking (≈ 1.5 d) — **GATE G2**

Tiny models (1–2 layers, d = 64–128), minutes per run, 5 seeds.

Tasks: prefix-product of random group elements for **S3, A5, S5** (and optionally Z₆₀ parity-like tasks as sanity). Train on lengths ≤ 32–64, test on lengths up to 256–512 (length generalization). Metric: per-length accuracy; also the minimum layers/width needed for ≥ 95%.

Transition variants (all in the same gated linear-attention/recurrence shell, same state size where possible, param-matched):
1. Scalar/diagonal real decay (GLA-style).
2. Data-dependent **abelian** rotation (complex-diagonal / selective RoPE).
3. Gated DeltaNet (rank-1 Householder-style).
4. DeltaProduct-style (two Householder products), if available in fla.
5. **Quaternion**: data-dependent unit-quaternion rotation per 4-block (Appendix A.4), left-only (SU(2)), and optionally left+right (SO(4)).
6. Dense unconstrained transition (upper bound, small state only).
7. Softmax attention baseline.

Also run the A9 equivalence test here (parallel form vs sequential recurrence) before any training.

**GATE G2:** If the quaternion variant shows a length-generalization advantage over variants 2 and 3 on at least one task (A5 expected), include quaternion arms in Track B at full priority (B3, B6). If it merely matches variant 3, run only B1–B4 and treat the quaternion arm as a cheap ablation. If it is worse than variant 2, cut all quaternion-rotation arms except B3 as a sanity check.

### 7.5 Phase 2.2 — Data prep and noise floor (≈ 1.5 d)

1. Download and tokenize FineWeb-Edu `sample-10BT` into uint16 shards (CPU work; parallelize across cores; check disk space first).
2. Run the baseline transformer at ~58M params for 1B tokens with **3 seeds** (≈ 3 h each at 30 TFLOPS assumed). Record final val loss mean and standard deviation. This standard deviation defines the noise floor for Rule 4.
3. Verify the loss curve is healthy (no spikes), tokens/s matches Phase 0.

### 7.6 LR sweeps (fairness)

Each new arm gets the same 3-point sweep {0.5×, 1×, 2×} of the baseline peak LR on a 250M-token proxy (~0.7–0.9 h each). Pick the best LR by proxy val loss, then run the 1B-token screening run. If time is short, cut to {1×, 2×}. Log which LR each arm used.

### 7.7 Track A — Weight structure, 50-58M scale (≈ 1.7 d incl. sweeps)

All arms replace the **FFN** linears (gate/up/down); attention projections stay dense in the first pass (optional second pass: attention too).

| Arm | Description | Compare on |
|-----|-------------|-----------|
| A0 | Dense (baseline) | reference |
| A1 | Quaternion FFN, **same shapes** as A0 (¼ FFN params) | iso-shape |
| A2 | Dense FFN with hidden size reduced to match A1's parameters | **iso-params vs A1** |
| A3 | Low-rank FFN at A1's parameter count | iso-params |
| A4 | PHM-4 FFN (learned Aᵢ) at A1's parameter count | iso-params |
| A5 | Monarch/butterfly-block FFN at A1's parameter count | iso-params (optional) |
| A6 | Quaternion FFN with hidden ×4 (iso-params vs A0, 4× FFN FLOPs) | information only |

Report for every arm: val loss vs tokens, val loss vs wall-clock, tokens/s, non-embedding params, stored bytes (FP16 and 4-bit equivalents). **Primary comparison: A1 vs A2/A3/A4** (same parameters). Secondary: A6 vs A0 (does a 4×-wider quaternion FFN beat the dense one at the same parameters, and at what FLOP cost).

### 7.8 Track B — Sequence mixer, 50-58M scale (≈ 2.4 d incl. sweeps)

All arms keep dense FFNs and the same parameter count (adjust FFN hidden slightly to compensate for extra mixer parameters).

| Arm | Description |
|-----|-------------|
| B0 | Transformer baseline (= A0; reuse runs) |
| B1 | GLA with scalar decay (all layers) |
| B2 | Gated DeltaNet (all layers) |
| B3 | Selective quaternion-rotation linear attention with scalar decay (all layers) |
| B4 | Selective **abelian** (complex, data-dependent) rotation linear attention with scalar decay (all layers) — the key ablation for non-commutativity |
| B5 | Hybrid 3:1 — GDN ×3 then softmax attention ×1 (mirrors Qwen3.8) |
| B6 | Hybrid 3:1 — quaternion-rotation layer ×3 then softmax attention ×1 |
| B7 | Hybrid 3:1 — GLA ×3 then softmax attention ×1 |

Report val loss, MQAR accuracy, tokens/s (train), decode-time state size in bytes (per layer, per head), and a per-token KV-cache/state footprint at 2K/32K/128K context (analytic). **Fair-comparison note:** B3 adds a rotation-generating projection; account for its parameters (adjust FFN width) and report.

### 7.9 Phase 2.6 — 100M confirmation (≈ 3.7 d) — **GATE G3**

Select up to **3 finalists** from Tracks A and B by: (i) val loss better than baseline-at-iso-params by > 2σ (σ from 2.2), or (ii) equal quality with materially smaller inference state/bytes. Always include the dense baseline and the best non-quaternion structured competitor. Run each at ~100M, 2B tokens, **2 seeds** (~11 h each at 30 TFLOPS assumed; ≈ 88 GPU-h total for 4 configs × 2 seeds). Run a 3rd seed (Phase 2.7) only if |Δ| is between 1σ and 2σ.

Deliverable `reports/part2.md`: tables, learning curves, scaling comparison (58M → 100M: does the gap grow, shrink, or flip?), plots of val loss vs stored bytes, and an explicit recommendation for each of O1 and O2.

**GATE G3:** If no quaternion arm is a finalist, skip Phase K and Part III items 5.4/5.5; still do 5.0–5.3 (they provide the Qwen baseline and diagnostics that don't depend on a quaternion win).

---

## 8. Phase K — Fused decode kernels (≈ 2 d) — **GATE G4**

Run only if a quaternion arm reached the finalists (G3) or 1.2/1.3 found a competitive quaternion use.

1. **Quaternion GEMV kernel** (batch 1, 4, 16): read the 4-parameter blocks, expand the 4×4 structure in registers/shared memory, accumulate in FP32. Variants: FP16 quaternion weights (¼ bytes), INT4 quaternion weights (¹⁄₁₆ bytes).
2. Comparators at **equal bytes**: dense INT4 GEMV (mature kernel, e.g. Marlin-style via vLLM/TensorRT-LLM), dense FP16 GEMV, low-rank two-stage GEMV.
3. **Rotation-linear-attention decode step:** cost of the per-token cumulative quaternion update plus the state update, versus GDN's decode step.
4. Metrics: time per call, achieved GB/s and fraction of the measured peak bandwidth from Phase 0, tokens/s for a full-model decode loop (CUDA graphs) using the 100M models.

Fallback if Triton on the Spark's GPU is unusable: write a CUDA C++/CUTLASS-style kernel, or measure with `torch.compile`-generated kernels and mark results as "not hand-tuned".

**GATE G4:** the quaternion kernel is worth pursuing only if its achieved bandwidth fraction is ≥ 85% of the INT4 dense kernel's **and** the quaternion arm matched or beat the iso-parameter baselines in quality (Part II). Otherwise report that the byte reduction does not convert to latency benefit at iso-quality.

---

## 9. Part III — Qwen3.8-27B experiments

**Known facts to verify (A12):** Qwen3.8 was released in July/August 2026. The flagship Qwen3.8-Max is a ~2.4T-parameter MoE (~95B active) and is **out of scope** for the Spark. Target model: **Qwen3.8-27B**, dense (~27.8B), released ~2026-08-13/14 under Apache 2.0 on Hugging Face, built on the Qwen3.5 architecture: 64 layers = 48 Gated DeltaNet + 16 gated full-attention, 262K native context (extendable), early-fusion vision-language (the language tower is what matters here), MTP speculative decoding supported. Community reports say the GDN modules expose `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`, and that stock LoRA target lists and bitsandbytes 4-bit can misbehave on them. **Read the actual `config.json` and modeling code and correct this paragraph.**

### 5.0 Acquisition and environment (0.5–1 d)

- Locate the official repository ID (do not guess; search the Hugging Face Qwen org), download BF16 (~56 GB) and any official FP8 checkpoint. Check free disk (BF16 + FP8 + NVFP4 copies ≈ 100 GB+).
- Inventory runtimes with Spark (aarch64, Blackwell) support and **Gated DeltaNet + MTP support**: vLLM (NVIDIA container), llama.cpp (GGUF, `--spec-type draft-mtp` per community guides), TensorRT-LLM (NVFP4). Record which runtime supports which format.
- Dump the module tree; count parameters/bytes per module type; confirm A12.

### 5.1 Latency and quality baseline (≈ 1.5 d)

Formats: BF16, official FP8, NVFP4 (reproduce the MINIMA-style W4A4 recipe from arXiv 2609.04098, or use an available NVFP4 build; note which GDN projections stay high-precision), plus INT4 (AWQ/GPTQ-like) if a build exists.

Measure at batch 1 (and 4/8 for reference), with and without MTP speculative decoding:
- Decode tokens/s at context 1K, 8K, 32K, 128K (prompt filled with real text); prefill tokens/s and time-to-first-token at 8K and 32K.
- Peak memory (weights, KV cache, GDN state, activations), separated where the runtime allows.
- Quality sanity: WikiText-2 perplexity (text-only path), a small MMLU subset, GSM8K subset, plus a needle-in-haystack test at 32K/128K. Use identical harness/settings across formats, reasoning mode noted (the model thinks by default; fix the setting and generation limits).
- Roofline: achieved GB/s vs the Phase 0 peak; compute the implied bytes/token and compare.

Deliverable: a table of tokens/s × format × context × MTP, with quality columns. **This is the baseline every structural idea must beat.**

**GATE G5:** Compute the quality gap of NVFP4/INT4 vs BF16. If the average downstream gap ≤ 0.5 points **and** perplexity gap ≤ 2% (rel.), rotation-based methods have little headroom: skip 5.4 and note "quantization already sufficient". Otherwise proceed to 5.4.

### 5.2 Byte-budget profile (≈ 0.5 d)

Compute analytically and confirm by profiling: bytes read per decoded token by module class (GDN projections, GDN state read/write, attention projections, KV cache reads at each context length, FFN, embeddings/lm_head), for each format. Output a stacked bar (per format) and a table. Identify what fraction of decode latency each class represents; this tells which quaternion/rotation idea could even matter.

### 5.3 Weight-projection diagnostic on the 27B model (≈ 1 d)

Repeat Section 1.2 on Qwen3.8-27B, layer by layer using forward hooks and streaming (model in BF16 resident, ~56 GB; capture X per layer for ~64 calibration sequences of 2048 tokens; keep Gram matrices in FP32; free after each layer). Cover **all** module types including the GDN projections (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`), gated attention q/k/v/o, and FFN. Report the same table and apply G1 criteria. Expect a negative result for most layers; document which (if any) pass.

### 5.4 Learned-rotation quantization on Qwen3.8-27B (cond. G5, ≈ 3.5 d)

Only if G5 fired and Part I's rotation experiment (1.3) was at least neutral. Steps:
1. **Folding feasibility check (day 1):** identify norm layers (verify whether RMSNorm uses `1 + w` "zero-centered" scaling and fold correctly), the residual stream layout, and GDN internals (short causal convs, L2-normalized q/k, gated RMS-normalized output, decay/write-strength gates). Rotations can be folded only where the surrounding ops are linear or rotation-equivariant; record which of R₁ (residual), per-head, and online Hadamards are legal. Untie embeddings if needed.
2. Verify function preservation in FP32/BF16 (logits close to original) **before** quantizing.
3. Apply fixed Hadamard R₁ + NVFP4/INT4 W4A4 (or the best format from 5.1) and measure the quality gap. Then optimize a QB-rotation (Section 1.3) on calibration data with the same budget as a SpinQuant-style dense learned rotation; compare.
4. Report quality (same suite as 5.1), decode latency (online-rotation overhead measured), and memory.

### 5.5 Identity-initialized quaternion rotation in GDN layers (cond. G6, stretch)

**G6:** run only if Part II shows a clear quaternion-rotation win (B3/B6 beating B2/B5 with margin > 2σ, on quality or on recall at equal state size) **and** feasibility below holds.

Idea: add to each GDN layer a small module that produces data-dependent unit quaternions (initialized so rotation = identity, i.e. function-preserving at step 0), rotating q/k as in Appendix A.4; freeze the backbone; train only the new projections by distillation from the frozen original model on a small token budget; test perplexity/needle-in-haystack/state-tracking probes.

**Feasibility calculation (do this first):** frozen-backbone training still costs roughly 4 × N × D FLOPs (forward + activation gradients). With N ≈ 2.8e10, D = 5e7 tokens: ≈ 5.6e18 FLOPs ≈ 2+ days at 30 TFLOPS. Larger budgets are infeasible. Because the delta rule and the decay gates interact with the added rotation, function preservation must be verified numerically at initialization. **Preferred alternative:** if a small hybrid Qwen (≈ 1–4B) with the same GDN structure exists, run the experiment there instead (10× cheaper) and extrapolate cautiously.

---

## 10. Decision gates summary

| Gate | Where | Pass condition | If it fails |
|------|-------|----------------|-------------|
| G1 | after 1.2 | Quaternion within 1.1× of best low-rank/PHM error on some layer type | Skip 1.4; Track A still runs |
| G2 | after 2.1 | Quaternion transition shows length-generalization advantage over abelian and Householder baselines | Reduce Track B quaternion arms (Section 7.4) |
| G3 | after 2.6 | A quaternion arm is a finalist beating iso-param baselines by > 2σ | Skip Phase K and Part III 5.4/5.5 |
| G4 | after K.1 | Quaternion kernel ≥ 85% of INT4 kernel's bandwidth fraction and equal-or-better quality | Report "no latency benefit at iso-quality" |
| G5 | after 5.1 | NVFP4/INT4 quality gap > 0.5 points avg or > 2% ppl | Skip 5.4 |
| G6 | before 5.5 | Clear quaternion win in Part II and feasible token budget | Skip 5.5 |

---

## 11. Deliverables and reporting

- `LOG.md`: chronological decisions, deviations, blockers, actual vs estimated time per task.
- `results/`: one JSONL line per run/evaluation with all fields from Rule 8.
- `reports/phase0.md`, `reports/algebra_checks.md`, `reports/part1.md`, `reports/part2.md`, `reports/kernels.md`, `reports/part3.md`.
- `reports/REPORT.md` (final): (1) one-page answer for each objective O1–O4 with evidence and confidence; (2) all tables; (3) a "claims audit": for every **[DERIVED]** and *(assumed)* item, state confirmed / refuted / untested; (4) the recommended next experiments, or a plain statement that the quaternion direction is closed and why.
- Every table row states: model/arm, seed(s), tokens, non-embedding params, stored bytes, val loss / perplexity / accuracy, tokens/s, and comparison baseline.

---

## 12. Risks and fallbacks

| Risk | Fallback |
|------|----------|
| aarch64/Blackwell wheels missing (Triton, flash-attn, fla kernels) | Use NVIDIA containers; use PyTorch SDPA for attention; use fla's torch reference implementations for small models (slow but correct); write CUDA C++ only for K.1 |
| Achieved TFLOPS far below 30 | Rescale token budgets (keep ~20 tokens/param only if affordable; otherwise shrink to 30–40M models); drop LR sweep to 2 points; reduce arms per gates |
| Naive `QuatLinear` too slow in training | Use einsum with the 4 basis tensors or pre-expanded dense W; accept slower training and report it separately from quality |
| Selective-rotation kernels not available | Compute cumulative quaternion product with a log-depth scan (FP32), rotate q/k, then call the standard chunked linear-attention kernel (Appendix A.4) |
| Loss spikes / instability in structured arms | Lower LR, add grad clipping 0.5, try QK-norm; record as a finding |
| Qwen3.8-27B runtime lacks GDN/MTP support on the Spark | Use whichever runtime supports it (community reports cover llama.cpp); note limitations; run the diagnostics (5.2, 5.3) with Hugging Face Transformers instead |
| Disk space | Delete intermediate checkpoints; keep only finalists and final results |
| Quantized-format tooling mishandles GDN modules | Verify module names and shapes explicitly; assert that every intended linear layer was actually quantized (count them) |

---

## Appendix A — Quaternion math reference

Construct everything from the Hamilton product and unit-test it; do not trust hand-typed matrices below.

**A.1 Basics.** q = a + b·i + c·j + d·k, with i² = j² = k² = ijk = −1. Hamilton product p⊗q. Conjugate q̄ = a − bi − cj − dk. For unit q (|q| = 1): q⁻¹ = q̄.

Left multiplication x ↦ q⊗x is the real 4×4 matrix
```
L(q) = [[a, -b, -c, -d],
        [b,  a, -d,  c],
        [c,  d,  a, -b],
        [d, -c,  b,  a]]
```
Right multiplication x ↦ x⊗q is `R(q)` (build it numerically; it differs from L in the signs of the off-diagonal blocks). Properties to test: `L(p)L(q) = L(p⊗q)`, `L` and `R` commute, `L(q)ᵀL(q) = |q|² I`.

**A.2 Block-quaternion linear layer and Frobenius projection.** Split W ∈ ℝ^{m×n} (m, n divisible by 4) into 4×4 blocks B_ij. Constrain B_ij = L(q_ij). The layer stores 4·(m/4)(n/4) = mn/4 parameters. Equivalent view: x is n/4 quaternions, y_i = Σ_j q_ij ⊗ x_j. The Frobenius-optimal projection of a block B onto the quaternion subspace uses the four basis matrices E₀ = L(1), E₁ = L(i), E₂ = L(j), E₃ = L(k), which are orthogonal with ‖E_k‖_F² = 4: `q_k = ⟨B, E_k⟩_F / 4`. Retained energy = Σ‖proj‖²/‖W‖².

**A.3 Activation-aware closed form [DERIVED].** For calibration inputs X (n × T), each column split into n/4 quaternions x_j(t). Since q⊗x = R(x)·q (as a vector map), output block i satisfies y_i(t) = Σ_j R(x_j(t)) q_ij. Stack over t: y_i = M · q_i with M ∈ ℝ^{4T × n} built from the R(x_j(t)) blocks, and q_i ∈ ℝ^n the concatenated parameters of output-block row i. The design matrix M depends only on X, so the normal-equation matrix G = MᵀM (n×n) is **shared by all output blocks**: `q_i = (G + λI)⁻¹ Mᵀ y_i`. One Gram build and one Cholesky per layer solves the optimal activation-weighted quaternion approximation for all rows. Verify against Adam-based optimization on a small layer.

**A.4 Selective-rotation linear attention [DERIVED].** Per head, split the key/query dimension d_k into d_k/4 quaternion blocks. State S_t ∈ ℝ^{d_k × d_v}. Recurrence:
```
S_t = λ_t · Rot(r_t) · S_{t-1} + k_t v_tᵀ,      o_t = S_tᵀ q_t
```
where Rot(r) applies L(r) to each 4-block of the key dimension, r_t = normalize(W_r x_t + bias) is a data-dependent unit quaternion per block (bias initialized to (1,0,0,0) and small W_r so the rotation starts near identity), and λ_t ∈ (0,1] is a scalar decay per head.
Define the cumulative product P_t = r_t ⊗ P_{t−1}, P_0 = 1 (blockwise), computed in FP32 by a log-depth associative scan or chunked scan, with periodic re-normalization. Then, because L(P)⁻¹ = L(P̄) for unit P,
```
o_t = Σ_{s≤t} (Π_{u=s+1..t} λ_u) · v_s · ⟨ L(P̄_s) k_s , L(P̄_t) q_t ⟩
```
i.e. ordinary scalar-decay causal linear attention on **rotated** keys and queries `k̃_s = P̄_s ⊗ k_s`, `q̃_t = P̄_t ⊗ q_t`. Unit-test this equality against the sequential recurrence (float64, random inputs). If it holds, train with standard chunked kernels; for decode, keep the state in the rotated frame and the running P_t (4 numbers per block).
Special cases: restricting r_t to a fixed axis gives complex (abelian) rotation on paired dimensions — data-dependent RoPE-like behaviour (arm B4); fixed r gives a RoPE-like fixed rotation. The non-abelian character only appears when the axis varies with the data. Optional SO(4) variant: apply `p ⊗ x ⊗ q̄` with two unit quaternions per block.

**A.5 Parameter/byte bookkeeping.** Quaternion FP16 weights: mn/4 params → ¼ dense FP16 bytes. Dense INT4 (g128): ≈ ¼ dense FP16 bytes plus scales (~+6%). PHM-4: 4 learned 4×4 matrices + 4 matrices S_i of shape (m/4 × n/4) = mn/4 + 64 params. Low-rank at the same count: r = mn / (4(m+n)). Quaternion + INT4 ≈ ¹⁄₁₆.

---

## Appendix B — Metrics glossary

- **Iso-bytes:** same stored bytes at deployment (including quantization scales).
- **Iso-params:** same non-embedding parameter count in training.
- **σ (noise floor):** standard deviation of final val loss across baseline seeds (Phase 2.2).
- **State size:** bytes per layer per sequence of the recurrent state (GDN/GLA/rotation layers) or the KV cache (attention layers) at a given context length.
- **Bandwidth fraction:** achieved GB/s divided by the measured peak from Phase 0.

## Appendix C — References to locate and verify (do not rely on the author's memory)

- Quaternion/PHM: "Beyond Fully-Connected Layers with Quaternions: PHM" (Zhang et al., 2021); Compacter (Mahabadi et al., 2021); quaternion transformer work (Tay et al.).
- "Compute Better Spent: Replacing Dense Layers with Structured Matrices" (Qiu et al., 2024).
- Mamba-2 / structured state space duality (Dao & Gu, 2024); Gated DeltaNet (Yang et al., 2024); DeltaProduct; RWKV-7 (state tracking); "The Illusion of State in State-Space Models" (Merrill et al.) for state-tracking expressiveness.
- Rotation for quantization: QuaRot, SpinQuant, QuIP#, OFT/BOFT.
- Qwen3.8: Qwen blog (qwen.ai/blog?id=qwen3.8), vLLM Ascend Qwen3.8-27B tutorial page, arXiv 2609.04098 (NVFP4 W4A4 for GDN hybrid; "MINIMA"), arXiv 2608.30320 ("On the Design of Qwen3.8-Next Architecture"). All found via web search on 2026-09-27; snippets only, not fully read.
- Lattice facts (low priority, unverified): D4 ≅ Hurwitz quaternion integers; E8 as a rank-2 module over them; binary tetrahedral group as the 24 unit Hurwitz quaternions.