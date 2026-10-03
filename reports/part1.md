# Part I — Pretrained-model experiments

Model: **Qwen3-0.6B** (28 layers, hidden 1024, 16 Q / 8 KV heads, head_dim 128, tied embeddings,
no q/k/v biases, QK-norm). Calibration: WikiText-2 *train*; evaluation: WikiText-2 *test*.
Device: GB10 (DGX Spark).

## 1.2 Weight-projection diagnostic (O1)

See `reports/EXPERIMENTS.md` (E3) for the full table. Summary:

1. **A5 is exactly true.** For every linear layer and depth, projecting the pretrained weight onto
   the block-quaternion subspace retains **0.250** of the Frobenius energy — the isotropic value.
   Pretrained weights are maximally far from quaternion-structured, i.e. they carry no quaternion
   structure to exploit.
2. Plain Frobenius quaternion projection is useless (err ≈ 0.86 = sqrt(1−0.25)).
3. The **activation-aware** quaternion projection (A.3) cuts the error substantially — it is
   competitive with activation-aware low-rank on the **output** projections `o_proj` and
   `down_proj` (G1 passes 5/5 each) but loses on `q/k/gate/up` and ties on `v`.
4. **Dense INT4 (group 128) beats both** at equal stored bytes in 6/7 projection types.
5. End-to-end, replacing all layers of a type collapses perplexity for both quaternion and
   low-rank projections (millions / hundreds), while INT4 stays within +0.7–2.7 ppl.

**Gate G1: FAIL on the ppl criterion.** The pretrained weight-structure track is closed (skip 1.4).
Consistent with A4/A5. Track A from scratch still runs.

Caveat / not yet done: this is one model (0.6B) and one calibration set; the 27B `5.3` run is
pending and may differ for the Gated-DeltaNet projections. The qualitative conclusion (isotropic
weights, INT4 dominance) is unlikely to flip, but the o_proj/down_proj quaternion competitiveness is
the one thing worth re-checking at scale.

## 1.3 Rotation-for-quantization (assumption #3) — PENDING
## 1.4 Layer-wise distillation — skipped (G1 not passed)
