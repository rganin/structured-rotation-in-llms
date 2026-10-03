# Part II (partial) — Track A weight structure from scratch (plan 2.4)

**Question (O1 / assumption #1 weights):** does replacing dense FFN linears with block-quaternion
linears help at equal parameter count, and could ¼-byte quaternion weights buy decode bandwidth?

## Setup
`src/models/train_lm.py`. Llama-style decoder: d=512, 6 layers, 8 heads, RoPE, RMSNorm, SwiGLU,
tied embeddings, seq 512. Data: WikiText-103, tiktoken GPT-2, 30M tokens per run, AdamW lr 1e-3
(cosine, 10% floor, 100-step warmup), bf16 autocast, grad clip 1.0, batch 96×512. Three seeds for
the primary arms A1/A2. `QuatLinear` stores `(out/4 × in/4 × 4)` params and rebuilds the dense
`L(q)` block matrix each forward (so FLOPs match a dense layer of the same hidden width).

| Arm | FFN | hidden | non-emb params | FFN stored bytes (fp16) | val loss |
|---|---|---|---|---|---|
| A0 | dense | 1408 | 19.27M | 1× | 4.404 (seed0) |
| **A1** | quaternion | 1408 | **9.54M** | **¼×** | **4.519 ± 0.031** (3 seeds) |
| A2 | dense | 352 | **9.54M** | 1× | 4.576 ± 0.011 (3 seeds) |
| A3 | low-rank | 1408 | 9.55M | 1× | 4.632 (seed0) |
| A6 | quaternion | 5632 | 19.27M | 1× (=A0) | 4.364 (seed0) |

## Findings
1. **A1 vs A2 (the primary iso-parameter test).** Quaternion wins all three paired seeds;
   paired difference −0.057 nats (paired t ≈ 1.95). This is borderline against the plan's 2σ rule
   (A1 σ=0.031 → 2σ=0.062 > 0.057; A2 σ=0.011 → 2σ=0.022 < 0.057). **Verdict: weak positive,
   likely a small real parameter-efficiency gain but not established at this budget.**
2. **The FLOPs caveat is decisive for inference.** A1 only has 352 effective FFN width worth of
   parameters; A2 has the same parameter count but a 352-wide dense matmul, while A1 materializes a
   1408-wide matmul → A1 costs ~4× the FFN FLOPs. At iso-params the quaternion arm is not
   iso-compute.
3. **A6 vs A0.** A 4×-wider quaternion FFN at the *same* parameter count and stored bytes as dense
   improves val loss by 0.040 (4.364 vs 4.404), at 4× FFN FLOPs. So the structure can beat dense
   when given the same parameter budget — but not for free.
4. **Low-rank (A3) is clearly worse** than both.

## Inference implication
A quaternion model at dense shapes (A1) reaches 4.519 while storing ¼ the FFN bytes, vs dense fp16
A0 at 4.404. Dense **INT4** at A0 shapes also stores ~¼ bytes and (per Part I, E3) quantizes better
than quaternion projections, so there is still **no convincing inference-bytes advantage** for
quaternion weights; the win would have to come from quality at iso-params, which is marginal here.

## Not done (scope cut, plan 7.6/2.6)
- LR sweep per arm, 1B-token runs, and the 100M confirmation (Gate G3).
- Track B (sequence mixer) at LM scale.
- Attention projections replaced (only FFN done).

## Addendum — Tay et al. 2019 (arXiv:1906.04393) replication and correction

We initially tested quaternion **FFN only** and concluded the from-scratch win was marginal. The
Tay et al. paper's actual contribution also includes **quaternion attention** (Q,K,V,O as
Hamilton-product linear maps) and a "full" quaternion transformer. Implementing that changes the
conclusion:

| Arm (30M tok, 3 seeds where noted) | hidden | non-emb | val loss |
|---|---|---|---|
| dense / dense (A0) | 1408 | 19.27M | 4.445 ± 0.036 (n=3) |
| **quat attn / quat FFN (iso-params)** | **7680** | **19.27M** | **4.333 ± 0.015 (n=3)** |
| quat attn / quat FFN (75% params) | 1408 | 4.82M | 4.550 (n=1) |
| quat attn / dense FFN (iso-params) | 1920 | 19.27M | 4.426 (n=1) |
| dense attn / quat FFN (iso-params) | 5632 | 19.27M | 4.364 (n=1) |

Paired difference (full quat − dense) **−0.112 nats, t≈6.5** → a significant iso-parameter win,
at +28% training wall time despite 4× FFN FLOPs. This reproduces the paper's central claim at LM
scale and corrects the FFN-only conclusion. Remaining caveats: one model size/corpus, 4× FLOPs,
and the comparison is iso-params (not iso-FLOPs). The distinctive untested mechanism is the
paper's **Hamilton-product attention with component-wise softmax** (we used standard softmax).
