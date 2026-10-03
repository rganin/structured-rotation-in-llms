# Part II (partial) — Track B sequence mixers at LM scale (plan 2.5)

**Question (O2 / assumption #1):** does the selective quaternion-rotation linear attention —
the plan's new architecture, whose state-tracking advantage passed Gate G2 synthetically — improve
natural-language modeling or recall at equal parameters?

## Setup
`src/models/train_mixer.py`. Llama-style decoder d=512, 6 layers, 8 heads, seq 512, RMSNorm,
SwiGLU, tied embeddings. All mixers are exact **parallel (quadratic) linear attention** so the
semantics are controlled; the rotation variants rotate keys/queries with the cumulative frame P_t
(plan A9):
```
scalar : S_t = exp(g_t) S_{t-1} + k_t v_t^T,              o_t = q_t^T S_t
complex: same with a data-dependent abelian 2x2 rotation of the state
quat   : same with a data-dependent non-abelian unit-quaternion rotation
fixed  : same with a fixed (non-data-dependent) block rotation
attn   : causal softmax attention + RoPE (baseline)
```
Correctness verified against the sequential recurrence (max error ≤ 3.6e-15 for scalar, quat,
complex). FFN width is auto-adjusted so every arm has **~19.27M non-embedding parameters**.
Data: WikiText-103 GPT-2, 20M tokens, AdamW 1e-3 cosine, batch 64×512, bf16.

## Results (validation loss, lower is better)

| Mixer | seed 0 | seed 1 | mean | comment |
|---|---|---|---|---|
| scalar (abelian decay) | 4.5644 | 4.5841 | 4.5743 | baseline |
| **quat (non-abelian rotation)** | 4.5980 | 4.5299 | 4.5640 | order flips across seeds |
| complex (abelian rotation) | 4.6384 | — | — | worst |
| attn | 4.6168 | — | — | |
| fixed rotation | 6.544* | — | — | *at 1.5M only, not comparable |

**Seed noise**: the quat–scalar difference is −0.010 on the mean but the per-seed spread is
~0.03–0.05, so by Rule 4 the result is **inconclusive** — there is **no evidence** that the
non-abelian rotation improves LM quality. Abelian rotation is clearly *worse* than no rotation.

## MQAR recall probe
`src/eval/mqar.py`, tiny 2-layer d128 model, sequences `[k1,v1,…,kN,vN,q1…qQ]`.
- 64 pairs: all mixers at chance (linear-attention capacity limit, as the plan predicted).
- 16 pairs with an open decay gate: scalar 0.097 / quat 0.097 accuracy (chance 0.008) — equal.
- **No recall advantage for the quaternion rotation.**

## State size / inference
The recurrent state is identical in all arms (d_k × d_v per head, per layer). At decode, the
quaternion variant additionally carries the running frame `P_t` (d_k numbers per head) — so it is
**not** more byte-efficient than scalar decay. There is no inference win here.

## Interpretation
The O2 hypothesis is confirmed *only* for the synthetic state-tracking setting: non-abelian
quaternion transitions are a genuine, low-dimensional inductive bias for tracking non-solvable
group structure (A5). When the task is natural-language prediction, that extra structure is not
used and does not help; at equal parameters the model is neutral-to-slightly worse. This is an
honest negative for the "new architecture as a general LM improvement" reading.

## Remaining to fully settle O2
- Gated DeltaNet baseline (blocked: `fla` `chunk_gla`/gate appears broken on this build).
- Hybdrids B5–B7 (3:1 mixer:attention) mirroring Qwen3.8 — where recall is recovered.
- Larger scale / ≥1B tokens and the `force_hidden` FFN control.
- A task that actually needs non-commutative state tracking (e.g. an LM with a group-tracking
  auxiliary objective), rather than plain next-token prediction.
