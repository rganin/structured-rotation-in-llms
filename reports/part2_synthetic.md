# Part II (partial) — Synthetic state tracking (plan 2.1, GATE G2)

**Question (assumption #1):** does a data-dependent **unit-quaternion** transition in a recurrent
sequence mixer give a real advantage over scalar/abelian decay and over Householder/delta-rule
transitions?

## Setup
`src/models/state_tracking.py`. Task: sequence of tokens `a_1..a_T` drawn from a finite group `G`;
at each position predict the prefix product `a_1·…·a_t`. The recurrence is
`h_t = A_t h_{t-1}`, `h_0` learned, then a shared 2-layer MLP readout. **Only** the family from
which the transition `A_t` is drawn (as a function of the token embedding) changes between arms.
Train length ≤ 8, test length up to 256 → measures length generalization. 12k steps, Adam 3e-3,
state_dim 8, d_emb 128, 1 GPU (GB10).

Representations: `quat` = block-diagonal `L(q)` for unit quaternions (2 blocks of 4);
`abelian` = block-diagonal 2×2 rotations (commutative); `householder` = `I-2vvᵀ` (GDN-like);
`deltaproduct` = product of two reflections; `orth` = dense `SO(8)` via Cayley (expressivity
upper bound); `dense` = unconstrained; `scalar`/`diagonal` = decay.

## Results (last-position accuracy at length 256; chance: A5 1.7%, S3 17%, S5 0.8%)

| Group | Variant | Seeds | Success (>0.9) | Best L256 | Median |
|---|---|---|---|---|---|
| A5 | scalar | 3 | 0 | 0.02 | ~0.01 |
| A5 | diagonal | 3 | 0 | 0.02 | ~0.01 |
| A5 | abelian | 3 | 0 | 0.04 | 0.02 |
| A5 | householder | 3 | 0 | 0.02 | 0.02 |
| A5 | deltaproduct | 8 | 1 | 1.00 | ~0.02 |
| **A5** | **quat** | **10** | **9** | **1.000** | **1.000** |
| A5 | orth (SO8) | 8 | 0 | 0.09 | 0.03 |
| S3 | **quat** | 2 | **2** | **1.000** | 1.000 |
| S3 | orth/abelian/householder | 2-3 | 0 | ≤0.25 | ~0.2 |
| S5 | quat/orth/deltaproduct | 2 each | 0 | 0.02 | ~0.01 |

Representative curves (A5, state_dim 8, train_len 8, 12k steps):

```
quat (seed 0):  L32=1.000 L64=1.000 L128=1.000 L256=1.000
orth (seed 1):  L32=0.609 L64=0.312 L128=0.160 L256=0.078
deltaproduct:   L32=1.000 L64=1.000 L128=1.000 L256=0.996   (1 of 8 seeds)
abelian:        L32=0.023 L64=0.012 L128=0.008 L256=0.035
```

## Interpretation
- **Positive for assumption #1 / A10.** The unit-quaternion transition is the only family that
  reliably tracks the smallest non-solvable group A5 and extrapolates ~32× beyond training length.
  The mechanism is the expected one: the binary icosahedral group `2I ⊂ SU(2)` double-covers A5,
  so a 4-real-dimensional quaternion block carries a faithful projective action.
- **Important caveat (do not overclaim).** Dense `SO(8)` *contains* a faithful A5 action, so the
  quaternion win is an **optimization / inductive-bias** win, not a strict expressivity win over
  dense orthogonal maps. There is no `err_Q` expressivity theorem here; the claim is "quaternion
  structure makes non-abelian state tracking learnable from data, where generic orthogonal
  parameterizations fail to optimize."
- **Boundary.** S5 fails for every family at this scale; consistent with plan A10 (left-only
  quaternion gives at most `SU(2)^k`, which cannot host a faithful S5 action). S5 would need a
  ≥5-dimensional faithful representation, i.e. not quaternion blocks.
- DeltaProduct occasionally solves A5 (1/8) but is far less reliable; scalar/diagonal/abelian and
  single-reflection Householder never do.
- The A9 equivalence (E1) means this transition can be realized inside a standard scalar-decay
  chunked linear-attention kernel, so the Track B arm `B3` is trainable without custom kernels.

## Gate decision
**G2 PASSES.** Quaternion arms `B3`, `B6` run at full priority in Track B.

## Threats / next checks
1. Confirm the optimization effect is not an artifact of the identity-biased init: `orth` also
   starts near identity (Cayley with `‖z‖≈0`), so the comparison is fair; still, run a quat arm
   with random init to be safe.
2. Multi-layer / multi-head shell, and the literal A.4 linear-attention shell (not just the
   transition RNN), on A5.
3. Seed count for `deltaproduct` and `orth` is modest but the separation is large.
