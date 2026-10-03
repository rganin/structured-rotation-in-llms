# Algebra checks (plan.md 1.1 / Appendix A)

Run: `python3 tests/test_algebra.py` — **16/16 pass** (float64, tol 1e-10 unless noted).

| Check | Result | Notes |
|---|---|---|
| `L(p)L(q) = L(p⊗q)` | PASS | |
| `Lmat(q)x = q⊗x`, `Rmat(q)x = x⊗q` | PASS | |
| `L(p)R(q) = R(q)L(p)` | PASS | |
| Unit `L(q)` orthogonal; `L(q)⁻¹ = L(q̄)` | PASS | |
| A6 `(H⊗I₄)W`, `W(H⊗I₄)` block-quat | PASS | residual 0 |
| A6 `D·W` block-quat (spec: yes) | PASS | |
| A6 `W·D` block-quat **(spec: NO)** | **PASS → spec wrong** | `L(q_ij)L(p_j)=L(q_ij⊗p_j)` |
| A5 isotropic retained energy | PASS | 26.5% of 64×120 gaussian |
| A3 recovers exact quat W | PASS | rel 3.6e-9 |
| A3 activation-aware ≤ Frobenius | PASS | 0.851 vs 0.885 |
| A9 seq == parallel rotated LA | PASS | rel 5e-8 |

## Corrections to the spec
1. **A6, second clause is false.** For block-diagonal `D = diag(L(p_1),…,L(p_n/4))`, both `D·W`
   and `W·D` remain block-quaternion. The set of block-quaternion (left-mult) matrices is a
   subalgebra; multiplication is closed on either side. Non-commutativity changes the resulting
   quaternions but not membership. Implication: no constraint is lost when multiplying a
   quaternion-structured weight by quaternion-diagonal structure on the right; the "right
   multiplication breaks structure" caveat in Section 1.1/A6 should be removed. (Right-
   multiplication by a *general* real matrix is still not allowed, only `H⊗I₄`-type.)

## Verified enablers
- **A9** lets selective quaternion-rotation linear attention be trained with **standard chunked
  scalar-decay linear-attention kernels** (rotate q/k by the cumulative quaternion frame first).
  This removes the need for custom kernels to *train* the B3/B6 arms.
- **A3** gives a one-Cholesky-per-layer optimal activation-weighted quaternion projection, usable
  for the 1.2 diagnostic and for weight init in later phases.
