"""Unit tests for plan.md 1.1 / Appendix A. Checks the [DERIVED] claims before use.

Run: python3 tests/test_algebra.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quat.ops import (  # noqa: E402
    BASIS, Lmat, Rmat, hamilton, conj, normalize,
    build_block_quat, frobenius_project_blocks, block_quat_matmul,
    activation_aware_quat,
)

torch.manual_seed(0)
D = torch.float64
TOL = 1e-10
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def rq(*shape):
    return normalize(torch.randn(*shape, dtype=D))


def rand_quat_params(m4, n4):
    return normalize(torch.randn(m4, n4, 4, dtype=D))


# --- A.2 basic properties ---------------------------------------------------
p, q, x = rq(4), rq(4), rq(4)
check("L(p)L(q)=L(p(x)q)", torch.allclose(Lmat(p) @ Lmat(q), Lmat(hamilton(p, q)), atol=TOL))
check("Lmat(q)x=q(x)x", torch.allclose(Lmat(q) @ x, hamilton(q, x), atol=TOL))
check("Rmat(q)x=x(x)q", torch.allclose(Rmat(q) @ x, hamilton(x, q), atol=TOL))
check("L(p)R(q)=R(q)L(p)", torch.allclose(Lmat(p) @ Rmat(q), Rmat(q) @ Lmat(p), atol=TOL))
check("L unit orthogonal", torch.allclose(Lmat(q).T @ Lmat(q), torch.eye(4, dtype=D), atol=TOL))
check("L(q)^-1=L(conj q)", torch.allclose(Lmat(q).inverse(), Lmat(conj(q)), atol=TOL))
check("conj is inverse", torch.allclose(hamilton(q, conj(q)), torch.tensor([1, 0, 0, 0], dtype=D), atol=TOL))

# --- A.6 structure preservation --------------------------------------------
m, n = 8, 12
W = build_block_quat(rand_quat_params(m // 4, n // 4))
Hr = torch.randn(m // 4, m // 4, dtype=D)
Hc = torch.randn(n // 4, n // 4, dtype=D)
HI_r = torch.kron(Hr, torch.eye(4, dtype=D))
HI_c = torch.kron(Hc, torch.eye(4, dtype=D))


def is_block_quat(M):
    rec = build_block_quat(frobenius_project_blocks(M))
    return torch.allclose(rec, M, atol=1e-9), (M - rec).norm().item()


ok, e = is_block_quat(HI_r @ W)
check("A6: (H kron I4)W stays block-quat", ok, f"resid={e:.2e}")
ok, e = is_block_quat(W @ HI_c)
check("A6: W(H kron I4) stays block-quat", ok, f"resid={e:.2e}")

def diag_unit_quat(dim):
    blocks = [Lmat(rq(4)) for _ in range(dim // 4)]
    DD = torch.zeros(dim, dim, dtype=D)
    for i in range(dim // 4):
        DD[i * 4:(i + 1) * 4, i * 4:(i + 1) * 4] = blocks[i]
    return DD


Dm = diag_unit_quat(m)
Dn = diag_unit_quat(n)
ok, e = is_block_quat(Dm @ W)
check("A6: D W stays block-quat (spec says yes)", ok, f"resid={e:.2e}")
ok, e = is_block_quat(W @ Dn)
check("A6: W D stays block-quat? (spec says NO)", ok, f"resid={e:.2e}")

# --- Frobenius projection / retained energy (A5 sanity) ---------------------
Wr = torch.randn(64, 120, dtype=D)
Wrec = build_block_quat(frobenius_project_blocks(Wr))
ret = (Wrec.norm() / Wr.norm()) ** 2
check("isotropic retained energy ~0.25", abs(ret - 0.25) < 0.05, f"ret={ret:.3f}")

# --- A.3 activation-aware closed form vs exact projection on quat W ---------
A = rand_quat_params(m // 4, n // 4)
Wq = build_block_quat(A)
X = torch.randn(n, 64, dtype=D)
Y = Wq @ X
sol = activation_aware_quat(X, Y)
Yhat = build_block_quat(sol) @ X
rel = (Y - Yhat).norm() / Y.norm()
check("A3 recovers exact quat W from noiseless data", rel < 1e-6, f"relerr={rel:.2e}")

# A.3 vs Frobenius projection on activation-weighted objective
Wgen = torch.randn(m, n, dtype=D)
Yg = Wgen @ X
solA = activation_aware_quat(X, Yg)
solF = frobenius_project_blocks(Wgen)
errA = ((Wgen @ X) - (build_block_quat(solA) @ X)).norm() / Yg.norm()
errF = ((Wgen @ X) - (build_block_quat(solF) @ X)).norm() / Yg.norm()
check("A3 activation-aware <= Frobenius err", errA <= errF + 1e-9, f"act={errA:.4f} frob={errF:.4f}")

# --- A.9 selective-rotation linear attention equivalence --------------------
def sequential(k, qv, v, r, lam):
    T_, dk = k.shape
    dv = v.shape[1]
    S = torch.zeros(dk, dv, dtype=D)
    P = torch.zeros(dk // 4, 4, dtype=D); P[..., 0] = 1.0
    out = []
    for t in range(T_):
        rb = r[t]                                  # (dk/4,4)
        P = hamilton(rb, P)
        Sr = torch.stack([Lmat(rb[b]) @ S[b * 4:(b + 1) * 4] for b in range(dk // 4)])  # (dk/4,4,dv)
        S = lam[t] * Sr.reshape(dk, dv) + torch.outer(k[t], v[t])
        out.append(S.T @ qv[t])
    return torch.stack(out)


def parallel(k, qv, v, r, lam):
    T_, dk = k.shape
    P = []
    Pcur = torch.zeros(dk // 4, 4, dtype=D); Pcur[..., 0] = 1.0
    for t in range(T_):
        Pcur = hamilton(r[t], Pcur)
        P.append(Pcur)
    P = torch.stack(P)                             # (T,dk/4,4)
    kt = torch.stack([hamilton(conj(P[s]), k[s].reshape(dk // 4, 4)).flatten() for s in range(T_)])
    qt = torch.stack([hamilton(conj(P[t]), qv[t].reshape(dk // 4, 4)).flatten() for t in range(T_)])
    out = []
    for t in range(T_):
        acc = torch.zeros(v.shape[1], dtype=D)
        loglam = torch.log(lam)
        cum = torch.cumsum(loglam, 0)
        for s in range(t + 1):
            w = torch.exp(cum[t] - cum[s]) if s < t else torch.tensor(1.0, dtype=D)
            acc = acc + w * (kt[s] @ qt[t]) * v[s]
        out.append(acc)
    return torch.stack(out)


T_, dk, dv = 10, 8, 5
k = torch.randn(T_, dk, dtype=D)
qv = torch.randn(T_, dk, dtype=D)
v = torch.randn(T_, dv, dtype=D)
r = rq(T_, dk // 4, 4)
lam = torch.rand(T_) * 0.5 + 0.5
o_seq = sequential(k, qv, v, r, lam)
o_par = parallel(k, qv, v, r, lam)
rel = (o_seq - o_par).norm() / o_seq.norm()
check("A9 sequential == parallel rotated LA", rel < 1e-6, f"relerr={rel:.2e}")

# block_quat_matmul equals dense build
qb = rand_quat_params(1, n // 4)
xb = torch.randn(n // 4, 4, dtype=D)
check("block_quat_matmul == dense", torch.allclose(
    block_quat_matmul(qb, xb)[0],
    build_block_quat(qb) @ xb.flatten(), atol=TOL))

print("\nSUMMARY:", sum(ok for _, ok, _ in results), "/", len(results), "passed")
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
