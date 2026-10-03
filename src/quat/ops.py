"""Quaternion algebra ops (plan.md Appendix A). Pure PyTorch, works on CPU/GPU.

Convention: quaternions are tensors with last dim 4 = (a, b, c, d) for a + bi + cj + dk.
Everything built from the Hamilton product; matrices L/R are constructed by action on
basis vectors so they cannot be mis-typed.
"""
import torch

BASIS = torch.eye(4, dtype=torch.float64)  # rows are basis quaternions 1, i, j, k


def hamilton(p, q):
    """Hamilton product p (x) q, broadcasting over all leading dims."""
    p0, p1, p2, p3 = p.unbind(-1)
    q0, q1, q2, q3 = q.unbind(-1)
    return torch.stack([
        p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3,
        p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2,
        p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1,
        p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0,
    ], dim=-1)


def conj(q):
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def normalize(q):
    return q / q.norm(dim=-1, keepdim=True)


def Lmat(q):
    """Left-multiplication: Lmat(q) @ x == hamilton(q, x). Shape (...,4,4)."""
    e = BASIS.to(q.dtype).to(q.device)
    return hamilton(q.unsqueeze(-2), e).transpose(-1, -2)


def Rmat(q):
    """Right-multiplication: Rmat(q) @ x == hamilton(x, q). Shape (...,4,4)."""
    e = BASIS.to(q.dtype).to(q.device)
    return hamilton(e, q.unsqueeze(-2)).transpose(-1, -2)


def block_quat_matmul(Wq, x):
    """Apply block-quaternion matrix.

    Wq: (..., m4, n4, 4); x: (..., n4, 4) -> (..., m4, 4).
    y_i = sum_j q_ij (x) x_j.
    """
    return hamilton(Wq, x.unsqueeze(-3)).sum(dim=-2)


def frobenius_project_blocks(W):
    """Project each 4x4 block of W onto the quaternion subspace.

    Basis E_k = Lmat(e_k), orthogonal, ||E_k||_F^2 = 4 => q_k = <B, E_k>/4.
    Returns (m/4, n/4, 4).
    """
    m, n = W.shape
    assert m % 4 == 0 and n % 4 == 0
    B = W.reshape(m // 4, 4, n // 4, 4).permute(0, 2, 1, 3)  # (m4, n4, 4, 4)
    Ek = torch.stack([Lmat(BASIS[k].to(W.device).to(W.dtype)) for k in range(4)])  # (4,4,4)
    return torch.einsum("ijrc,krc->ijk", B, Ek) / 4.0


def build_block_quat(q):
    """Assemble dense (m x n) matrix from quaternion params (m4, n4, 4)."""
    m4, n4, _ = q.shape
    return Lmat(q).permute(0, 2, 1, 3).reshape(m4 * 4, n4 * 4)


def activation_aware_quat(X, Y, lam=1e-6):
    """Closed-form activation-weighted quaternion fit (plan.md A.3).

    X: (n, T) inputs, Y: (m, T) targets. Returns quaternion params (m4, n4, 4).
    y_i(t) = sum_j R(x_j(t)) q_ij = M(t) q_i, M(t) = [R(x_1(t)) | ... | R(x_{n/4}(t))].
    G = sum_t M(t)^T M(t) (n x n) is shared by all output blocks.
    """
    n, T = X.shape
    m = Y.shape[0]
    assert n % 4 == 0 and m % 4 == 0
    n4, m4 = n // 4, m // 4
    xq = X.reshape(n4, 4, T).permute(2, 0, 1).contiguous()  # (T, n4, 4)
    R = Rmat(xq)                                            # (T, n4, 4, 4)
    # G block (a,b) = sum_t R[t,a]^T R[t,b] -> indexed (a, b, x, y)
    Gblk = torch.einsum("tarx,tbry->abxy", R, R)  # (n4,n4,4,4)
    Gmat = Gblk.permute(0, 2, 1, 3).reshape(n, n)
    jitter = lam * (Gmat.diagonal().mean().clamp_min(1e-12))
    Gmat = Gmat + jitter * torch.eye(n, device=X.device, dtype=X.dtype)
    try:
        Gm = torch.linalg.cholesky(Gmat)
    except Exception:
        Gm = torch.linalg.cholesky(Gmat + 1e-3 * Gmat.diagonal().mean().clamp_min(1e-12)
                                   * torch.eye(n, device=X.device, dtype=X.dtype))
    yq = Y.reshape(m4, 4, T).permute(2, 0, 1).contiguous()  # (T, m4, 4)
    # rhs[(b,c), i] = sum_t sum_r R[t,b,r,c] yq[t,i,r]
    rhs = torch.einsum("tbrc,tir->bci", R, yq).reshape(n, m4)
    sol = torch.cholesky_solve(rhs, Gm)          # (n, m4)
    return sol.reshape(n4, 4, m4).permute(2, 0, 1).contiguous()  # (m4, n4, 4)
