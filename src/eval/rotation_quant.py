"""Rotation for low-bit quantization (plan.md 1.3) — assumption #3 / O3.

We test the *quality* effect of rotating each linear's input basis before quantization:
    out = Q4(x R) @ Q4(W R)^T + b
which equals the original matmul exactly when quantization is removed (R orthogonal). This
captures the rotation benefit without QuaRot's norm folding. Methods:
  fp16, w4a16, w4a4 (R=I), w4a4_had (fixed Hadamard), w4a4_rand (fixed random orthogonal),
  w4a4_learn (Cayley-optimized dense R), w4a4_qb (quaternion-butterfly structured R).

Usage: python3 src/eval/rotation_quant.py --model Qwen/Qwen3-0.6B --method w4a4_had
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quat.ops import Lmat, hamilton  # noqa: E402

DEV = "cuda"
ROOT = Path(__file__).resolve().parents[2]


# ----------------------------- rotations ------------------------------------
def hadamard(d):
    assert d & (d - 1) == 0, "power of 2 only"
    H = torch.ones(1, 1)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(d)


def random_orth(d, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R))
    return Q


def fixed_rot(d, kind):
    if kind == "had":
        if d & (d - 1) == 0:
            return hadamard(d)
        # Hadamard-friendly fallback: block-diagonal Hadamards on the largest 2^k divisor
        p = 1
        while (d // p) % 2 == 0 and p * 2 <= d:
            p *= 2
        blocks = d // p
        R = torch.zeros(d, d)
        for b in range(blocks):
            R[b * p:(b + 1) * p, b * p:(b + 1) * p] = hadamard(p)
        return R
    if kind == "rand":
        return random_orth(d)
    raise ValueError(kind)


def qb_build(params, d):
    """Quaternion-butterfly R = prod_s [G_s . Diag(L(p))] . H  (plan 1.3).

    G_s acts as a Givens rotation in n-block space (identical for the 4 components) = P_s (x) I4.
    params = [theta (S * n/2)] ++ [quats (S * n * 4)]. Requires n = d/4 a power of two.
    """
    n = d // 4
    S = int(math.log2(n))
    theta = params[: S * (n // 2)].reshape(S, n // 2)
    q = params[S * (n // 2):].reshape(S, n, 4)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    R = hadamard(d).to(params.device, params.dtype)
    eye4 = torch.eye(4, dtype=params.dtype, device=params.device)
    for s in range(S):
        step = 1 << s
        P = torch.eye(n, dtype=params.dtype, device=params.device)
        for b in range(n):
            if b & step:
                continue
            b2 = b ^ step
            pidx = (b // (2 * step)) * step + (b % step)
            c, sn = torch.cos(theta[s, pidx]), torch.sin(theta[s, pidx])
            P[b, b], P[b, b2] = c, -sn
            P[b2, b], P[b2, b2] = sn, c
        G = torch.kron(P, eye4)
        Lq = Lmat(q[s])                                      # (n,4,4)
        Rq = torch.zeros(d, d, dtype=params.dtype, device=params.device)
        for i in range(n):
            Rq[i * 4:(i + 1) * 4, i * 4:(i + 1) * 4] = Lq[i]
        R = G @ Rq @ R
    return R


def qb_init(d):
    n = d // 4
    S = int(math.log2(n))
    p = torch.zeros(S * (n // 2) + S * n * 4)
    p[S * (n // 2):].reshape(S * n, 4)[:, 0] = 1.0
    return p


def _block_diag_quat(Lq):
    n = Lq.shape[0]
    d = n * 4
    R = torch.zeros(d, d, dtype=Lq.dtype, device=Lq.device)
    for i in range(n):
        R[i * 4:(i + 1) * 4, i * 4:(i + 1) * 4] = Lq[i]
    return R


# ----------------------------- quantization ---------------------------------
def q4_weight(W, group=128):
    m, n = W.shape
    pad = (group - n % group) % group
    Wp = F.pad(W, (0, pad)).reshape(m, -1, group)
    scale = Wp.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7.0
    q = (Wp / scale).round().clamp(-8, 7)
    return (q * scale).reshape(m, -1)[:, :n]


def q4_act(x):
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7.0
    return (x / scale).round().clamp(-8, 7) * scale


class RotQuantLinear(nn.Module):
    def __init__(self, lin, rot=None, weight_bits=4, act_bits=4):
        super().__init__()
        self.weight = lin.weight
        self.bias = lin.bias
        self.rot = rot
        self.weight_bits = weight_bits
        self.act_bits = act_bits

    def forward(self, x):
        W = self.weight
        if self.rot is not None:
            R = self.rot
            xr = x @ R.t()
            Wr = W @ R.t()
        else:
            xr, Wr = x, W
        qw = q4_weight(Wr.float()).to(Wr.dtype) if self.weight_bits == 4 else Wr
        if self.act_bits == 4 and self.training:
            qx = q4_act(xr.float()).to(xr.dtype)
        elif self.act_bits == 4:
            qx = q4_act(xr.float()).to(xr.dtype)
        else:
            qx = xr
        return F.linear(qx, qw, self.bias)


# ----------------------------- apply / eval ---------------------------------
def rotate_model(model, kind, per_dim=True):
    rots = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and name.startswith("model.layers"):
            d = m.in_features
            if d not in rots:
                rots[d] = fixed_rot(d, kind) if kind in ("had", "rand") else None
            m.forward = _make_forward(m, rots[d], 4, 4)


def _make_forward(lin, R, wb, ab):
    W = lin.weight
    if R is not None:
        Rf = R.to(W.device, torch.float32)
        Wr = W.float() @ Rf.t()
    else:
        Rf, Wr = None, W.float()
    qw = (q4_weight(Wr).to(W.dtype) if wb == 4 else Wr.to(W.dtype)).detach()

    def fwd(x):
        xr = (x @ Rf.t()) if Rf is not None else x
        qx = (q4_act(xr.float()).to(x.dtype) if ab == 4 else xr)
        return F.linear(qx, qw, lin.bias)
    return fwd


@torch.no_grad()
def perplexity(model, ids, seq=1024):
    model.eval()
    nll, ntok = 0.0, 0
    for i in range(0, len(ids) - seq, seq):
        x = ids[i:i + seq].unsqueeze(0).to(DEV)
        y = ids[i + 1:i + seq + 1].unsqueeze(0).to(DEV)
        logits = model(x).logits
        nll += F.cross_entropy(logits[0].float(), y[0], reduction="sum").item()
        ntok += y.numel()
    return math.exp(nll / ntok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--method", default="w4a4_had")
    ap.add_argument("--eval_seqs", type=int, default=40)
    ap.add_argument("--out", default="results/rotation_quant.jsonl")
    args = ap.parse_args()
    import os
    os.environ.setdefault("HF_HOME", str(ROOT / "checkpoints/hf"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(DEV).eval()
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(t for t in test["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][: args.eval_seqs * 1024]

    if args.method == "fp16":
        ppl = perplexity(model, ids)
    elif args.method == "w4a16":
        _install(model, None, 4, 16)
        ppl = perplexity(model, ids)
    elif args.method == "w4a4":
        _install(model, None, 4, 4)
        ppl = perplexity(model, ids)
    elif args.method in ("w4a4_had", "w4a4_rand"):
        _install(model, "had" if args.method.endswith("had") else "rand", 4, 4)
        ppl = perplexity(model, ids)
    else:
        raise SystemExit(f"method {args.method} not implemented in this pass")
    print(f"{args.model} {args.method}: ppl={ppl:.3f}")
    with open(ROOT / args.out, "a") as f:
        f.write(json.dumps({"model": args.model, "method": args.method, "ppl": ppl}) + "\n")


def _install(model, rot_kind, wb, ab):
    rots = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and name.startswith("model.layers"):
            d = m.in_features
            if d not in rots:
                rots[d] = fixed_rot(d, rot_kind) if rot_kind else None
            m.forward = _make_forward(m, rots[d], wb, ab)


if __name__ == "__main__":
    main()
