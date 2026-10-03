"""Synthetic state-tracking with group prefix products (plan.md 2.1).

Task: input tokens a_1..a_T in a finite group G; output at each position t is the
prefix product a_1 * a_2 * ... * a_t. Length generalization is the metric.

Model shell: recurrent state h_t = A_t h_{t-1}, h_0 learned; logits_t = W_out h_t.
The only difference between arms is the family A_t is drawn from, given token embedding.
This isolates transition expressiveness (plan A10/A11), which is what assumption #1 needs.

Run: python3 src/models/state_tracking.py --group A5 --seeds 3
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quat.ops import Lmat  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "results"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------- groups ---------------------------------------
def _parity(p):
    seen = [False] * len(p)
    par = 0
    for i in range(len(p)):
        if not seen[i]:
            j, c = i, 0
            while not seen[j]:
                seen[j] = True
                j = p[j]
                c += 1
            par += c - 1
    return (-1) ** par


def group_table(name):
    if name == "S3":
        perms = list(itertools.permutations(range(3)))
    elif name == "S5":
        perms = list(itertools.permutations(range(5)))
    elif name == "A5":
        perms = [p for p in itertools.permutations(range(5)) if _parity(p) == 1]
    else:
        raise ValueError(name)
    perms = sorted(perms)
    idx = {p: i for i, p in enumerate(perms)}
    n = len(perms)
    mult = torch.zeros(n, n, dtype=torch.long)
    for i, p in enumerate(perms):
        for j, q in enumerate(perms):
            comp = tuple(p[q[k]] for k in range(len(p)))
            mult[i, j] = idx[comp]
    ident = idx[tuple(range(len(perms[0])))]
    return perms, mult, ident


def blockdiag_bt(blocks):
    """blocks (B,T,nb,k,k) -> block-diagonal (B,T,nb*k,nb*k)."""
    B, T, nb, k, _ = blocks.shape
    out = blocks.new_zeros(B, T, nb * k, nb * k)
    for i in range(nb):
        out[:, :, i * k:(i + 1) * k, i * k:(i + 1) * k] = blocks[:, :, i]
    return out


def cayley(A):
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return (eye + A) @ torch.linalg.inv(eye - A)


# ----------------------------- transition families --------------------------
class Transition(nn.Module):
    def __init__(self, name, d_emb, d):
        super().__init__()
        self.name, self.d = name, d
        sizes = {"scalar": 1, "diagonal": d, "abelian": d // 2, "quat": d,
                 "householder": d, "deltaproduct": 2 * d,
                 "orth": d * (d - 1) // 2, "dense": d * d}
        self.proj = nn.Linear(d_emb, sizes[name])
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        if name == "quat":
            with torch.no_grad():
                self.proj.bias.copy_(torch.tensor([1.0, 0, 0, 0]).repeat(d // 4))
        if name in ("householder", "deltaproduct"):
            with torch.no_grad():
                self.proj.bias.copy_(torch.zeros(sizes[name]))
                self.proj.bias[:d] = 0.0

    def forward(self, e):
        B, T, _ = e.shape
        d = self.d
        z = self.proj(e)
        eye = torch.eye(d, device=e.device, dtype=e.dtype)
        if self.name == "scalar":
            return eye[None, None] * torch.sigmoid(z).unsqueeze(-1)
        if self.name == "diagonal":
            return torch.diag_embed(torch.sigmoid(z))
        if self.name == "abelian":
            th = z
            c, s = torch.cos(th), torch.sin(th)
            R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # B,T,nb,2,2
            return blockdiag_bt(R)
        if self.name == "quat":
            nb = d // 4
            q = F.normalize(z.reshape(B, T, nb, 4), dim=-1)
            return blockdiag_bt(Lmat(q))                       # B,T,nb,4,4
        if self.name == "householder":
            v = F.normalize(z, dim=-1, eps=1e-5)
            return eye - 2 * v.unsqueeze(-1) * v.unsqueeze(-2)
        if self.name == "deltaproduct":
            v1, v2 = z[..., :d], z[..., d:]
            v1 = F.normalize(v1, dim=-1, eps=1e-5)
            v2 = F.normalize(v2, dim=-1, eps=1e-5)
            H1 = eye - 2 * v1.unsqueeze(-1) * v1.unsqueeze(-2)
            H2 = eye - 2 * v2.unsqueeze(-1) * v2.unsqueeze(-2)
            return H2 @ H1
        if self.name == "orth":
            A = torch.zeros(B, T, d, d, device=e.device, dtype=e.dtype)
            k = 0
            for i in range(d):
                for j in range(i + 1, d):
                    A[..., i, j] = z[..., k]
                    k += 1
            A = A - A.transpose(-1, -2)
            return cayley(A)
        if self.name == "dense":
            return z.reshape(B, T, d, d)
        raise ValueError(self.name)


class StateTracking(nn.Module):
    def __init__(self, n_tokens, variant, d_emb=64, state_dim=4, head_hidden=0):
        super().__init__()
        self.emb = nn.Embedding(n_tokens, d_emb)
        self.trans = Transition(variant, d_emb, state_dim)
        self.h0 = nn.Parameter(torch.randn(state_dim) * 0.1)
        self.norm = nn.LayerNorm(state_dim)
        if head_hidden:
            self.head = nn.Sequential(nn.Linear(state_dim, head_hidden), nn.GELU(),
                                      nn.Linear(head_hidden, n_tokens))
        else:
            self.head = nn.Linear(state_dim, n_tokens)
        nn.init.normal_(self.emb.weight, std=1.0)

    def forward(self, x, return_states=False):
        B, T = x.shape
        e = self.emb(x)
        A = self.trans(e)
        h = self.h0[None].expand(B, -1).contiguous()
        outs, states = [], []
        for t in range(T):
            h = torch.einsum("bij,bj->bi", A[:, t], h)
            outs.append(self.head(self.norm(h)))
            states.append(h)
        logits = torch.stack(outs, 1)
        return (logits, torch.stack(states, 1)) if return_states else logits


# ----------------------------- data / train ----------------------------------
def make_batch(mult, ident, B, T):
    n = mult.shape[0]
    x = torch.randint(0, n, (B, T))
    y = torch.empty(B, T, dtype=torch.long)
    cur = torch.full((B,), ident, dtype=torch.long)
    for t in range(T):
        cur = mult[cur, x[:, t]]
        y[:, t] = cur
    return x, y


def train_eval(group, variant, seed, state_dim=4, steps=3000, lr=3e-3,
               train_len=32, batch=128, d_emb=64, head_hidden=0):
    torch.manual_seed(seed)
    perms, mult, ident = group_table(group)
    n = mult.shape[0]
    model = StateTracking(n, variant, d_emb, state_dim, head_hidden).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    loss = torch.tensor(float("nan"))
    for step in range(steps):
        Tlen = torch.randint(1, train_len + 1, (1,)).item()
        x, y = make_batch(mult, ident, batch, Tlen)
        x, y = x.to(DEV), y.to(DEV)
        loss = F.cross_entropy(model(x).reshape(-1, n), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    accs = {}
    with torch.no_grad():
        for L in [8, 16, 32, 48, 64, 96, 128, 256]:
            x, y = make_batch(mult, ident, 256, L)
            x, y = x.to(DEV), y.to(DEV)
            pred = model(x).argmax(-1)
            accs[str(L)] = {"last": (pred[:, -1] == y[:, -1]).float().mean().item(),
                            "mean": (pred == y).float().mean().item()}
    return {"group": group, "variant": variant, "seed": seed, "n": n,
            "params": sum(p.numel() for p in model.parameters()),
            "train_len": train_len, "steps": steps, "wall_s": time.time() - t0,
            "acc": accs, "final_loss": float(loss)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="A5")
    ap.add_argument("--variants", nargs="+",
                    default=["scalar", "diagonal", "abelian", "householder", "quat", "so4"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--state_dim", type=int, default=4)
    ap.add_argument("--head_hidden", type=int, default=0)
    ap.add_argument("--train_len", type=int, default=32)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d_emb", type=int, default=64)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    out = []
    for v in args.variants:
        for s in range(args.seeds):
            r = train_eval(args.group, v, s, state_dim=args.state_dim, steps=args.steps,
                           head_hidden=args.head_hidden, train_len=args.train_len,
                           batch=args.batch, lr=args.lr, d_emb=args.d_emb)
            out.append(r)
            a = r["acc"]
            print(f"[{args.group}] {v:14s} s{s} p={r['params']:6d} "
                  f"L32={a['32']['last']:.3f} L64={a['64']['last']:.3f} "
                  f"L128={a['128']['last']:.3f} L256={a['256']['last']:.3f} ({r['wall_s']:.0f}s)")
            sys.stdout.flush()
    RESULTS.mkdir(exist_ok=True)
    tag = args.tag or args.group
    with (RESULTS / f"statetracking_{tag}.jsonl").open("a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in out))


if __name__ == "__main__":
    main()
