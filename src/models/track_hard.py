"""Hard state tracking: multi-layer transition recurrence + distractor tokens (plan 2.1 / O2).

Stream = group tokens with random noise/distractor tokens interleaved. Target at every position is
the running product of the *group* tokens only (distractors must be ignored). Multi-layer stacked
recurrence h_l,t = A_l(.) h_l,t-1. Compares scalar/diagonal/abelian/complex/householder/quat/dense
transitions and depth. Length generalization to 4-8x training length.

Usage: python3 src/models/track_hard.py --group A5 --variant quat --layers 2 --noise 4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.state_tracking import Transition, group_table  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEV = "cuda"


class StackRec(nn.Module):
    def __init__(self, n_group, n_noise, variant, n_layers, d_state, d_emb=64, hidden=128):
        super().__init__()
        self.n_group, self.n_noise = n_group, n_noise
        self.d_state, self.L = d_state, n_layers
        self.emb = nn.Embedding(n_group + n_noise, d_emb)
        nn.init.normal_(self.emb.weight, std=1.0)
        self.trans = nn.ModuleList()
        for l in range(n_layers):
            self.trans.append(Transition(variant, d_emb if l == 0 else d_state, d_state))
        self.h0 = nn.Parameter(torch.randn(n_layers, d_state) * 0.1)
        self.norm = nn.LayerNorm(d_state)
        self.head = nn.Sequential(nn.Linear(d_state, hidden), nn.GELU(),
                                  nn.Linear(hidden, n_group))

    def forward(self, x):
        return forward_stack(self, x)


def forward_stack(model, x):
    """Sequential multi-layer recurrence."""
    B, T = x.shape
    e = model.emb(x)
    hs = [model.h0[l].unsqueeze(0).expand(B, -1).contiguous() for l in range(model.L)]
    outs = []
    for t in range(T):
        inp = e[:, t]
        for l in range(model.L):
            A = model.trans[l]((inp if l == 0 else hs[l - 1]).unsqueeze(1))[:, 0]
            hs[l] = torch.einsum("bij,bj->bi", A, hs[l])
            inp = hs[l]
        outs.append(model.head(model.norm(hs[-1])))
    return torch.stack(outs, 1)


def make_batch(mult, ident, B, T, n_noise, p_noise):
    n = mult.shape[0]
    x = torch.zeros(B, T, dtype=torch.long)
    y = torch.empty(B, T, dtype=torch.long)
    cur = torch.full((B,), ident, dtype=torch.long)
    for t in range(T):
        is_noise = torch.rand(B) < p_noise
        g = torch.randint(0, n, (B,))
        x[:, t] = torch.where(is_noise, n + torch.randint(0, max(1, n_noise), (B,)), g)
        cur = mult[cur, g]                                # only group tokens update the product
        y[:, t] = cur
    return x, y


def train_eval(group, variant, seed, n_layers=2, d_state=8, noise=4, p_noise=0.4,
               steps=6000, lr=3e-3, train_len=32, batch=128, gate_bias=0.0):
    torch.manual_seed(seed)
    perms, mult, ident = group_table(group)
    n = mult.shape[0]
    model = StackRec(n, noise, variant, n_layers, d_state).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    loss = torch.tensor(float("nan"))
    for step in range(steps):
        L = torch.randint(1, train_len + 1, (1,)).item()
        x, y = make_batch(mult, ident, batch, L, noise, p_noise)
        x, y = x.to(DEV), y.to(DEV)
        logits = forward_stack(model, x)
        loss = F.cross_entropy(logits.reshape(-1, n), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    accs = {}
    with torch.no_grad():
        for L in [16, 32, 64, 128, 256]:
            x, y = make_batch(mult, ident, 256, L, noise, p_noise)
            x, y = x.to(DEV), y.to(DEV)
            pred = forward_stack(model, x).argmax(-1)
            accs[str(L)] = (pred[:, -1] == y[:, -1]).float().mean().item()
    return {"group": group, "variant": variant, "layers": n_layers, "d_state": d_state,
            "noise": noise, "p_noise": p_noise, "seed": seed,
            "params": sum(p.numel() for p in model.parameters()),
            "acc": accs, "loss": float(loss), "wall_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="A5")
    ap.add_argument("--variants", nargs="+", default=["scalar", "abelian", "householder", "quat", "orth"])
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d_state", type=int, default=8)
    ap.add_argument("--noise", type=int, default=4)
    ap.add_argument("--p_noise", type=float, default=0.4)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args()
    for v in args.variants:
        for s in range(args.seeds):
            r = train_eval(args.group, v, s, args.layers, args.d_state, args.noise,
                           args.p_noise, args.steps)
            a = r["acc"]
            print(f"[{args.group}] {v:12s} L{r['layers']} s{s} p={r['params']:6d} "
                  f"L32={a['32']:.3f} L64={a['64']:.3f} L128={a['128']:.3f} "
                  f"L256={a['256']:.3f} ({r['wall_s']:.0f}s)")
            sys.stdout.flush()
            with open(ROOT / "results/track_hard.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
