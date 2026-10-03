"""MQAR recall probe for mixer variants (plan 7.1 / Track B), O2.

Multi-query associative recall: sequence = [k1,v1,...,kN,vN, q1,...,qQ]; at each query the
model must output the value bound to that key. Accuracy measured only at query positions.
Trains a tiny model with the same mixer variants as Track B.

Usage: python3 src/eval/mqar.py --mixer quat --n_pairs 64
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.train_mixer import Mixer, RMSNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEV = "cuda"


class MQARModel(nn.Module):
    def __init__(self, n_keys=256, d=128, n_layers=2, n_heads=4, hidden=512, mixer="scalar", seq=512, gate_bias=0.0):
        super().__init__()
        self.n_keys = n_keys
        self.emb = nn.Embedding(2 * n_keys, d)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "n1": RMSNorm(d), "mix": Mixer(d, n_heads, mixer, gate_bias=gate_bias),
                "n2": RMSNorm(d),
                "ff": nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d)),
            }))
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, 2 * n_keys, bias=False)
        self.register_buffer("mask", torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1))

    def forward(self, idx):
        x = self.emb(idx)
        m = self.mask.unsqueeze(0).unsqueeze(0)
        for b in self.blocks:
            x = x + b["mix"](b["n1"](x), m)
            x = x + b["ff"](b["n2"](x))
        return self.head(self.norm(x))


def make_batch(bs, n_pairs, n_queries, n_keys):
    T = 2 * n_pairs + n_queries
    x = torch.zeros(bs, T, dtype=torch.long)
    y = torch.full((bs, T), -100, dtype=torch.long)
    qpos = torch.zeros(bs, n_queries, dtype=torch.long)
    for b in range(bs):
        keys = torch.randperm(n_keys)[:n_pairs]
        vals = torch.randint(0, n_keys, (n_pairs,))
        for i in range(n_pairs):
            x[b, 2 * i] = keys[i]
            x[b, 2 * i + 1] = n_keys + vals[i]
        qi = torch.randint(0, n_pairs, (n_queries,))
        for j in range(n_queries):
            pos = 2 * n_pairs + j
            x[b, pos] = keys[qi[j]]
            y[b, pos] = n_keys + vals[qi[j]]
            qpos[b, j] = pos
    return x, y, qpos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixer", default="quat")
    ap.add_argument("--n_keys", type=int, default=128)
    ap.add_argument("--n_pairs", type=int, default=64)
    ap.add_argument("--n_queries", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate_bias", type=float, default=0.0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    T = 2 * args.n_pairs + args.n_queries
    model = MQARModel(args.n_keys, mixer=args.mixer, seq=T, gate_bias=args.gate_bias).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    t0 = time.time()
    for step in range(args.steps):
        x, y, _ = make_batch(args.batch, args.n_pairs, args.n_queries, args.n_keys)
        x, y = x.to(DEV), y.to(DEV)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 2 * args.n_keys), y.reshape(-1),
                               ignore_index=-100)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    accs = {}
    with torch.no_grad():
        for nq in [args.n_queries, 32]:
            acc = 0.0
            for _ in range(8):
                x, y, _ = make_batch(64, args.n_pairs, nq, args.n_keys)
                x, y = x.to(DEV), y.to(DEV)
                pred = model(x).argmax(-1)
                m = y != -100
                acc += (pred[m] == y[m]).float().mean().item() / 8
            accs[str(nq)] = acc
    print(f"MQAR mixer={args.mixer} n_pairs={args.n_pairs} loss={loss.item():.3f} "
          f"acc_q16={accs[str(args.n_queries)]:.3f} acc_q32={accs['32']:.3f} ({time.time()-t0:.0f}s)")
    with open(ROOT / "results/mqar.jsonl", "a") as f:
        f.write(json.dumps({"mixer": args.mixer, "n_pairs": args.n_pairs, **accs,
                            "loss": loss.item(), "seed": args.seed}) + "\n")


if __name__ == "__main__":
    main()
