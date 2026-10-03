"""State-tracking with the Track B mixer architecture (plan A.4 / O2).

Trains the actual MixerLM (selective-rotation linear attention) on the group prefix-product
task, so the sequence mixer is forced to use its transition structure. Compares mixer variants
and depths; measures length generalization.

Usage: python3 src/models/track_mixer.py --group A5 --mixer quat --layers 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.state_tracking import group_table, make_batch  # noqa: E402
from models.train_mixer import MixerLM                    # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="A5")
    ap.add_argument("--mixer", default="quat")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--train_len", type=int, default=32)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate_bias", type=float, default=4.0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    perms, mult, ident = group_table(args.group)
    n = mult.shape[0]
    model = MixerLM(n, [args.mixer] * args.layers, args.d, args.heads, args.seq,
                    hidden0=512, gate_bias=args.gate_bias).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    t0 = time.time()
    loss = torch.tensor(float("nan"))
    for step in range(args.steps):
        L = torch.randint(1, args.train_len + 1, (1,)).item()
        x, y = make_batch(mult, ident, args.batch, L)
        x, y = x.to(DEV), y.to(DEV)
        loss = F.cross_entropy(model(x).reshape(-1, n), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    accs = {}
    with torch.no_grad():
        for L in [8, 16, 32, 64, 128, 256]:
            if L > args.seq:
                continue
            x, y = make_batch(mult, ident, 256, L)
            x, y = x.to(DEV), y.to(DEV)
            pred = model(x).argmax(-1)
            accs[str(L)] = (pred[:, -1] == y[:, -1]).float().mean().item()
    print(f"[{args.group}] {args.mixer} L{args.layers} d{args.d} seed{args.seed} "
          f"p={sum(p.numel() for p in model.parameters())/1e3:.0f}k "
          f"L32={accs['32']:.3f} L64={accs.get('64', -1):.3f} "
          f"L128={accs.get('128', -1):.3f} L256={accs.get('256', -1):.3f} "
          f"loss={loss.item():.3f} ({time.time()-t0:.0f}s)")
    with open(ROOT / "results/track_mixer_statetracking.jsonl", "a") as f:
        f.write(json.dumps({"group": args.group, "mixer": args.mixer, "layers": args.layers,
                            "d": args.d, "seed": args.seed, "acc": accs,
                            "loss": float(loss)}) + "\n")


if __name__ == "__main__":
    main()
