"""Learn a rotation R per input-dimension to minimize W4A4 quantization error (plan 1.3).

Compares against fixed Hadamard. Methods: 'learn' (dense orthogonal, Cayley) and 'qb'
(quaternion-butterfly, plan 1.3). Objective: sum over layers of the STE
activation+weight-quantization output MSE on calibration activations.

Usage: python3 src/eval/learn_rotation.py --method qb --steps 300
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.rotation_quant import qb_build, qb_init, hadamard, perplexity, _make_forward  # noqa: E402

DEV = "cuda"
ROOT = Path(__file__).resolve().parents[2]


def q4_ste(x, group=128):
    """Group RTN (last dim) with straight-through gradient."""
    m, n = x.shape
    pad = (group - n % group) % group
    xg = F.pad(x, (0, pad)).reshape(m, -1, group)
    scale = xg.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7.0
    q = (xg / scale).round().clamp(-8, 7) * scale
    q = q.reshape(m, -1)[:, :n]
    return x + (q - x).detach()


def q4_act_ste(x):
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7.0
    q = (x / scale).round().clamp(-8, 7) * scale
    return x + (q - x).detach()


class ParamR(nn.Module):
    def __init__(self, d, kind):
        super().__init__()
        self.d, self.kind = d, kind
        if kind == "learn":
            self.A = nn.Parameter(torch.zeros(d, d))
        elif kind == "qb":
            n = d // 4
            if n & (n - 1) == 0:
                self.base = None
                self.p = nn.Parameter(qb_init(d))
            else:
                self.base = 1024
                assert d % 1024 == 0
                self.p = nn.Parameter(torch.stack([qb_init(1024)] * (d // 1024)))

    def R(self):
        if self.kind == "learn":
            S = self.A - self.A.t()
            I = torch.eye(self.d, device=self.A.device, dtype=self.A.dtype)
            return torch.linalg.solve(I - S, I + S)
        if self.base is None:
            return qb_build(self.p, self.d)
        blocks = [qb_build(self.p[i], self.base) for i in range(self.p.shape[0])]
        return torch.block_diag(*blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--method", default="qb")
    ap.add_argument("--calib_tokens", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--eval_seqs", type=int, default=40)
    args = ap.parse_args()
    import os
    os.environ.setdefault("HF_HOME", str(ROOT / "checkpoints/hf"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(DEV).eval()
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    etext = "\n".join(t for t in test["text"] if t.strip())
    eids = tok(etext, return_tensors="pt").input_ids[0][: args.eval_seqs * 1024]
    cds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ctext = "\n".join(t for t in cds["text"] if t.strip())
    cids = tok(ctext, return_tensors="pt").input_ids[0][: args.calib_tokens]

    # collect calibration inputs
    mods = []
    cap = {}
    handles = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and name.startswith("model.layers"):
            mods.append((name, m))

            def h(mod, inp, out, name=name):
                cap[name] = inp[0].detach().reshape(-1, inp[0].shape[-1]).to(DEV)
            handles.append(m.register_forward_hook(h))
    with torch.no_grad():
        for s in range(0, len(cids), 1024):
            model(cids[s:s + 1024].unsqueeze(0).to(DEV))
    for h in handles:
        h.remove()

    data = {name: (m.weight.detach().float().to(DEV), cap[name]) for name, m in mods}
    dims = sorted({m.in_features for _, m in mods})
    print("input dims:", dims, "n linears:", len(mods))

    rots = {}
    params = []
    for d in dims:
        pr = ParamR(d, args.method).to(DEV)
        rots[d] = pr
        params += list(pr.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    t0 = time.time()
    for step in range(args.steps):
        R = {d: pr.R() for d, pr in rots.items()}
        loss = 0.0
        for name, (W, X) in data.items():
            d = W.shape[1]
            Rp = R[d]
            xr = X @ Rp.t()
            Wr = W @ Rp.t()
            qx = q4_act_ste(xr)
            qw = q4_ste(Wr)
            ref = X @ W.t()
            hat = qx @ qw.t()
            loss = loss + (hat - ref).pow(2).sum() / ref.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0 or step == args.steps - 1:
            print(f"step {step:4d} loss={loss.item():.5f} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()

    # orthogonality check
    with torch.no_grad():
        for d, pr in rots.items():
            Rp = pr.R()
            err = (Rp.t() @ Rp - torch.eye(d, device=DEV)).abs().max().item()
            print(f"dim {d}: orthogonality max err {err:.2e}")

    Rfinal = {d: pr.R().detach() for d, pr in rots.items()}
    # evaluate ppl
    rots_by_dim = Rfinal

    def install():
        rmap = {d: r.cpu() for d, r in rots_by_dim.items()}
        for name, m in mods:
            m.forward = _make_forward(m, rmap[m.in_features], 4, 4)
    install()
    ppl = perplexity(model, eids)
    print(f"{args.model} w4a4_{args.method}: ppl={ppl:.3f}")
    torch.save({d: r.cpu() for d, r in Rfinal.items()}, ROOT / "checkpoints" / f"rot_{args.method}.pt")
    with open(ROOT / "results/rotation_quant.jsonl", "a") as f:
        f.write(json.dumps({"model": args.model, "method": f"w4a4_{args.method}",
                            "ppl": ppl, "steps": args.steps}) + "\n")


if __name__ == "__main__":
    main()
