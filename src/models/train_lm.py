"""Small from-scratch LM screening for Track A (plan 2.4 / 2.0).

Llama-style decoder with configurable FFN linear type. Compares dense vs block-quaternion vs
low-rank vs PHM at iso-params. Training data: tiktoken GPT-2 tokens from WikiText-103 (or -2).

Usage:
  python3 src/models/train_lm.py --ffn dense --hidden 1408 --tokens 100e6 --tag A0
  python3 src/models/train_lm.py --ffn quat  --hidden 1408 --tokens 100e6 --tag A1
  python3 src/models/train_lm.py --ffn dense --hidden 352  --tokens 100e6 --tag A2
  python3 src/models/train_lm.py --ffn quat  --hidden 5632 --tokens 100e6 --tag A6
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
from quat.ops import build_block_quat  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEV = "cuda"


# ----------------------------- layers ---------------------------------------
class QuatLinear(nn.Module):
    """Block-quaternion linear: stores mn/4 params, builds dense L(q) blocks each forward."""

    def __init__(self, in_features, out_features):
        super().__init__()
        assert in_features % 4 == 0 and out_features % 4 == 0
        self.in_features, self.out_features = in_features, out_features
        k = 1.0 / math.sqrt(in_features)
        self.q = nn.Parameter(torch.empty(out_features // 4, in_features // 4, 4).uniform_(-k, k))

    def forward(self, x):
        W = build_block_quat(self.q).to(x.dtype)
        return F.linear(x, W)


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        r = max(1, round(in_features * out_features / (4 * (in_features + out_features))))
        self.r = r
        self.A = nn.Parameter(torch.empty(out_features, r))
        self.B = nn.Parameter(torch.empty(r, in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(F.linear(x, self.B), self.A)


class PHMLinear(nn.Module):
    def __init__(self, in_features, out_features, n=4):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.n = n
        assert in_features % n == 0 and out_features % n == 0
        self.S = nn.Parameter(torch.empty(n, out_features // n, in_features // n))
        self.A = nn.Parameter(torch.randn(n, n, n) * 0.02)
        k = 1.0 / math.sqrt(in_features)
        self.S.data.uniform_(-k, k)

    def forward(self, x):
        W = torch.einsum("iab,ipq->apbq", self.A, self.S).reshape(
            self.out_features, self.in_features)
        return F.linear(x, W)


def make_linear(kind, in_features, out_features):
    if kind == "dense":
        return nn.Linear(in_features, out_features, bias=False)
    if kind == "quat":
        return QuatLinear(in_features, out_features)
    if kind == "lowrank":
        return LowRankLinear(in_features, out_features)
    if kind == "phm":
        return PHMLinear(in_features, out_features)
    raise ValueError(kind)


class FFN(nn.Module):
    def __init__(self, d, hidden, kind):
        super().__init__()
        self.gate = make_linear(kind, d, hidden)
        self.up = make_linear(kind, d, hidden)
        self.down = make_linear(kind, hidden, d)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.w


def rope(x, base=10000.0):
    B, H, T, D = x.shape
    half = D // 2
    freqs = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    pos = torch.arange(T, device=x.device, dtype=torch.float32)
    ang = torch.outer(pos, freqs)                       # (T, half)
    cos = ang.cos()[None, None]
    sin = ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Block(nn.Module):
    def __init__(self, d, n_heads, hidden, kind, attn_kind="dense"):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = make_linear(attn_kind, d, 3 * d)
        self.o = make_linear(attn_kind, d, d)
        self.ffn = FFN(d, hidden, kind)

    def forward(self, x, mask):
        B, T, D = x.shape
        h = self.n1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = rope(q), rope(k)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        att = att.masked_fill(mask[:, :, :T, :T], float("-inf")).softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.o(out)
        x = x + self.ffn(self.n2(x))
        return x


class SmallLM(nn.Module):
    def __init__(self, vocab, d=512, n_layers=6, n_heads=8, hidden=1408, kind="dense", seq=512, attn_kind="dense"):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, hidden, kind, attn_kind) for _ in range(n_layers)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight             # tied
        self.seq = seq
        self.register_buffer("mask", torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1))
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, idx):
        x = self.emb(idx)
        m = self.mask.unsqueeze(0).unsqueeze(0)
        for b in self.blocks:
            x = b(x, m)
        return self.head(self.norm(x))

    def non_emb_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if "emb" not in n)


# ----------------------------- data -----------------------------------------
def load_tokens(vocab_size):
    cache = ROOT / "data" / "tokens_gpt2.pt"
    if cache.exists():
        d = torch.load(cache)
        return d["train"], d["val"]
    import tiktoken
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    text = "\n".join(t for t in ds["text"] if t.strip())
    train = torch.tensor(enc.encode_ordinary(text), dtype=torch.uint16)
    vs = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    vtext = "\n".join(t for t in vs["text"] if t.strip())
    val = torch.tensor(enc.encode_ordinary(vtext), dtype=torch.uint16)
    torch.save({"train": train, "val": val}, cache)
    return train, val


# ----------------------------- train ----------------------------------------
def lr_at(step, total, peak, warmup=100):
    if step < warmup:
        return peak * step / warmup
    p = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffn", default="dense")
    ap.add_argument("--attn_kind", default="dense")
    ap.add_argument("--hidden", type=int, default=1408)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tokens", type=float, default=100e6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    train, val = load_tokens(50304)
    model = SmallLM(50257, args.d, args.layers, args.heads, args.hidden, args.ffn, args.seq, args.attn_kind).to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] ffn={args.ffn} attn={args.attn_kind} hidden={args.hidden} params={nparam/1e6:.2f}M "
          f"non_emb={model.non_emb_params()/1e6:.2f}M")
    if args.compile:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    tokens_per_step = args.batch * args.seq
    total = int(args.tokens / tokens_per_step)
    print(f"[{args.tag}] steps={total} tokens={total*tokens_per_step/1e6:.0f}M")
    best_val = None
    t0 = time.time()
    for step in range(total):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total, args.lr)
        ix = torch.randint(len(train) - args.seq - 1, (args.batch,))
        x = torch.stack([train[i:i + args.seq] for i in ix]).long().to(DEV)
        y = torch.stack([train[i + 1:i + args.seq + 1] for i in ix]).long().to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, total // 10) == 0 or step == total - 1:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vl = 0.0
                for i in range(0, 10 * args.seq, args.seq):
                    vx = val[i:i + args.seq].long().unsqueeze(0).to(DEV)
                    vy = val[i + 1:i + args.seq + 1].long().unsqueeze(0).to(DEV)
                    vlogits = model(vx)[0].float()
                    vl += F.cross_entropy(vlogits, vy[0]).item()
                vl /= 10
            best_val = vl
            print(f"[{args.tag}] step {step:5d} train={loss.item():.4f} val={vl:.4f} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()
    rec = {"tag": args.tag, "ffn": args.ffn, "attn_kind": args.attn_kind, "hidden": args.hidden, "params": nparam,
           "non_emb": model.non_emb_params(), "tokens": total * tokens_per_step,
           "val": best_val, "seed": args.seed, "lr": args.lr, "wall_s": time.time() - t0}
    with open(ROOT / "results/trackA.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("FINAL", json.dumps(rec))


if __name__ == "__main__":
    main()
