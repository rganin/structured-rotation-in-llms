"""Track B screening: sequence-mixer variants (plan 2.5), O2 / assumption #1.

All mixers are implemented in exact parallel (quadratic) form so semantics are controlled:
  - attn    : softmax causal attention (baseline)
  - scalar  : S_t = exp(g_t) S_{t-1} + k_t v_t^T, o_t = q_t^T S_t           (GLA-like, abelian)
  - complex : same with a data-dependent abelian 2x2 rotation of the state   (B4)
  - quat    : same with a data-dependent non-abelian unit-quaternion rotation (B3)
  - fixed   : same with a fixed (non-data-dependent) block rotation (RoPE-like)
A data-dependent rotation r_t is applied by rotating k/q with the cumulative frame P_t
(plan A9): k~_s = conj(P_s) k_s, q~_t = conj(P_t) q_t. Verified against the sequential recurrence.

Usage: python3 src/models/train_mixer.py --mixer quat --tokens 30e6 --tag B3
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
from quat.ops import hamilton, conj  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEV = "cuda"


# ----------------------------- primitives -----------------------------------
def cumulative_quat(r):
    """Inclusive scan of the Hamilton product along dim -3. r: (...,T,nb,4)."""
    T = r.shape[-3]
    P = r
    d = 1
    while d < T:
        ident = torch.zeros_like(P[..., :d, :, :])
        ident[..., 0] = 1.0
        shifted = torch.cat([ident, P[..., :-d, :, :]], dim=-3)
        P = hamilton(P, shifted)
        d *= 2
    return P


def rotate_qk_quat(q, k, r):
    """q,k: (B,H,T,D). r: (B,H,T,D). Returns rotated q,k."""
    B, H, T, D = q.shape
    nb = D // 4
    rb = F.normalize(r.reshape(B, H, T, nb, 4), dim=-1)
    P = cumulative_quat(rb)                      # (B,H,T,nb,4)
    Pi = conj(P)                                 # inverse frame
    qr = hamilton(Pi, q.reshape(B, H, T, nb, 4)).reshape(B, H, T, D)
    kr = hamilton(Pi, k.reshape(B, H, T, nb, 4)).reshape(B, H, T, D)
    return qr, kr


def rotate_qk_complex(q, k, th):
    """th: (B,H,T,D/2) angles. Abelian cumulative sum of angles."""
    B, H, T, half = th.shape
    Theta = torch.cumsum(th, dim=2)              # (B,H,T,half)
    def rot(x, sign):
        x0, x1 = x[..., :half], x[..., half:]
        a = sign * Theta
        c, s = a.cos(), a.sin()
        return torch.cat([x0 * c - x1 * s, x0 * s + x1 * c], -1)
    return rot(q, -1), rot(k, -1)


def la_parallel(q, k, v, g):
    """Exact scalar-decay causal linear attention.

    q,k: (B,H,T,D); v: (B,H,T,Dv); g: (B,H,T) log decay (<=0).
    o_t = sum_{s<=t} exp(G_t-G_s) <q_t,k_s> v_s.
    """
    G = torch.cumsum(g, dim=-1)                  # (B,H,T)
    scores = torch.einsum("bhid,bhjd->bhij", q, k)
    rel = (G.unsqueeze(-1) - G.unsqueeze(-2)).clamp(max=0.0)   # avoid exp overflow/NaN grads
    decay = torch.exp(rel)
    T = q.shape[2]
    mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
    att = scores * decay.masked_fill(~mask, 0.0)
    return torch.einsum("bhij,bhjd->bhid", att, v)


class Mixer(nn.Module):
    def __init__(self, d_model, n_heads, variant, head_dim=None, gate_bias=0.0):
        super().__init__()
        self.variant = variant
        self.h = n_heads
        self.d_k = head_dim or d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * n_heads * self.d_k, bias=False)
        self.o = nn.Linear(n_heads * self.d_k, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, n_heads)
        nn.init.zeros_(self.g_proj.weight)
        nn.init.constant_(self.g_proj.bias, gate_bias)
        if variant in ("complex", "quat"):
            out_dim = n_heads * (self.d_k // 2) if variant == "complex" else n_heads * self.d_k
            self.r_proj = nn.Linear(d_model, out_dim, bias=False)
            nn.init.normal_(self.r_proj.weight, std=0.02)
            if variant == "quat":
                with torch.no_grad():
                    b = torch.zeros(n_heads * self.d_k)
                    b.reshape(n_heads, self.d_k)[:, :] = torch.tensor([1.0, 0, 0, 0]).repeat(
                        self.d_k // 4)
                    self.r_proj.bias = nn.Parameter(b)
        elif variant == "fixed":
            self.r_fixed = nn.Parameter(torch.randn(n_heads, self.d_k // 4, 4))
        self.variant_params = sum(p.numel() for p in self.parameters())

    def forward(self, x, mask):
        B, T, _ = x.shape
        H, D = self.h, self.d_k
        qkv = self.qkv(x).reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # (B,H,T,D)
        g = F.logsigmoid(self.g_proj(x)).transpose(1, 2)       # (B,H,T)
        if self.variant == "scalar":
            pass
        elif self.variant == "complex":
            th = self.r_proj(x).reshape(B, T, H, D // 2).permute(0, 2, 1, 3)
            q, k = rotate_qk_complex(q, k, th)
        elif self.variant == "quat":
            r = self.r_proj(x).reshape(B, T, H, D).permute(0, 2, 1, 3)
            q, k = rotate_qk_quat(q, k, r)
        elif self.variant == "fixed":
            r = self.r_fixed.unsqueeze(0).unsqueeze(2).expand(B, H, T, D // 4, 4)
            q, k = rotate_qk_quat(q, k, r.reshape(B, H, T, D))
        o = la_parallel(q, k, v, g)                            # (B,H,T,D)
        o = o.transpose(1, 2).reshape(B, T, H * D)
        return self.o(o)


class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        q, k = self._rope(q), self._rope(k)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_k)
        att = att.masked_fill(mask[:, :, :T, :T], float("-inf")).softmax(-1)
        return self.o((att @ v).transpose(1, 2).reshape(B, T, self.h * self.d_k))

    def _rope(self, x, base=10000.0):
        B, H, T, D = x.shape
        half = D // 2
        f = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
        ang = torch.outer(torch.arange(T, device=x.device, dtype=torch.float32), f)
        c, s = ang.cos()[None, None], ang.sin()[None, None]
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.w


class Block(nn.Module):
    def __init__(self, d_model, n_heads, hidden, mixer, gate_bias=0.0):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d_model), RMSNorm(d_model)
        self.mix = Attention(d_model, n_heads) if mixer == "attn" else Mixer(d_model, n_heads, mixer, gate_bias=gate_bias)
        self.ffn = nn.Sequential()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x, mask):
        x = x + self.mix(self.n1(x), mask)
        h = self.n2(x)
        x = x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        return x


class MixerLM(nn.Module):
    def __init__(self, vocab, pattern, d_model=512, n_heads=8, seq=512, hidden0=1408, gate_bias=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        # pad last layer to attention if divisible, else mirror
        base = 4 * d_model * d_model + 3 * d_model * hidden0       # baseline per-layer non-emb
        d_k = d_model // n_heads
        mixer_cost = {"attn": 4 * d_model * d_model,
                      "scalar": 4 * d_model * d_model + d_model * n_heads,
                      "complex": 4 * d_model * d_model + d_model * n_heads + d_model * (d_model // 2),
                      "quat": 4 * d_model * d_model + d_model * n_heads + d_model * d_model,
                      "fixed": 4 * d_model * d_model + d_model * n_heads + d_model}
        self.blocks = nn.ModuleList()
        for m in pattern:
            ffn = max(3 * d_model * 64, base - mixer_cost[m])
            hidden = max(64, int(round(ffn / (3 * d_model) / 4)) * 4)
            self.blocks.append(Block(d_model, n_heads, hidden, m, gate_bias))
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.emb.weight
        self.register_buffer("mask", torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1))
        nn.init.normal_(self.emb.weight, std=0.02)
        self.pattern = pattern
        self.hidden = hidden

    def forward(self, idx):
        x = self.emb(idx)
        m = self.mask.unsqueeze(0).unsqueeze(0)
        for b in self.blocks:
            x = b(x, m)
        return self.head(self.norm(x))

    def non_emb_params(self):
        return sum(p.numel() for n, p in self.named_parameters() if "emb" not in n)


def lr_at(step, total, peak, warmup=100):
    if step < warmup:
        return peak * step / warmup
    p = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def load_tokens():
    cache = ROOT / "data" / "tokens_gpt2.pt"
    d = torch.load(cache)
    return d["train"], d["val"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixer", default="quat")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--tokens", type=float, default=30e6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--force_hidden", type=int, default=0)
    ap.add_argument("--gate_bias", type=float, default=0.0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    train, val = load_tokens()
    # parse pattern: "quat" or "quat*3+attn*1" (repeated/truncated to n_layers)
    pat = []
    for part in args.mixer.split("+"):
        if "*" in part:
            m, n = part.split("*")
            pat += [m] * int(n)
        else:
            pat.append(part)
    pattern = (pat * ((args.layers + len(pat) - 1) // len(pat)))[:args.layers]
    model = MixerLM(50257, pattern, args.d, args.heads, args.seq, gate_bias=args.gate_bias).to(DEV)
    if args.force_hidden:
        for b in model.blocks:
            d = args.d
            b.w_gate = nn.Linear(d, args.force_hidden, bias=False).to(DEV)
            b.w_up = nn.Linear(d, args.force_hidden, bias=False).to(DEV)
            b.w_down = nn.Linear(args.force_hidden, d, bias=False).to(DEV)
        model.hidden = args.force_hidden
    print(f"[{args.tag}] pattern={pattern} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"non_emb={model.non_emb_params()/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    tps = args.batch * args.seq
    total = int(args.tokens / tps)
    print(f"[{args.tag}] steps={total} tokens={total*tps/1e6:.0f}M")
    t0 = time.time()
    for step in range(total):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, total, args.lr)
        ix = torch.randint(len(train) - args.seq - 1, (args.batch,))
        x = torch.stack([train[i:i + args.seq] for i in ix]).long().to(DEV)
        y = torch.stack([train[i + 1:i + args.seq + 1] for i in ix]).long().to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).reshape(-1, 50257).float(), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, total // 8) == 0 or step == total - 1:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vl = 0.0
                for i in range(0, 10 * args.seq, args.seq):
                    vx = val[i:i + args.seq].long().unsqueeze(0).to(DEV)
                    vy = val[i + 1:i + args.seq + 1].long().unsqueeze(0).to(DEV)
                    vl += F.cross_entropy(model(vx)[0].float(), vy[0]).item()
                vl /= 10
            print(f"[{args.tag}] step {step:5d} train={loss.item():.4f} val={vl:.4f} "
                  f"({time.time()-t0:.0f}s)")
            sys.stdout.flush()
    rec = {"tag": args.tag, "mixer": args.mixer, "hidden": model.hidden, "gate_bias": args.gate_bias,
           "non_emb": model.non_emb_params(), "tokens": total * tps, "val": vl,
           "seed": args.seed, "wall_s": time.time() - t0}
    with open(ROOT / "results/trackB.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("FINAL", json.dumps(rec))


if __name__ == "__main__":
    main()
