"""Weight-projection diagnostic (plan.md 1.2 / 5.3) — O1.

For selected linear layers of a pretrained model, fit several approximations at equal stored
bytes and report the activation-weighted relative error
    err = ||(W - W_hat) X||_F / ||W X||_F .
Methods: Q_frob (block-quaternion Frobenius), Q_act (activation-aware closed form, A.3),
LR (activation-aware low-rank), PHM4 (ALS), INT4 (RTN g128), Q+INT4 (quat params at 4 bits).

Usage: python3 src/eval/proj_diag.py --model Qwen/Qwen3-0.6B --layers 0 14 27 --tokens 16384
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quat.ops import frobenius_project_blocks, build_block_quat, activation_aware_quat  # noqa: E402

DEV = "cuda"
RESULTS = Path(__file__).resolve().parents[2] / "results"


def rel_err(W, X, What):
    num = ((W - What) @ X).norm()
    den = (W @ X).norm()
    return (num / den).item()


def quat_frob(W):
    return build_block_quat(frobenius_project_blocks(W))


def quat_act(W, X):
    Y = W @ X
    return build_block_quat(activation_aware_quat(X, Y))


def low_rank_act(W, X):
    """Activation-aware low rank (SVD-LLM): min ||(W-What)X||, What = (W C^1/2)_r C^-1/2."""
    m, n = W.shape
    C = (X.double() @ X.double().t()) / X.shape[1]
    evals, evecs = torch.linalg.eigh(C)
    evals = evals.clamp_min(0)
    Csqrt = (evecs * evals.sqrt()) @ evecs.t()
    Cnegsqrt = (evecs * (1.0 / evals.sqrt().clamp_min(1e-8))) @ evecs.t()
    A = W.double() @ Csqrt
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    r = max(1, round(m * n / (4 * (m + n))))
    r = min(r, U.shape[1])
    Ar = (U[:, :r] * S[:r]) @ Vh[:r]
    return (Ar @ Cnegsqrt).float()


def int4_rtn(W, group=128):
    m, n = W.shape
    g = group
    pad = (g - n % g) % g
    Wp = torch.nn.functional.pad(W, (0, pad))
    Wg = Wp.reshape(m, -1, g)
    scale = Wg.abs().amax(-1, keepdim=True) / 7.0
    scale = scale.clamp_min(1e-8)
    q = (Wg / scale).round().clamp(-8, 7)
    Wq = (q * scale).reshape(m, -1)[:, :n]
    return Wq


def int4_quat_params(qparams, group=128):
    m4, n4, _ = qparams.shape
    flat = qparams.reshape(m4, n4 * 4)
    n = flat.shape[1]
    pad = (group - n % group) % group
    fp = torch.nn.functional.pad(flat, (0, pad))
    g = fp.reshape(m4, -1, group)
    scale = g.abs().amax(-1, keepdim=True) / 7.0
    q = (g / scale.clamp_min(1e-8)).round().clamp(-8, 7)
    out = (q * scale).reshape(m4, -1)[:, :n].reshape(m4, n4, 4)
    return build_block_quat(out)


def phm4_als(W, X, iters=10):
    """Fit W ~= sum_i A_i (x) S_i with A_i 4x4, S_i (m4 x n4), contiguous 4x4 blocks.

    Block layout matches build_block_quat: W[I*4+c, J*4+k] = sum_i A_i[c,k] S_i[I,J].
    PHM-4 contains block-quaternion (A_i = L(e_i), S_i = quaternion components).
    """
    m, n = W.shape
    assert m % 4 == 0 and n % 4 == 0
    m4, n4 = m // 4, n // 4
    R = W.reshape(m4, 4, n4, 4)                              # (I,c,J,k)
    from quat.ops import Lmat, BASIS
    A = torch.stack([Lmat(BASIS[i].to(W.device).to(W.dtype)) for i in range(4)])  # (4,4,4)
    S = torch.zeros(4, m4, n4, device=W.device, dtype=W.dtype)

    def full():
        return torch.einsum("ick,iIJ->IcJk", A, S)

    for _ in range(iters):
        for i in range(4):
            other = R - full() + torch.einsum("ck,IJ->IcJk", A[i], S[i])
            S[i] = torch.einsum("IcJk,ck->IJ", other, A[i]) / (A[i].pow(2).sum() + 1e-12)
        for i in range(4):
            other = R - full() + torch.einsum("ck,IJ->IcJk", A[i], S[i])
            A[i] = torch.einsum("IcJk,IJ->ck", other, S[i]) / (S[i].pow(2).sum() + 1e-12)
    M = full()
    return M.reshape(m, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 7, 14, 21, 27])
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--out", default="results/proj_diag_qwen06.jsonl")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    import os
    os.environ.setdefault("HF_HOME", str(Path.cwd() / "checkpoints/hf"))

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(DEV).eval()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][: args.tokens]
    nseq = len(ids) // args.seq
    X_in = ids[: nseq * args.seq].reshape(nseq, args.seq)
    print(f"calibration tokens: {X_in.numel()}")

    cap = {}

    def make_hook(name):
        def hook(mod, inp, out):
            cap[name] = inp[0].detach().reshape(-1, inp[0].shape[-1]).float().cpu()
        return hook

    handles = []
    for li in args.layers:
        layer = model.model.layers[li]
        for pname in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            mod = getattr(layer.self_attn, pname, None) or getattr(layer.mlp, pname, None)
            if mod is not None:
                handles.append(mod.register_forward_hook(make_hook(f"{li}.{pname}")))

    with torch.no_grad():
        for s in range(nseq):
            model(X_in[s:s + 1].to(DEV))
    for h in handles:
        h.remove()

    rows = []
    for li in args.layers:
        for pname in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            key = f"{li}.{pname}"
            if key not in cap:
                continue
            mod = getattr(model.model.layers[li].self_attn, pname, None) or \
                getattr(model.model.layers[li].mlp, pname, None)
            W = mod.weight.detach().float().cpu()
            X = cap.pop(key).t().contiguous()          # (in, T)
            t0 = time.time()
            qf = frobenius_project_blocks(W)
            Wqf = build_block_quat(qf)
            rows_methods = {
                "Q_frob": Wqf,
                "Q_act": quat_act(W, X),
                "LR": low_rank_act(W, X),
                "PHM4": phm4_als(W, X),
                "INT4": int4_rtn(W),
                "Q_INT4": int4_quat_params(qf),
            }
            rec = {"layer": li, "proj": pname, "shape": list(W.shape),
                   "retained_energy_Qfrob": (Wqf.norm() / W.norm()).pow(2).item(),
                   "err": {k: rel_err(W, X, v) for k, v in rows_methods.items()},
                   "wall_s": time.time() - t0}
            rows.append(rec)
            print(f"L{li:02d}.{pname:10s} {str(W.shape):14s} "
                  + " ".join(f"{k}={v:.4f}" for k, v in rec["err"].items())
                  + f" retE={rec['retained_energy_Qfrob']:.3f} ({rec['wall_s']:.1f}s)")
            sys.stdout.flush()
    (Path.cwd() / args.out).parent.mkdir(exist_ok=True)
    with open(args.out, "a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
