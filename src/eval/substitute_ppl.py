"""End-to-end WikiText-2 perplexity after substituting one linear type at a time.

Supports the G1 ppl criterion in plan 1.2. Methods: orig, Q_act (activation-aware quaternion,
A.3), LR (activation-aware low-rank), INT4 (RTN g128). All at equal stored bytes for Q/LR/INT4.

Usage: python3 src/eval/substitute_ppl.py --model Qwen/Qwen3-0.6B --proj o_proj
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quat.ops import build_block_quat, activation_aware_quat  # noqa: E402
from eval.proj_diag import low_rank_act, int4_rtn            # noqa: E402

DEV = "cuda"
ROOT = Path(__file__).resolve().parents[2]


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
    return torch.exp(torch.tensor(nll / ntok)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--proj", default="o_proj")
    ap.add_argument("--calib_tokens", type=int, default=16384)
    ap.add_argument("--eval_seqs", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import os
    os.environ.setdefault("HF_HOME", str(ROOT / "checkpoints/hf"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(DEV).eval()
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(t for t in test["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][: args.eval_seqs * 1024]
    calib_ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ctext = "\n".join(t for t in calib_ds["text"] if t.strip())
    cids = tok(ctext, return_tensors="pt").input_ids[0][: args.calib_tokens]

    layers = model.model.layers
    mods = [getattr(l.self_attn, args.proj, None) or getattr(l.mlp, args.proj, None) for l in layers]
    mods = [m for m in mods if m is not None]
    n = len(mods)

    # capture inputs
    cap = {}
    handles = []
    for i, m in enumerate(mods):
        def h(mod, inp, out, i=i):
            cap[i] = inp[0].detach().reshape(-1, inp[0].shape[-1]).float().cpu()
        handles.append(m.register_forward_hook(h))
    with torch.no_grad():
        for s in range(0, len(cids), 1024):
            model(cids[s:s + 1024].unsqueeze(0).to(DEV))
    for h in handles:
        h.remove()

    orig = [m.weight.detach().clone() for m in mods]
    results = {"orig": perplexity(model, ids)}

    for method in ["Q_act", "LR", "INT4"]:
        for i, m in enumerate(mods):
            W = orig[i].float().cpu()
            X = cap[i].t().contiguous()
            if method == "Q_act":
                Wh = build_block_quat(activation_aware_quat(X, W @ X))
            elif method == "LR":
                Wh = low_rank_act(W, X)
            elif method == "INT4":
                Wh = int4_rtn(W)
            m.weight.data.copy_(Wh.to(DEV).to(torch.bfloat16))
        ppl = perplexity(model, ids)
        results[method] = ppl
        for i, m in enumerate(mods):
            m.weight.data.copy_(orig[i])
        print(f"{args.proj} {method}: ppl={ppl:.3f} (delta={ppl-results['orig']:+.3f})")
        sys.stdout.flush()
    results["proj"] = args.proj
    results["model"] = args.model
    out = args.out or f"results/substitute_ppl_{args.proj}.json"
    (ROOT / out).write_text(json.dumps(results, indent=2))
    print(f"orig baseline ppl = {results['orig']:.3f}  -> {out}")


if __name__ == "__main__":
    main()
