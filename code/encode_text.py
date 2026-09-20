"""Frozen LLM semantic item encoder (paper Sec. IV-E).

Precomputes and caches item-level semantic embeddings with a frozen Qwen3
embedding model.  Default: Qwen/Qwen3-Embedding-0.6B -- the compact variant
explicitly evaluated in the paper.  The 4B variant does not fit a CPU-only
budget on this host (see the report for the measured cost/performance trade-off).
Pooling follows the Qwen3-Embedding convention: last non-pad token + L2 norm.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CACHE, DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--tag", default="qwen3_0.6b")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    from transformers import AutoModel, AutoTokenizer

    with open(os.path.join(DATA, f"{a.dataset}.pkl"), "rb") as f:
        ds = pickle.load(f)
    items = ds["items"]
    texts = [ds["profiles"].get(it, "") or "unknown item" for it in items]
    if a.bench:
        texts = texts[: a.bench]
    print(f"[{a.dataset}] encoding {len(texts):,} item texts with {a.model}, "
          f"max_tokens={a.max_tokens}", flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModel.from_pretrained(a.model, dtype=torch.float32).eval()
    nparam = sum(p.numel() for p in model.parameters())
    print(f"  model loaded in {time.time()-t0:.1f}s  params={nparam/1e6:.1f}M", flush=True)

    fn = os.path.join(CACHE, f"{a.dataset}_{a.tag}.npy")
    ck = fn + ".partial.npz"
    dim = model.config.hidden_size
    out = np.zeros((len(texts), dim), dtype=np.float32)
    done_mask = np.zeros(len(texts), dtype=bool)
    if os.path.exists(ck) and not a.bench:
        try:
            z = np.load(ck)
            if z["out"].shape == out.shape:
                out = z["out"]
                done_mask = z["done"]
                print(f"  resuming: {int(done_mask.sum()):,} items already encoded", flush=True)
        except Exception as e:
            print("  [warn] could not resume:", e)

    todo = [i for i in range(len(texts)) if not done_mask[i]] if not a.bench else list(range(len(texts)))
    if not todo:
        print("  nothing to do")
        np.save(fn, out)
        return

    # length-sorted batching: padding waste is the dominant cost on CPU
    toks = [len(tok(texts[i], truncation=True, max_length=a.max_tokens)["input_ids"])
            for i in todo]
    order = np.array([todo[j] for j in np.argsort(toks)])
    print(f"  token lengths: min={min(toks)} mean={np.mean(toks):.1f} max={max(toks)} "
          f"(max_tokens={a.max_tokens}); to encode={len(todo):,}", flush=True)

    t0, n_done = time.time(), 0
    with torch.inference_mode():
        for s in range(0, len(order), a.batch):
            sel = order[s: s + a.batch]
            enc = tok([texts[i] for i in sel], padding=True, truncation=True,
                      max_length=a.max_tokens, return_tensors="pt")
            h = model(**enc).last_hidden_state
            emb = torch.nn.functional.normalize(h[:, -1], p=2, dim=1)
            out[sel] = emb.float().numpy()
            done_mask[sel] = True
            n_done += len(sel)
            if (s // a.batch) % 20 == 0:
                el = time.time() - t0
                rate = n_done / max(el, 1e-9)
                print(f"   {n_done:,}/{len(order):,}  {rate:.1f} it/s  "
                      f"ETA {(len(order)-n_done)/max(rate,1e-9)/60:.1f} min", flush=True)
                if not a.bench:
                    np.savez(ck, out=out, done=done_mask)
    el = time.time() - t0
    print(f"  encoded {len(todo):,} in {el/60:.2f} min "
          f"({len(todo)/max(el,1e-9):.1f} it/s) dim={dim}", flush=True)

    if a.bench:
        print(f"  full catalog estimate for {a.dataset}: "
              f"{len(items)/max(len(texts)/el,1e-9)/60:.1f} min")
        print("BENCH ONLY - not saving")
        return

    fn = os.path.join(CACHE, f"{a.dataset}_{a.tag}.npy")
    np.save(fn, out)
    if os.path.exists(fn + ".partial.npz"):
        os.remove(fn + ".partial.npz")
    json.dump({"model": a.model, "dim": int(dim), "n": int(len(texts)),
               "max_tokens": a.max_tokens, "seconds": round(el, 1),
               "params_M": round(nparam / 1e6, 2), "throughput_items_per_s":
               round(len(texts) / max(el, 1e-9), 2)},
              open(fn.replace(".npy", ".json"), "w"), indent=2)
    print("saved", fn, out.shape)


if __name__ == "__main__":
    main()
