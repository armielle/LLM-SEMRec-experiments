"""Frozen SigLIP 2 visual item encoder (paper Sec. IV-D).

Precomputes and caches item-level visual embeddings for the 224x224 images with a
frozen google/siglip2-base-patch16-224 vision tower.  Items without an available
image get an all-zero vector; availability travels separately through the
missing-modality indicator (Eq. 16) and the trainable missingness vector.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CACHE, DATA, IMAGES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="google/siglip2-base-patch16-224")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--tag", default="siglip2_base")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    def load_vision(name):
        """google/siglip2-base-patch16-224 loads as SiglipModel with a SiglipVisionModel tower."""
        try:
            from transformers import SiglipVisionModel as VM
            return VM.from_pretrained(name, dtype=torch.float32).eval()
        except Exception as e1:
            print("  [info] direct vision-tower load failed, using AutoModel:", e1)
            full = AutoModel.from_pretrained(name, dtype=torch.float32)
            return full.vision_model.eval()

    with open(os.path.join(DATA, f"{a.dataset}.pkl"), "rb") as f:
        ds = pickle.load(f)
    n_items = len(ds["items"])
    d = os.path.join(IMAGES, a.dataset)
    paths = [os.path.join(d, f"{i}.jpg") for i in range(n_items)]
    avail = np.array([os.path.exists(p) for p in paths])
    print(f"[{a.dataset}] {avail.sum():,}/{n_items:,} images present "
          f"({100*avail.mean():.2f}%)", flush=True)

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(a.model)
    model = load_vision(a.model)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"  model loaded in {time.time()-t0:.1f}s params={nparam/1e6:.1f}M "
          f"hidden={model.config.hidden_size}", flush=True)
    dim = model.config.hidden_size

    idxs = [i for i in range(n_items) if avail[i]]
    fn = os.path.join(CACHE, f"{a.dataset}_{a.tag}.npy")
    ck = fn + ".partial.npz"
    out = np.zeros((n_items, dim), dtype=np.float32)
    done_mask = np.zeros(n_items, dtype=bool)
    if os.path.exists(ck) and not a.bench:
        try:
            z = np.load(ck)
            if z["out"].shape == out.shape:
                out, done_mask = z["out"], z["done"]
                print(f"  resuming: {int(done_mask.sum()):,} images already encoded",
                      flush=True)
        except Exception as e:
            print("  [warn] could not resume:", e)
    if a.bench:
        idxs = idxs[: a.bench]
    else:
        idxs = [i for i in idxs if not done_mask[i]]
    if not idxs:
        np.save(fn, out)
        np.save(os.path.join(CACHE, f"{a.dataset}_img_avail.npy"), avail)
        print("  nothing to do")
        return
    print(f"  to encode: {len(idxs):,}", flush=True)

    t0, done = time.time(), 0
    with torch.inference_mode():
        for s in range(0, len(idxs), a.batch):
            chunk = idxs[s: s + a.batch]
            ims = [Image.open(paths[i]).convert("RGB") for i in chunk]
            pv = proc(images=ims, return_tensors="pt")["pixel_values"]
            o = model(pixel_values=pv)
            h = getattr(o, "pooler_output", None)
            if h is None:
                h = o.last_hidden_state.mean(dim=1)
            v = torch.nn.functional.normalize(h.float(), p=2, dim=1).numpy()
            for j, i in enumerate(chunk):
                out[i] = v[j]
            done_mask[chunk] = True
            done += len(chunk)
            if (s // a.batch) % 20 == 0:
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                print(f"   {done:,}/{len(idxs):,}  {rate:.1f} im/s  "
                      f"ETA {(len(idxs)-done)/max(rate,1e-9)/60:.1f} min", flush=True)
                if not a.bench:
                    np.savez(ck, out=out, done=done_mask)
    el = time.time() - t0
    print(f"  encoded {done:,} images in {el/60:.2f} min")
    if a.bench:
        print("BENCH ONLY - not saving")
        return

    fn = os.path.join(CACHE, f"{a.dataset}_{a.tag}.npy")
    np.save(fn, out)
    if os.path.exists(fn + ".partial.npz"):
        os.remove(fn + ".partial.npz")
    np.save(os.path.join(CACHE, f"{a.dataset}_img_avail.npy"), avail)
    json.dump({"model": a.model, "dim": int(dim), "n_items": int(n_items),
               "n_images": int(avail.sum()),
               "coverage_pct": round(100 * float(avail.mean()), 2),
               "seconds": round(el, 1), "params_M": round(nparam / 1e6, 2),
               "throughput_images_per_s": round(len(idxs) / max(el, 1e-9), 2)},
              open(fn.replace(".npy", ".json"), "w"), indent=2)
    print("saved", fn, out.shape)


if __name__ == "__main__":
    main()
