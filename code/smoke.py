"""Smoke test + throughput benchmark: runs a real training epoch and times it."""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import model as M
import train_eval as TE

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))

ds = common.load_ds(sys.argv[1] if len(sys.argv) > 1 else "amazon_beauty")
print("dataset:", ds["stats"], flush=True)
n_items = len(ds["items"])

# synthetic stand-in features (independent of the frozen-encoder jobs)
rng = np.random.RandomState(0)
img = torch.from_numpy(rng.randn(n_items, 768).astype(np.float32))
txt = torch.from_numpy(rng.randn(n_items, 1024).astype(np.float32))
avail = torch.from_numpy((rng.rand(n_items) > 0.4))
feats = (img, txt, avail, 768, 1024)

cfg = dict(TE.DEFAULT_CFG)
cfg.update(dict(d=128, dz=64, l_max=20, epochs=1, batch=128, patience=99, kl_warmup=1))
print("cfg:", {k: cfg[k] for k in ("d", "dz", "l_max", "batch")}, flush=True)

t0 = time.time()
tr = TE.build_split(ds, "train", cfg["l_max"])
va = TE.build_split(ds, "val", cfg["l_max"])
te = TE.build_split(ds, "test", cfg["l_max"])
print(f"splits built in {time.time()-t0:.1f}s train={tr.idx.shape} test={te.idx.shape}", flush=True)
assert not np.any(np.diff(tr.padmask.astype(int), axis=1) > 0), \
    "padding must be a left prefix in every row"
valid = ~tr.padmask
print("valid train positions:", int(valid.sum()))

# ---- fake negative sampler (real one needs the cached LLM embeddings)
class FakeNeg:
    def __init__(self, split, k=9):
        self.neg = np.random.randint(0, n_items, size=(*split.idx.shape, k)).astype(np.int32)

    def sample_inbatch(self, t, n):
        return t[np.random.randint(0, len(t), size=n)]


neg = FakeNeg(tr)
t0 = time.time()
m = TE.build_model(ds, cfg, feats)
print(f"model built: {sum(p.numel() for p in m.parameters()):,} params "
      f"({time.time()-t0:.1f}s)", flush=True)

m.train()
t0 = time.time()
idx, rb, tb, pm, tgt = tr.to_torch(np.arange(64))
out = m.encode_sequence(idx, rb, tb, pm)
print("encode_sequence shapes:", {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)})
V = m.candidate_vectors(use_cache=False)
print(f"candidate_vectors {tuple(V.shape)} in {time.time()-t0:.2f}s (64-user batch)", flush=True)

# ---- one full training epoch with a wall-clock cap
print("\n-- running 1 training epoch (capped) --", flush=True)
cfg["max"] = None
t0 = time.time()
m2, info = TE.train(ds, cfg, feats=feats, verbose=True, max_seconds=240,
                    neg_sampler=neg, train_split=tr, val_split=va)
el = time.time() - t0
n_steps = len(ds["seqs"]) / cfg["batch"]
print(f"\nepoch wall time (partial if capped): {el:.0f}s | full-batch estimate: "
      f"{el / max(len(info['history']), 1):.0f}s per epoch "
      f"({n_steps:.0f} steps)", flush=True)

print("\n-- evaluation timing --", flush=True)
t0 = time.time()
res = TE.evaluate(m2, va, ds)
print(f"val eval on {va.idx.shape[0]:,} users took {time.time()-t0:.1f}s")
print("val metrics:", {k: round(v, 4) for k, v in res["full"].items()})
print("\nSMOKE OK")
