"""Short end-to-end validation of the training path on real data (MovieLens-1M).

Uses random stand-in foundation features so it does not depend on the encoder
jobs, runs a few real gradient steps and one evaluation, and prints timings.
Writes nothing into results/.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines as B
import common
import train_eval as TE

torch.set_num_threads(int(os.environ.get("NTHREADS", "3")))

ds = common.load_ds("movielens_1m")
n_items = len(ds["items"])
print("dataset stats:", ds["stats"], flush=True)

rng = np.random.RandomState(0)
feats = (torch.from_numpy(rng.randn(n_items, 768).astype(np.float32)),
         torch.from_numpy(rng.randn(n_items, 1024).astype(np.float32)),
         torch.from_numpy(rng.rand(n_items) > 0.5), 768, 1024)

cfg = dict(TE.DEFAULT_CFG)
cfg.update(dict(d=128, dz=64, l_max=20, epochs=1, batch=256, patience=99, kl_warmup=1))

L = cfg["l_max"]
t0 = time.time()
tr = TE.build_split(ds, "train", L)
va = TE.build_split(ds, "val", L)
te = TE.build_split(ds, "test", L)
print(f"splits in {time.time()-t0:.1f}s  train={tr.idx.shape}", flush=True)


class FakeNeg:
    """Stand-in for MixedNegatives (avoids needing the cached LLM space)."""

    def __init__(self, split, k=9):
        self.neg = np.random.randint(0, n_items, size=(*split.idx.shape, k)).astype(np.int32)

    def sample_inbatch(self, t, n):
        return t[np.random.randint(0, len(t), size=n)]


neg = FakeNeg(tr)
m = TE.build_model(ds, cfg, feats)
print(f"model: {sum(p.numel() for p in m.parameters()):,} params", flush=True)

# --- 3 real gradient steps
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
t0 = time.time()
for step in range(3):
    b = np.arange(step * cfg["batch"], (step + 1) * cfg["batch"])
    idx, rbin, tb, pad, tgt = tr.to_torch(b)
    negb = torch.from_numpy(neg.neg[b].astype(np.int64))
    valid = tgt >= 0
    ms = (torch.rand_like(tgt, dtype=torch.float) < cfg["mip_rate"]) & valid
    idx_m = idx.clone()
    idx_m[ms] = tr.n_items + 1
    out = m.encode_sequence(idx_m, rbin, tb, pad)
    V = m.candidate_vectors(use_cache=False)
    rs = valid & (~ms)
    bi, ti = torch.nonzero(rs, as_tuple=True)
    zi = out["z"][bi, ti]
    pos = tgt[bi, ti]
    s_pos = (zi * V[pos]).sum(-1)
    s_neg = torch.einsum("nd,nkd->nk", zi, V[negb[bi, ti]])
    loss_rec = __import__("model").sampled_softmax_rec(s_pos, s_neg)
    vmask = (~pad).float()
    kl = (out["kl"] * vmask).sum() / vmask.sum().clamp_min(1)
    loss = loss_rec + 0.2 * 0.0 + 1e-4 * kl
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
    opt.step()
    print(f"  step {step}: rec={float(loss_rec):.4f} kl={float(kl):.2f} "
          f"grads_ok={all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())}",
          flush=True)
el = time.time() - t0
print(f"3 steps in {el:.1f}s -> {el/3:.2f}s/step, "
      f"epoch estimate {(el/3)*(len(tr.users)/cfg['batch']):.0f}s "
      f"({len(tr.users)/cfg['batch']:.0f} steps)", flush=True)

# --- baselines build check
for name in ("sasrec", "gru4rec", "caser", "bert4rec"):
    kw = dict(n_items=n_items, n_bins=5, dz=64, d=128, l_max=20)
    mm = {"sasrec": B.SASRec, "gru4rec": B.GRU4Rec, "caser": B.Caser,
          "bert4rec": B.BERT4Rec}[name](**kw)
    o = mm.encode_sequence(*tr.to_torch(np.arange(8))[:4])
    Vb = mm.candidate_vectors()
    print(f"  {name}: out={tuple(o['mu'].shape)} V={tuple(Vb.shape)}", flush=True)

mm = B.SASRec(n_items=n_items, n_bins=5, dz=64, d=128, l_max=20, content="gated", d_img=768, d_txt=1024)
mm.set_item_features(img_feat=feats[0], txt_feat=feats[1], img_avail=feats[2])
o = mm.encode_sequence(*tr.to_torch(np.arange(8))[:4])
print("  mmsasrec(gated):", tuple(o["mu"].shape), tuple(mm.candidate_vectors().shape), flush=True)

bm = B.BPRMF(n_items=n_items, n_users=len(ds["seqs"]), dz=64)
print("  bprmf: u=", tuple(bm.user_vectors(torch.arange(4)).shape),
      "V=", tuple(bm.candidate_vectors().shape), flush=True)

# --- evaluation path for a static-user model
class R(common.load_json.__class__ if False else object):
    pass


t0 = time.time()
res = TE.evaluate(m, va, ds)
print(f"eval on {va.idx.shape[0]:,} users in {time.time()-t0:.1f}s "
      f"| NDCG@10={res['full']['NDCG@10']:.5f}", flush=True)

bpr = B.BPRMF(n_items=n_items, n_users=len(ds["seqs"]), dz=64)
res2 = TE.evaluate(bpr, va, ds)
print(f"static-user (BPR-MF) eval OK: NDCG@10={res2['full']['NDCG@10']:.5f}", flush=True)
print("\nVALIDATION OK")
