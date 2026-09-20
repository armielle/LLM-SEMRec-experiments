"""
Training / evaluation engine for LLM-SEMRec and the re-implemented baselines.

Protocol (paper Sec. VI-B, VI-E):
  * leave-two-out chronological split; no future interaction is ever used
  * max history length L (most recent L interactions, left padded)
  * teacher forcing: the sequence encoder is causal, so the prefix-pooled user
    representation at position k depends only on interactions <= k; all positions
    are therefore supervised in a single forward pass (Eq. 33-35 on every causal
    prefix).  At inference the last valid prefix is used, which is exactly Eq. 35
    over the whole history.
  * mixed negative sampling (Eq. 53): 1 in-batch + 4 popularity + 5 semantic hard
  * AdamW lr 1e-3, weight decay 1e-5 (the lambda_2 L2 term), grad clipping 5.0,
    early stopping on validation NDCG@10
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M
from common import (RESULTS, build_eval_candidates, list_metrics, load_ds,
                    metrics_from_ranks, save_json)
from paths import CACHE


# --------------------------------------------------------------------------- tensors
class Split:
    def __init__(self, idx, rbin, tb, tgt, padmask, users, n_items):
        self.idx = idx
        self.rbin = rbin
        self.tb = tb
        self.tgt = tgt
        self.padmask = padmask
        self.users = users
        self.n_items = n_items

    def to_torch(self, b):
        return (torch.from_numpy(self.idx[b]), torch.from_numpy(self.rbin[b]),
                torch.from_numpy(self.tb[b]), torch.from_numpy(self.padmask[b]),
                torch.from_numpy(self.tgt[b]))


def build_split(ds, split, L):
    users = sorted(ds["seqs"].keys())
    n_items = len(ds["items"])
    N = len(users)
    idx = np.full((N, L), n_items, dtype=np.int64)
    rbin = np.zeros((N, L), dtype=np.int64)
    tb = np.zeros((N, L), dtype=np.int64)
    tgt = np.full((N, L), -1, dtype=np.int64)
    padmask = np.ones((N, L), dtype=bool)
    edges = M.time_bucket_edges().numpy()

    for r, u in enumerate(users):
        arr = np.asarray(ds["seqs"][u], dtype=np.int64)
        hist = arr[:-2] if split in ("train", "val") else arr[:-1]
        if len(hist) > L:
            hist = hist[-L:]
        n = len(hist)
        s = L - n
        idx[r, s:] = hist[:, 0]
        rbin[r, s:] = hist[:, 1]
        if n > 1:
            d = np.concatenate([[0], np.diff(hist[:, 2])])
        else:
            d = np.zeros(1, dtype=np.int64)
        b = np.zeros(n, dtype=np.int64)
        b[d > 0] = 1
        for i, e in enumerate(edges):
            b[d > e] = i + 2
        tb[r, s:] = np.clip(b, 0, M.N_TIME - 1)
        padmask[r, s:] = False
        if split == "train" and n > 1:
            tgt[r, s:s + n - 1] = hist[1:, 0]
        elif split in ("val", "test"):
            tgt[r, L - 1] = arr[-2][0] if split == "val" else arr[-1][0]
    return Split(idx, rbin, tb, tgt, padmask, users, n_items)


# --------------------------------------------------------------------------- negatives
class MixedNegatives:
    """Eq. 53 mixed negative sampling: precomputed once and reused across epochs."""

    def __init__(self, ds, train_split, k_pop=4, k_sem=5, seed=0, uniform=False):
        rng = np.random.RandomState(seed)
        n_items = len(ds["items"])
        pop = np.zeros(n_items, dtype=np.float64)
        for u, v in ds["train_targets"].items():
            for (i, _, _) in v:
                pop[i] += 1
        self.pop = pop
        p = np.power(pop + 1e-9, 0.75)
        self.p_pop = p / p.sum()
        self.k_pop, self.k_sem = k_pop, k_sem
        self.uniform = uniform

        txt_p = os.path.join(CACHE, f"{ds['name']}_qwen3_0.6b.npy")
        if os.path.exists(txt_p):
            txt = np.load(txt_p).astype(np.float32)
            tn = txt / (np.linalg.norm(txt, axis=1, keepdims=True) + 1e-9)
            knn, B = [], 2048
            for s in range(0, n_items, B):
                sim = tn[s:s + B] @ tn.T
                for j in range(sim.shape[0]):
                    sim[j, s + j] = -2
                kk = min(50, n_items - 1)
                knn.append(np.argpartition(-sim, kk, axis=1)[:, :kk])
            self.knn = np.concatenate(knn, 0)
            self.has_semantic = True
        else:
            print("  [warn] semantic embeddings not cached yet -> falling back to "
                  "popularity + uniform negatives only (ID-only baselines)")
            self.knn = None
            self.has_semantic = False

        seen = {u: np.array(sorted({x[0] for x in v}), dtype=np.int64)
                for u, v in ds["seqs"].items()}
        N, L = train_split.idx.shape
        K = k_pop + k_sem
        neg = np.zeros((N, L, K), dtype=np.int32)
        pool = np.arange(n_items)
        for r, u in enumerate(train_split.users):
            sv = seen[u]
            for t in range(L):
                if train_split.tgt[r, t] < 0:
                    continue
                pos = train_split.tgt[r, t]
                if uniform:
                    # ablation: all negatives drawn uniformly from the training catalogue
                    cand = rng.randint(0, n_items, size=K)
                    bad = np.isin(cand, sv)
                    while bad.any():
                        cand[bad] = rng.randint(0, n_items, size=int(bad.sum()))
                        bad = np.isin(cand, sv)
                    neg[r, t] = cand
                    continue
                cand = rng.choice(pool, size=k_pop * 4, p=self.p_pop, replace=True)
                cand = cand[~np.isin(cand, sv)][:k_pop]
                while len(cand) < k_pop:
                    c = rng.randint(0, n_items)
                    if c not in sv:
                        cand = np.append(cand, c)
                nn = self.knn[pos][~np.isin(self.knn[pos], sv)][:k_sem] \
                    if self.knn is not None else np.array([], dtype=np.int64)
                while len(nn) < k_sem:
                    c = rng.randint(0, n_items)
                    if c not in sv and c != pos:
                        nn = np.append(nn, c)
                neg[r, t] = np.concatenate([cand, nn])
        self.neg = neg
        print(f"  mixed negatives precomputed: {neg.shape}"
              f"{' (UNIFORM ablation)' if uniform else ''}")

    def sample_inbatch(self, targets, n):
        return targets[np.random.randint(0, len(targets), size=n)]


# --------------------------------------------------------------------------- augmentation
def augment(batch, p_item=0.1, p_mod=0.1, min_keep=0.7, pad_id=None):
    """Eq. 25 views: item dropout + contiguous temporal crop + modality dropout.

    All three transformations preserve chronological causality and never touch a
    position that would not be available at prediction time.
    """
    idx, rbin, tb, padmask = batch
    B, L = idx.shape
    dev = idx.device
    pid = pad_id if pad_id is not None else getattr(augment, "pad_id", idx.max().item())
    new_idx = idx.clone()
    new_pad = padmask.clone()
    mod_mask = torch.ones(B, L, 2, device=dev)
    for b in range(B):
        valid = torch.nonzero(~padmask[b], as_tuple=False).flatten()
        n = len(valid)
        if n == 0:
            continue
        keep_len = max(int(math.ceil(min_keep * n)), 1)
        start = int(torch.randint(0, n - keep_len + 1, (1,)).item())
        sel = valid[start:start + keep_len]
        keep = torch.zeros(L, dtype=torch.bool, device=dev)
        keep[sel] = True
        drop = torch.rand(len(sel), device=dev) < p_item          # (i) item dropout
        keep[sel[drop]] = False
        new_idx[b][~keep] = pid
        new_pad[b] = ~keep
        md = (torch.rand(L, 2, device=dev) >= p_mod).float()      # (iii) modality dropout
        mod_mask[b] = md
    return new_idx, rbin, tb, new_pad, mod_mask


# --------------------------------------------------------------------------- config
DEFAULT_CFG = dict(
    d=128, dz=64, l_max=20, n_heads=4, n_tf_layers=2, ff=1024, dropout=0.2,
    dilations=(1, 2), use_vae=True, use_img=True, use_txt=True, gated_fusion=True,
    lam_mip=0.2, lam_seqcl=0.1, lam_cmcl=0.1, lam_isc=0.5, beta=1e-4,
    tau_s=0.07, tau_m=0.07, gamma=0.0, batch=256, lr=1e-3, wd=1e-5, clip=5.0,
    epochs=40, patience=10, mip_rate=0.15, seqcl_every=2, kl_warmup=5,
    k_pop=4, k_sem=5, seed=0, uniform_negatives=False,
)


def load_features(ds_name, cfg):
    img = txt = None
    d_img, d_txt = 768, 1024
    img_avail = None
    if cfg.get("use_img", True):
        pi = os.path.join(CACHE, f"{ds_name}_siglip2_base.npy")
        if os.path.exists(pi):
            img = torch.from_numpy(np.load(pi))
            d_img = img.shape[1]
            pa = os.path.join(CACHE, f"{ds_name}_img_avail.npy")
            if os.path.exists(pa):
                img_avail = torch.from_numpy(np.load(pa)).bool()
    if cfg.get("use_txt", True):
        pt = os.path.join(CACHE, f"{ds_name}_qwen3_0.6b.npy")
        if os.path.exists(pt):
            txt = torch.from_numpy(np.load(pt))
            d_txt = txt.shape[1]
    return img, txt, img_avail, d_img, d_txt


def build_model(ds, cfg, feats):
    img, txt, img_avail, d_img, d_txt = feats
    if img is None and cfg.get("use_img", True):
        # no visual features exist for this dataset (e.g. MovieLens-1M): the image
        # modality must be *absent*, not a zero vector flagged as available
        cfg = dict(cfg)
        cfg["use_img"] = False
    if txt is None and cfg.get("use_txt", True):
        cfg = dict(cfg)
        cfg["use_txt"] = False
    n_items = len(ds["items"])
    n_bins = int(ds["stats"]["n_bins"])
    counts = np.zeros(n_items)
    rsum = np.zeros(n_items)
    for u, v in ds["train_targets"].items():
        for (i, b, _) in v:
            counts[i] += 1
            rsum[i] += b
    mean_rbin = np.zeros(n_items, dtype=np.int64)
    nz = counts > 0
    mean_rbin[nz] = np.round(rsum[nz] / counts[nz]).astype(np.int64)
    m = M.LLMSEMRec(cfg, n_items, n_bins, d_img, d_txt,
                    has_img=img_avail, has_txt=None, mean_rating_bin=mean_rbin)
    m.set_item_features(img_feat=img, txt_feat=txt, img_avail=img_avail)
    return m


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, split, ds, cand_sets=None, gamma=0.0, chunk=256,
             return_lists=False, item_repr=None, k=10, all_scores=False):
    """Full-catalogue ranking; also reports the sampled-100 protocol (Sec. VI-B)."""
    model.eval()
    model.invalidate_candidates()
    V = model.candidate_vectors()
    N = split.idx.shape[0]
    ranks_full = np.zeros(N, dtype=np.int64)
    ranks_samp = np.zeros(N, dtype=np.int64)
    sigmas = np.zeros(N, dtype=np.float64)
    topk = [] if return_lists else None
    samp_scores = np.zeros((N, cand_sets.shape[1]), dtype=np.float32) if cand_sets is not None else None
    score_rows = []
    target = split.tgt[:, -1]
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        idx, rb, tb, pm, _ = split.to_torch(np.arange(s, e))
        if getattr(model, "static_user", False):
            # non-sequential baseline: the user vector does not depend on the history
            mu = model.user_vectors(torch.arange(s, e))
            sig = torch.ones_like(mu)
        else:
            out = model.encode_sequence(idx, rb, tb, pm)
            mu = out["mu"][:, -1] if out["mu"].dim() == 3 else out["mu"]
            sig = out["sig"][:, -1] if out["sig"].dim() == 3 else out["sig"]
        sigmas[s:e] = sig.mean(-1).cpu().numpy()
        sc = model.score(mu, sig, V, gamma=gamma).cpu().numpy()
        tgt_col = target[s:e]
        rows = np.arange(e - s)
        ranks_full[s:e] = (sc > sc[rows, tgt_col][:, None]).sum(1)
        if cand_sets is not None:
            cand = cand_sets[s:e]
            sc_c = sc[rows[:, None], cand]
            ranks_samp[s:e] = (sc_c[:, 1:] > sc_c[:, :1]).sum(1)
            samp_scores[s:e] = sc_c
        if return_lists:
            part = np.argpartition(-sc, k, axis=1)[:, :k]
            order = np.take_along_axis(
                part, np.argsort(-np.take_along_axis(sc, part, 1), 1), 1)
            topk.append(order)
        if all_scores:
            score_rows.append(sc)
    res = {"full": metrics_from_ranks(ranks_full, ks=(5, 10, 20)),
           "sampled": metrics_from_ranks(ranks_samp, ks=(5, 10, 20)),
           "sigma_mean": float(sigmas.mean()),
           "ranks_full": ranks_full if return_lists else None,
           "sigma": sigmas if return_lists else None,
           "topk": np.concatenate(topk, 0) if return_lists else None,
           "sampled_scores": samp_scores,
           "scores": np.concatenate(score_rows, 0) if all_scores else None}
    model.train()
    return res


# --------------------------------------------------------------------------- training
def train(ds, cfg, feats=None, verbose=True, max_seconds=None, neg_sampler=None,
          train_split=None, val_split=None):
    t_start = time.time()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    L = cfg["l_max"]
    ds_name = ds["name"]
    if feats is None:
        feats = load_features(ds_name, cfg)
    train_split = train_split or build_split(ds, "train", L)
    val_split = val_split or build_split(ds, "val", L)
    if neg_sampler is None:
        neg_sampler = MixedNegatives(ds, train_split, k_pop=cfg["k_pop"], k_sem=cfg["k_sem"])
    augment.pad_id = train_split.n_items

    m = build_model(ds, cfg, feats)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    N = train_split.idx.shape[0]
    hist, best, best_state, bad, step = [], -1.0, None, 0, 0
    for ep in range(cfg["epochs"]):
        perm = np.random.permutation(N)
        tot = {k: 0.0 for k in ("rec", "mip", "cl", "cmcl", "kl")}
        nb, t0 = 0, time.time()
        beta = cfg["beta"] * min(1.0, (ep + 1) / max(cfg["kl_warmup"], 1))
        for s in range(0, N, cfg["batch"]):
            b = perm[s:s + cfg["batch"]]
            idx, rbin, tb, pad, tgt = train_split.to_torch(b)
            neg = torch.from_numpy(neg_sampler.neg[b].astype(np.int64))
            valid_t = tgt >= 0

            # ---- masked item prediction (Eq. 24): mask the inputs in place
            mask_sel = (torch.rand_like(tgt, dtype=torch.float) < cfg["mip_rate"]) & valid_t
            if cfg["lam_mip"] <= 0:
                mask_sel = torch.zeros_like(mask_sel)
            idx_m = idx.clone()
            idx_m[mask_sel] = train_split.n_items + 1
            out = m.encode_sequence(idx_m, rbin, tb, pad)
            z_all = out["z"]                       # [B,L,dz]

            V = m.candidate_vectors(use_cache=False)

            # ---- L_Rec (Eq. 54) on positions that were not masked
            rec_sel = valid_t & (~mask_sel)
            if cfg.get("min_hist", 0) > 0:
                # supervise only prefixes that carry at least min_hist interactions
                pos_idx = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0)
                n_valid = (~pad).sum(-1, keepdim=True)
                n_seen = (pos_idx - (idx.shape[1] - n_valid) + 1)
                rec_sel = rec_sel & (n_seen >= cfg["min_hist"])
            bi, ti = torch.nonzero(rec_sel, as_tuple=True)
            zi = z_all[bi, ti]
            pos_i = tgt[bi, ti]
            s_pos = (zi * V[pos_i]).sum(-1)
            neg_i = neg[bi, ti]
            s_neg = torch.einsum("nd,nkd->nk", zi, V[neg_i])
            ib = neg_sampler.sample_inbatch(pos_i.numpy(), len(pos_i))
            s_ib = (zi * V[torch.from_numpy(ib)]).sum(-1, keepdim=True)
            l_rec = M.sampled_softmax_rec(s_pos, torch.cat([s_neg, s_ib], 1))

            # ---- L_MIP (Eq. 24)
            if cfg["lam_mip"] > 0 and mask_sel.any():
                bi2, ti2 = torch.nonzero(mask_sel, as_tuple=True)
                l_mip = M.mip_loss(m, out["q"], V, tgt[bi2, ti2], (bi2, ti2),
                                   neg[bi2, ti2])
            else:
                l_mip = torch.zeros((), device=idx.device)

            # ---- L_KL (Eq. 40)
            vmask = (~pad).float()
            l_kl = (out["kl"] * vmask).sum() / vmask.sum().clamp_min(1)

            # ---- L_SeqCL (Eq. 25) on a fraction of steps for cost
            l_seq = torch.zeros((), device=idx.device)
            if cfg["lam_seqcl"] > 0 and (step % max(cfg["seqcl_every"], 1) == 0):
                views = []
                for _ in range(2):
                    ai, ar, at, ap, am = augment((idx, rbin, tb, pad),
                                                 pad_id=train_split.n_items)
                    o = m.encode_sequence(ai, ar, at, ap, mod_mask=am)
                    views.append(o["z"][:, -1])
                l_seq = M.seqcl_loss(views[0], views[1], tau=cfg["tau_s"])

            # ---- L_CMCL (Eq. 26-28)
            l_cmcl = torch.zeros((), device=idx.device)
            if (cfg["lam_cmcl"] > 0 and m.inputs._imgfeat is not None
                    and m.inputs._txtfeat is not None):
                items = torch.unique(tgt[valid_t])
                if len(items) > 4:
                    ii = items.unsqueeze(0)
                    _, img_e, txt_e, _, a_img, a_txt = m.inputs.item_modalities(ii)
                    img_e, txt_e = img_e[0], txt_e[0]
                    ok = (a_img[0] > 0.5) & (a_txt[0] > 0.5)
                    if ok.sum() > 4:
                        l_cmcl = (M.itc_loss(img_e[ok], txt_e[ok], tau=cfg["tau_m"])
                                  + cfg["lam_isc"] * M.isc_loss(
                                      m.inputs.id_emb(ii)[0][ok], txt_e[ok], cfg["tau_m"]))

            loss = (l_rec + cfg["lam_mip"] * l_mip + cfg["lam_seqcl"] * l_seq
                    + cfg["lam_cmcl"] * l_cmcl + beta * l_kl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), cfg["clip"])
            opt.step()

            tot["rec"] += float(l_rec.detach()) * len(b)
            tot["mip"] += float(l_mip.detach()) * len(b)
            tot["cl"] += float(l_seq.detach()) * len(b)
            tot["cmcl"] += float(l_cmcl.detach()) * len(b)
            tot["kl"] += float(l_kl.detach()) * len(b)
            nb += len(b)
            step += 1
            if max_seconds and time.time() - t_start > max_seconds:
                break
        sched.step()
        vs = evaluate(m, val_split, ds, gamma=0.0)
        ndcg = vs["full"]["NDCG@10"]
        hist.append({"epoch": ep + 1, **{f"loss_{k}": v / max(nb, 1) for k, v in tot.items()},
                     "val_NDCG@10": ndcg, "val_Recall@10": vs["full"]["Recall@10"],
                     "val_MRR@10": vs["full"]["MRR@10"], "sec": round(time.time() - t0, 1)})
        if verbose:
            print(f"  ep{ep+1:02d} rec={tot['rec']/max(nb,1):.4f} mip={tot['mip']/max(nb,1):.4f} "
                  f"cl={tot['cl']/max(nb,1):.4f} cmcl={tot['cmcl']/max(nb,1):.4f} "
                  f"kl={tot['kl']/max(nb,1):.2f} | val NDCG@10={ndcg:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if ndcg > best + 1e-5:
            best, bad = ndcg, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                if verbose:
                    print(f"  early stopping at epoch {ep+1}")
                break
        if max_seconds and time.time() - t_start > max_seconds:
            break
    if best_state is not None:
        m.load_state_dict(best_state)
    return m, {"history": hist, "best_val_NDCG@10": best, "n_params": n_params,
               "train_seconds": round(time.time() - t_start, 1)}


# --------------------------------------------------------------------------- driver
def run_one(ds_name, cfg, out_name, seeds=(0,), max_seconds=None, verbose=True,
            ds=None, splits=None, feats=None, neg=None, cands=None):
    ds = ds or load_ds(ds_name)
    L = cfg["l_max"]
    if splits is None:
        splits = (build_split(ds, "train", L), build_split(ds, "val", L),
                  build_split(ds, "test", L))
    train_split, val_split, test_split = splits
    if feats is None:
        feats = load_features(ds_name, cfg)
    if neg is None:
        neg = MixedNegatives(ds, train_split, k_pop=cfg["k_pop"], k_sem=cfg["k_sem"])
    if cands is None:
        cands = build_eval_candidates(ds, n_neg=100, seed=42)
    cand_test = cands["test"]["items"]
    assert cands["test"]["users"] == test_split.users

    pop = np.zeros(len(ds["items"]))
    for u, v in ds["train_targets"].items():
        for (i, _, _) in v:
            pop[i] += 1

    runs = []
    for sd in seeds:
        c = dict(cfg)
        c["seed"] = sd
        print(f"\n### {out_name} seed={sd}", flush=True)
        m, info = train(ds, c, feats=feats, verbose=verbose, max_seconds=max_seconds,
                        neg_sampler=neg, train_split=train_split, val_split=val_split)
        res = evaluate(m, test_split, ds, cand_sets=cand_test, gamma=c.get("gamma", 0.0),
                       return_lists=True)
        V = m.candidate_vectors().detach().cpu().numpy()
        lm = list_metrics(res["topk"], pop, len(ds["items"]), item_repr=V, k=10)
        runs.append({"seed": sd, "test_full": res["full"], "test_sampled": res["sampled"],
                     "sigma_mean": res["sigma_mean"], "list_metrics": lm,
                     "history": info["history"], "n_params": info["n_params"],
                     "train_seconds": info["train_seconds"],
                     "ranks_full": res["ranks_full"].tolist(),
                     "sigma": res["sigma"].tolist(),
                     "topk": res["topk"].tolist(), "test_users": test_split.users})
        print(f"  TEST full  R@10={res['full']['Recall@10']:.4f} "
              f"NDCG@10={res['full']['NDCG@10']:.4f} MRR@10={res['full']['MRR@10']:.4f} | "
              f"R@20={res['full']['Recall@20']:.4f} NDCG@20={res['full']['NDCG@20']:.4f} "
              f"MRR@20={res['full']['MRR@20']:.4f}", flush=True)
        print(f"  TEST samp  R@10={res['sampled']['Recall@10']:.4f} "
              f"NDCG@10={res['sampled']['NDCG@10']:.4f} MRR@10={res['sampled']['MRR@10']:.4f}",
              flush=True)
        print(f"  list metrics: { {k: round(v,4) for k,v in lm.items()} }", flush=True)
    agg = {k: {"mean": float(np.mean([r["test_full"][k] for r in runs])),
               "std": float(np.std([r["test_full"][k] for r in runs]))}
           for k in runs[0]["test_full"]}
    out = {"model": out_name, "dataset": ds_name, "config": cfg, "runs": runs,
           "aggregate_full": agg, "n_seeds": len(seeds)}
    save_json(out, os.path.join(RESULTS, f"{out_name}.json"))
    return out
