"""
Baselines for the LLM-SEMRec comparison (paper Sec. VI-C / Table III).

All sequential models share the same chronological splits, candidate sets,
negative sampler and evaluation code as the proposed model, so the comparison is
like-for-like.  These are re-implementations from the cited papers, not the
official codebases.

  Non-sequential : Popularity, BPR-MF
  Sequential     : GRU4Rec, Caser, SASRec, BERT4Rec
  Self-supervised: CL4SRec, S3-Rec
  Multimodal     : MM-SASRec (simple concatenation fusion)
  LLM-enhanced   : LLM-only (frozen Qwen3 semantics + item-ID fusion, no cross-modal
                   contrastive learning, no variational module)
  Multimodal SSL : MMSSL-lite (gated multimodal fusion + cross-modal contrastive,
                   no variational module)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as M


def _causal_mask(T, device):
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


# ============================================================================ base
class BaseSeq(nn.Module):
    def __init__(self, n_items, dz):
        super().__init__()
        self.n_items = n_items
        self.dz = dz
        self._cand_cache = None

    def invalidate_candidates(self):
        self._cand_cache = None

    def candidate_vectors(self, use_cache=True):
        if use_cache and self._cand_cache is not None:
            return self._cand_cache
        V = self._item_matrix()
        if use_cache and not self.training:
            self._cand_cache = V
        return V

    def _item_matrix(self):
        raise NotImplementedError

    @staticmethod
    def score(mu, sig, V, gamma=0.0):
        return mu @ V.t()


# ============================================================================ transformer family
class SASRec(BaseSeq):
    """Self-attentive sequential recommendation with optional multimodal item fusion."""

    def __init__(self, n_items, n_bins, dz=64, d=128, n_heads=4, n_layers=2, ff=256,
                 dropout=0.2, l_max=20, content="none", d_img=768, d_txt=1024,
                 use_time=True, use_fb=True, **kw):
        super().__init__(n_items, dz)
        self.l_max = l_max
        self.content = content
        self.d = d
        self.d_img = d_img
        self.d_txt = d_txt
        self.item_emb = nn.Embedding(n_items + 1, d)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        self.pos = nn.Embedding(l_max, d)
        self.rbin_emb = nn.Embedding(n_bins, d) if use_fb else None
        self.time_emb = nn.Embedding(M.N_TIME, d) if use_time else None
        self.drop = nn.Dropout(dropout)
        if content == "concat":
            self.cat = nn.Linear(d + d_img + d_txt, d)
        elif content == "txt":
            self.cat = nn.Linear(d + d_txt, d)
        elif content == "gated":
            nmod = 3
            self.Wg = nn.Linear(nmod * d + nmod, nmod)
            self.Wf = nn.Linear(d, d)
            self.img_proj = nn.Linear(d_img, d)
            self.txt_proj = nn.Linear(d_txt, d)
        self.enc = M.TransformerStack(d, n_heads, n_layers, ff, dropout)
        self.out = nn.Linear(d, dz)
        self._imgfeat = self._txtfeat = self._imgavail = None

    def set_item_features(self, img_feat=None, txt_feat=None, img_avail=None):
        self._imgfeat = img_feat
        self._txtfeat = txt_feat
        self._imgavail = img_avail
        self.invalidate_candidates()

    def _item_parts(self, idx):
        """Item-side representations, always shaped [B, T, d] (+ flat auxiliaries)."""
        B, T = idx.shape
        flat = idx.reshape(-1)
        safe = flat.clamp(0, self.n_items - 1)
        valid = (flat != self.n_items).float()
        e = self.item_emb(flat)
        if self.content == "none":
            return e.view(B, T, -1), None, None, None, None
        txt = (self._txtfeat[safe] if self._txtfeat is not None
               else torch.zeros(flat.numel(), self.d_txt))
        img = (self._imgfeat[safe] if self._imgfeat is not None
               else torch.zeros(flat.numel(), self.d_img))
        # a modality with no feature source must be *absent* (availability 0),
        # never a zero vector flagged as available
        if self._imgavail is not None:
            a_img = self._imgavail[safe].float() * valid
        else:
            a_img = valid.clone() if self._imgfeat is not None else torch.zeros_like(valid)
        a_txt = valid.clone() if self._txtfeat is not None else torch.zeros_like(valid)
        if self.content == "concat":
            h = self.cat(torch.cat([e, img, txt], -1)) * valid.unsqueeze(1)
        elif self.content == "txt":
            h = self.cat(torch.cat([e, txt], -1)) * valid.unsqueeze(1)
        else:  # gated fusion of {id, img, txt}
            mods = [e, self.img_proj(img) * a_img.unsqueeze(1),
                    self.txt_proj(txt) * a_txt.unsqueeze(1)]
            av = [valid, a_img, a_txt]
            o = self.Wg(torch.cat(mods + [a.unsqueeze(1) for a in av], -1))
            o = o.masked_fill(torch.stack(av, -1) < 0.5, -1e9)
            g = torch.softmax(o, -1) * torch.stack(av, -1)
            g = g / g.sum(-1, keepdim=True).clamp_min(1e-9)
            h = sum(g[:, i:i + 1] * mods[i] for i in range(3))
            h = h + F.gelu(self.Wf(h))
        return h.view(B, T, -1), txt, img, a_img, a_txt

    def encode_sequence(self, idx, rbin, tb, padmask):
        B, T = idx.shape
        e, _, _, _, _ = self._item_parts(idx)
        if self.rbin_emb is not None:
            e = e + self.rbin_emb(rbin)
        if self.time_emb is not None:
            e = e + self.time_emb(tb)
        x = self.drop(e + self.pos(torch.arange(T, device=idx.device)).unsqueeze(0))
        h = self.enc(x, M.build_attn_mask(B, T, padmask, causal=True, device=idx.device))
        z = self.out(h[:, -1])
        return {"mu": z, "sig": torch.ones_like(z), "z": z,
                "kl": torch.zeros(B, device=idx.device), "h": h}

    def _item_matrix(self):
        dev = self.item_emb.weight.device
        idx = torch.arange(self.n_items, device=dev).unsqueeze(0)
        h, _, _, _, _ = self._item_parts(idx)
        return self.out(h[0])


class BERT4Rec(SASRec):
    """Bidirectional encoder trained with masked item prediction (cloze)."""

    def encode_sequence(self, idx, rbin, tb, padmask):
        B, T = idx.shape
        idx2 = idx.clone()
        idx2[:, -1] = self.n_items                     # mask the target position
        e, _, _, _, _ = self._item_parts(idx2)
        if self.rbin_emb is not None:
            e = e + self.rbin_emb(rbin)
        if self.time_emb is not None:
            e = e + self.time_emb(tb)
        x = self.drop(e + self.pos(torch.arange(T, device=idx.device)).unsqueeze(0))
        h = self.enc(x, M.build_attn_mask(B, T, padmask, causal=False, device=idx.device))
        z = self.out(h[:, -1])
        return {"mu": z, "sig": torch.ones_like(z), "z": z,
                "kl": torch.zeros(B, device=idx.device), "h": h}

    def encode_masked(self, idx, rbin, tb, padmask):
        e, _, _, _, _ = self._item_parts(idx)
        if self.rbin_emb is not None:
            e = e + self.rbin_emb(rbin)
        if self.time_emb is not None:
            e = e + self.time_emb(tb)
        x = self.drop(e + self.pos(torch.arange(idx.shape[1], device=idx.device)).unsqueeze(0))
        return self.enc(x, M.build_attn_mask(idx.shape[0], idx.shape[1], padmask,
                                             causal=False, device=idx.device))

    def _item_matrix(self):
        dev = self.item_emb.weight.device
        idx = torch.arange(self.n_items, device=dev).unsqueeze(0)
        h, _, _, _, _ = self._item_parts(idx)
        return self.out(h[0])


class GRU4Rec(BaseSeq):
    def __init__(self, n_items, n_bins, dz=64, d=128, dropout=0.2, **kw):
        super().__init__(n_items, dz)
        self.item_emb = nn.Embedding(n_items + 1, d)
        self.rbin_emb = nn.Embedding(n_bins, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, dz)
        self._imgfeat = self._txtfeat = self._imgavail = None

    def set_item_features(self, **kw):
        pass

    def encode_sequence(self, idx, rbin, tb, padmask):
        e = self.item_emb(idx) + self.rbin_emb(rbin)
        q, _ = self.gru(self.drop(e))
        # left padding: the last position is always a valid interaction
        z = self.out(q[:, -1])
        return {"mu": z, "sig": torch.ones_like(z), "z": z,
                "kl": torch.zeros(idx.shape[0], device=idx.device), "h": q}

    def _item_matrix(self):
        dev = self.item_emb.weight.device
        idx = torch.arange(self.n_items, device=dev).unsqueeze(0)
        return self.out(self.item_emb(idx)[0])


class Caser(BaseSeq):
    """Convolutional sequence embedding (Tang & Wang, WSDM 2018)."""

    def __init__(self, n_items, n_bins, dz=64, d=128, l_max=20, n_h=8, dropout=0.2, **kw):
        super().__init__(n_items, d)
        self.item_emb = nn.Embedding(n_items + 1, d)
        self.l_max, self.d = l_max, d
        self.convs = nn.ModuleList([nn.Conv1d(d, n_h, k, padding=k - 1) for k in (1, 2, 3, 4)])
        self.vconv = nn.Conv1d(l_max, 8, 1)
        self.fc = nn.Linear(n_h * 4 + d * 8, d)
        self.drop = nn.Dropout(dropout)
        self._imgfeat = self._txtfeat = self._imgavail = None

    def set_item_features(self, **kw):
        pass

    def encode_sequence(self, idx, rbin, tb, padmask):
        e = self.item_emb(idx)
        x = e.transpose(1, 2)
        hs = [F.relu(c(x)).max(-1).values for c in self.convs]
        v = F.relu(self.vconv(e)).flatten(1)
        z = self.fc(self.drop(torch.cat(hs + [v], -1)))
        return {"mu": z, "sig": torch.ones_like(z), "z": z,
                "kl": torch.zeros(idx.shape[0], device=idx.device), "h": None}

    def _item_matrix(self):
        return self.item_emb.weight[: self.n_items]


# ============================================================================ BPR-MF
class BPRMF(nn.Module):
    """Non-sequential matrix factorisation with a BPR pairwise objective."""

    static_user = True

    def __init__(self, n_items, n_users=1, dz=64, **kw):
        super().__init__()
        self.n_items = n_items
        self.dz = dz
        self.i = nn.Embedding(n_items, dz)
        self.u = nn.Embedding(max(n_users, 1), dz)
        nn.init.normal_(self.i.weight, std=0.05)
        nn.init.normal_(self.u.weight, std=0.05)

    def user_vectors(self, rows):
        return self.u.weight[rows]

    def candidate_vectors(self, use_cache=True):
        return self.i.weight

    def invalidate_candidates(self):
        pass

    @staticmethod
    def score(mu, sig, V, gamma=0.0):
        return mu @ V.t()


# ============================================================================ training
def fit(model, ds, cfg, splits, feats=None, neg_sampler=None, objective="next",
        aux=(), verbose=True, max_seconds=None, name="baseline", unknown=None):
    """Generic trainer. objective in {'next','mip'}; aux subset of {'mip','seqcl','cmcl'}."""
    import train_eval as TE
    train_split, val_split = splits
    m = model
    if hasattr(m, "set_item_features") and feats is not None:
        img, txt, img_avail, _, _ = feats
        m.set_item_features(img_feat=img, txt_feat=txt, img_avail=img_avail)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    TE.augment.pad_id = train_split.n_items
    N = train_split.idx.shape[0]
    n_items = train_split.n_items
    best, best_state, bad, step = -1.0, None, 0, 0
    hist, t_start = [], time.time()

    for ep in range(cfg["epochs"]):
        perm = np.random.permutation(N)
        t0, tot, nb = time.time(), 0.0, 0
        for s in range(0, N, cfg["batch"]):
            b = perm[s:s + cfg["batch"]]
            idx, rbin, tb, pad, tgt = train_split.to_torch(b)
            neg = torch.from_numpy(neg_sampler.neg[b].astype(np.int64))
            valid_t = tgt >= 0
            V = m.candidate_vectors(use_cache=False)
            loss = torch.zeros((), device=idx.device)

            if objective == "mip":
                mask_sel = (torch.rand_like(tgt, dtype=torch.float) < cfg["mip_rate"]) & valid_t
                idx_m = idx.clone()
                idx_m[mask_sel] = n_items
                h = m.encode_masked(idx_m, rbin, tb, pad)
                bi, ti = torch.nonzero(mask_sel, as_tuple=True)
                head = m.out(h[bi, ti])
                pos = tgt[bi, ti]
                s_pos = (head * V[pos]).sum(-1)
                s_neg = torch.einsum("nd,nkd->nk", head, V[neg[bi, ti]])
                loss = loss + M.sampled_softmax_rec(s_pos, s_neg)
            else:
                out = m.encode_sequence(idx, rbin, tb, pad)
                z = out["mu"]
                bi, ti = torch.nonzero(valid_t, as_tuple=True)
                zi = z[bi]
                pos = tgt[bi, ti]
                s_pos = (zi * V[pos]).sum(-1)
                s_neg = torch.einsum("nd,nkd->nk", zi, V[neg[bi, ti]])
                ib = neg_sampler.sample_inbatch(pos.numpy(), len(pos))
                s_ib = (zi * V[torch.from_numpy(ib)]).sum(-1, keepdim=True)
                loss = loss + M.sampled_softmax_rec(s_pos, torch.cat([s_neg, s_ib], 1))

                if "mip" in aux:
                    ms = (torch.rand_like(tgt, dtype=torch.float) < cfg["mip_rate"]) & valid_t
                    idx_m = idx.clone()
                    idx_m[ms] = n_items
                    o2 = m.encode_sequence(idx_m, rbin, tb, pad)
                    b2, t2 = torch.nonzero(ms, as_tuple=True)
                    if len(b2) > 0:
                        z2 = o2["mu"][b2]
                        p2 = tgt[b2, t2]
                        sp = (z2 * V[p2]).sum(-1)
                        sn = torch.einsum("nd,nkd->nk", z2, V[neg[b2, t2]])
                        loss = loss + cfg["lam_mip"] * M.sampled_softmax_rec(sp, sn)

                if "seqcl" in aux and step % max(cfg.get("seqcl_every", 2), 1) == 0:
                    views = []
                    for _ in range(2):
                        ai, ar, at, ap, _ = TE.augment((idx, rbin, tb, pad), pad_id=n_items)
                        o = m.encode_sequence(ai, ar, at, ap)
                        views.append(o["mu"])
                    loss = loss + cfg["lam_seqcl"] * M.seqcl_loss(views[0], views[1], cfg["tau_s"])

                if "cmcl" in aux and m._imgfeat is not None and m._txtfeat is not None:
                    items = torch.unique(tgt[valid_t])
                    if len(items) > 4:
                        _, txt_e, img_e, a_img, a_txt = m._item_parts(items.unsqueeze(0))
                        txt_e, img_e = txt_e[0], img_e[0]
                        if a_img is not None:
                            ok = (a_img[0] > 0.5)
                            if ok.sum() > 4:
                                loss = loss + cfg["lam_cmcl"] * M.itc_loss(
                                    img_e[ok], txt_e[ok], cfg["tau_m"])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), cfg["clip"])
            opt.step()
            tot += float(loss) * len(b)
            nb += len(b)
            step += 1
            if max_seconds and time.time() - t_start > max_seconds:
                break
        sched.step()
        vs = TE.evaluate(m, val_split, ds)
        ndcg = vs["full"]["NDCG@10"]
        hist.append({"epoch": ep + 1, "loss": tot / max(nb, 1), "val_NDCG@10": ndcg,
                     "sec": round(time.time() - t0, 1)})
        if verbose:
            print(f"  [{name}] ep{ep+1:02d} loss={tot/max(nb,1):.4f} valNDCG@10={ndcg:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if ndcg > best + 1e-5:
            best, bad = ndcg, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                if verbose:
                    print(f"  [{name}] early stopping @ep{ep+1}")
                break
        if max_seconds and time.time() - t_start > max_seconds:
            break
    if best_state is not None:
        m.load_state_dict(best_state)
    return m, {"history": hist, "best_val_NDCG@10": best,
               "train_seconds": round(time.time() - t_start, 1),
               "n_params": sum(p.numel() for p in m.parameters() if p.requires_grad)}


def fit_bpr(model, ds, splits, cfg, name="BPR-MF", verbose=True, max_seconds=None):
    """BPR-MF on (user, positive, negative) triples -- non-sequential baseline."""
    import train_eval as TE
    train_split, val_split = splits
    n_items = train_split.n_items
    rows = np.repeat(np.arange(train_split.idx.shape[0]), train_split.idx.shape[1])
    items = train_split.tgt.reshape(-1)
    keep = items >= 0
    rows, items = rows[keep], items[keep]
    seen = {r: set(train_split.idx[r][~train_split.padmask[r]].tolist())
            for r in range(train_split.idx.shape[0])}
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    t_start, best, best_state, bad, hist = time.time(), -1.0, None, 0, []
    for ep in range(cfg["epochs"]):
        perm = np.random.permutation(len(rows))
        t0, tot, nb = time.time(), 0.0, 0
        for s in range(0, len(perm), 2048):
            b = perm[s:s + 2048]
            u = torch.from_numpy(rows[b])
            p = torch.from_numpy(items[b])
            n = np.empty(len(b), dtype=np.int64)
            for i in range(len(b)):
                c = np.random.randint(0, n_items)
                while c in seen[rows[b[i]]]:
                    c = np.random.randint(0, n_items)
                n[i] = c
            n = torch.from_numpy(n)
            eu = model.u.weight[u]
            sp = (eu * model.i.weight[p]).sum(-1)
            sn = (eu * model.i.weight[n]).sum(-1)
            loss = -F.logsigmoid(sp - sn).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
            nb += len(b)
        vs = TE.evaluate(model, val_split, ds)
        ndcg = vs["full"]["NDCG@10"]
        hist.append({"epoch": ep + 1, "loss": tot / max(nb, 1), "val_NDCG@10": ndcg,
                     "sec": round(time.time() - t0, 1)})
        if verbose:
            print(f"  [{name}] ep{ep+1:02d} loss={tot/max(nb,1):.4f} valNDCG@10={ndcg:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if ndcg > best + 1e-5:
            best, bad = ndcg, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, {"history": hist, "best_val_NDCG@10": best,
                   "train_seconds": round(time.time() - t_start, 1),
                   "n_params": sum(p.numel() for p in model.parameters())}
