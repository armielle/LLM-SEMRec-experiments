"""
LLM-SEMRec -- full PyTorch implementation of the architecture in the paper.

Section IV component map (equation -> module):
  Eq. 6-9    collaborative / feedback / temporal embeddings      -> ItemInputs
  Eq. 10-11  frozen SigLIP 2 projection                          -> ItemInputs.img_proj
  Eq. 13-14  frozen Qwen3 semantic projection                    -> ItemInputs.txt_proj
  Eq. 15-20  reliability-aware gated multimodal fusion           -> GatedFusion
  Eq. 29-32  hybrid causal TCN + Transformer + GRU encoder       -> HybridEncoder
  Eq. 33-35  recency-aware temporal attention                    -> TemporalAttention
  Eq. 36-40  variational user preference distribution + KL       -> VariationalPreference
  Eq. 41-52  candidate encoding + uncertainty-aware ranking      -> LLMSEMRec.candidate_vectors / score
  Eq. 24-28  self-supervised losses                              -> module-level loss functions
  Eq. 53-54  mixed negative sampling + sampled softmax           -> losses + train_eval.py
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_TIME = 16          # Eq. 9 : discrete temporal buckets (index 0 reserved for pad / delta = 0)


def time_bucket_edges(n_buckets=N_TIME, lo=60.0, hi=3.15e8):
    return torch.logspace(math.log10(lo), math.log10(hi), n_buckets - 1).float()


def bucket_time(delta_sec, edges):
    """b(.) in Eq. 9 : log-spaced bucketing of the inter-interaction interval."""
    b = torch.zeros_like(delta_sec, dtype=torch.long)
    b[delta_sec > 0] = 1
    for i, e in enumerate(edges):
        b[delta_sec > e] = i + 2
    return b.clamp(0, N_TIME - 1)


# --------------------------------------------------------------------- Eq. 6-14
class ItemInputs(nn.Module):
    def __init__(self, n_items, d, d_img, d_txt, n_rating_bins,
                 use_img=True, use_txt=True):
        super().__init__()
        self.n_items = n_items
        self.d = d
        self.use_img = use_img
        self.use_txt = use_txt
        # Eq. 6 : trainable collaborative identifier embedding (+ PAD / [MASK] rows)
        self.id_emb = nn.Embedding(n_items + 2, d)
        nn.init.normal_(self.id_emb.weight, std=0.02)
        self.PAD_ID = n_items
        self.MASK_ID = n_items + 1
        self.rating_emb = nn.Embedding(n_rating_bins, d)      # Eq. 7
        self.time_emb = nn.Embedding(N_TIME, d)               # Eq. 9
        self.img_proj = nn.Linear(d_img, d)                   # Eq. 11
        self.img_ln = nn.LayerNorm(d)
        self.img_miss = nn.Parameter(torch.zeros(d))          # Sec. IV-D missingness vector
        self.txt_proj = nn.Linear(d_txt, d)                   # Eq. 14
        self.txt_ln = nn.LayerNorm(d)
        self.txt_miss = nn.Parameter(torch.zeros(d))
        self.register_buffer("time_edges", time_bucket_edges())
        self._imgfeat = None
        self._txtfeat = None
        self._imgavail = None

    def item_modalities(self, idx):
        """Item-side modality embeddings + availability indicators (Eq. 15-16)."""
        B, T = idx.shape
        flat = idx.reshape(-1)
        safe = flat.clamp(0, self.n_items - 1)
        is_pad = flat == self.PAD_ID
        is_mask = flat == self.MASK_ID
        idv = self.id_emb(flat)
        a_id = (~is_pad).float()

        n = flat.numel()
        img = torch.zeros(n, self.d, device=idx.device)
        txt = torch.zeros(n, self.d, device=idx.device)
        a_img = torch.zeros(n, device=idx.device)
        a_txt = torch.zeros(n, device=idx.device)

        if self.use_img and self._imgfeat is not None:
            f = self._imgfeat[safe].to(torch.float32)
            ok = torch.ones(n, device=idx.device) if self._imgavail is None \
                else self._imgavail[safe].float()
            ok = ok * (~is_pad).float() * (~is_mask).float()   # masked positions lose the modality
            img = self.img_ln(self.img_proj(f)) * ok.unsqueeze(1) \
                + self.img_miss * (1.0 - ok).unsqueeze(1)
            a_img = ok
        elif self.use_img:
            a_img = (~is_pad).float() * (~is_mask).float()

        if self.use_txt and self._txtfeat is not None:
            g = self._txtfeat[safe].to(torch.float32)
            ok = (~is_pad).float() * (~is_mask).float()
            txt = self.txt_ln(self.txt_proj(g)) * ok.unsqueeze(1) \
                + self.txt_miss * (1.0 - ok).unsqueeze(1)
            a_txt = ok
        elif self.use_txt:
            a_txt = (~is_pad).float() * (~is_mask).float()

        return (idv.view(B, T, self.d), img.view(B, T, self.d), txt.view(B, T, self.d),
                a_id.view(B, T), a_img.view(B, T), a_txt.view(B, T))


# --------------------------------------------------------------------- Eq. 15-20
class GatedFusion(nn.Module):
    """Reliability-aware gated multimodal fusion.

    Modality set M = {id, img, txt, r, time}.  The controller observes both the
    representations and their availability indicators (Eq. 16) so that missing
    modalities cannot be confused with informative zero-valued features.
    """

    N_MOD = 5

    def __init__(self, d, dropout=0.2, use_gating=True):
        super().__init__()
        self.d = d
        self.use_gating = use_gating
        self.Wg = nn.Linear(self.N_MOD * d + self.N_MOD, self.N_MOD, bias=True)  # Eq. 17
        self.Wf = nn.Linear(d, d)                                               # Eq. 20
        self.ln = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, mods, avails, id_residual):
        x = torch.cat(list(mods) + [a.unsqueeze(-1) for a in avails], dim=-1)
        av = torch.stack(avails, dim=-1)
        if self.use_gating:
            o = self.Wg(x)                                       # Eq. 17
            o = o.masked_fill(av < 0.5, -1e9)                    # Eq. 18 masked softmax
            g = torch.softmax(o, dim=-1) * av
            g = g / g.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        else:
            g = av / av.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        h = sum(g[..., i:i + 1] * mods[i] for i in range(self.N_MOD))            # Eq. 19
        h = self.ln(h + id_residual + self.drop(F.gelu(self.Wf(h))))             # Eq. 20
        return h


# --------------------------------------------------------------------- attention
def build_attn_mask(B, T, pad_mask, causal, device):
    """Boolean [B,1,T,T] attention mask; True = allowed.

    The diagonal is always allowed.  Without that, a position that lies entirely
    inside the left padding can only attend to itself under a causal mask, which
    makes its key set empty, and the resulting softmax(-inf) is NaN -- a NaN that
    then contaminates the whole sequence through later attention layers.
    """
    key_ok = (torch.ones(B, T, device=device, dtype=torch.bool) if pad_mask is None
              else ~pad_mask)
    allowed = key_ok.unsqueeze(1).expand(B, T, T).clone()
    if causal:
        allowed &= ~torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    allowed |= torch.eye(T, device=device, dtype=torch.bool).unsqueeze(0)
    return allowed.unsqueeze(1)


class MHASelf(nn.Module):
    """Multi-head self-attention with an explicit boolean mask (NaN-safe)."""

    def __init__(self, d, n_heads, dropout=0.2):
        super().__init__()
        assert d % n_heads == 0
        self.h = n_heads
        self.dh = d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.p = dropout

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.p if self.training else 0.0)
        o = o.transpose(1, 2).reshape(B, T, -1)
        return self.out(o)


class TransformerStack(nn.Module):
    """Pre-norm Transformer encoder stack with optional causal masking (Eq. 31)."""

    def __init__(self, d, n_heads=4, n_layers=2, ff=1024, dropout=0.2):
        super().__init__()
        self.attns = nn.ModuleList([MHASelf(d, n_heads, dropout) for _ in range(n_layers)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.ffs = nn.ModuleList([
            nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
            for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.n_layers = n_layers

    def forward(self, x, mask=None):
        for i in range(self.n_layers):
            x = x + self.drop(self.attns[i](self.ln1[i](x), mask))
            x = x + self.drop(self.ffs[i](self.ln2[i](x)))
        return x


# --------------------------------------------------------------------- Eq. 30
class CausalTCN(nn.Module):
    def __init__(self, d, kernel=3, dilations=(1, 2), dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for dil in dilations:
            self.convs.append(nn.Conv1d(d, d, kernel, dilation=dil, padding=(kernel - 1) * dil))
            self.norms.append(nn.LayerNorm(d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = x.transpose(1, 2)
        T = h.shape[-1]
        for conv, ln in zip(self.convs, self.norms):
            y = conv(h)[..., :T]
            y = self.drop(F.gelu(y))
            h = h + y
            h = ln(h.transpose(1, 2)).transpose(1, 2)
        return h.transpose(1, 2)


# --------------------------------------------------------------------- Eq. 29-32
class HybridEncoder(nn.Module):
    def __init__(self, d, l_max=50, n_heads=4, n_layers=2, ff=1024, dropout=0.2,
                 kernel=3, dilations=(1, 2)):
        super().__init__()
        self.l_max = l_max
        self.tcn = CausalTCN(d, kernel, dilations, dropout)
        self.pos = nn.Embedding(l_max, d)                        # Eq. 31 : learnable P_u
        self.enc = TransformerStack(d, n_heads, n_layers, ff, dropout)
        self.gru = nn.GRU(d, d, batch_first=True)                # Eq. 32

    def forward(self, h, pad_mask=None):
        B, T, d = h.shape
        c = self.tcn(h)                                          # Eq. 30
        pos = self.pos(torch.arange(T, device=h.device)).unsqueeze(0)
        # Eq. 31: causal mask -- the representation at t depends only on t' <= t
        mask = build_attn_mask(B, T, pad_mask, causal=True, device=h.device)
        g = self.enc(c + pos, mask)
        q, _ = self.gru(g)                                       # Eq. 32
        return c, g, q


# --------------------------------------------------------------------- Eq. 33-35
class TemporalAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Wa = nn.Linear(d, d, bias=False)
        self.Wd = nn.Linear(d, d, bias=False)
        self.wa = nn.Linear(d, 1, bias=False)
        self.ba = nn.Parameter(torch.zeros(d))

    def forward(self, q, time_emb, valid):
        e = self.wa(torch.tanh(self.Wa(q) + self.Wd(time_emb) + self.ba)).squeeze(-1)  # Eq. 33
        e = e.masked_fill(~valid, -1e9)
        a = torch.softmax(e, dim=-1)                                                   # Eq. 34
        a = a * valid.float()
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        c = (a.unsqueeze(-1) * q).sum(dim=1)                                           # Eq. 35
        return c, a


# --------------------------------------------------------------------- Eq. 36-40
class VariationalPreference(nn.Module):
    def __init__(self, d, dz):
        super().__init__()
        self.mu = nn.Linear(d, dz)             # Eq. 37
        self.logsig2 = nn.Linear(d, dz)        # Eq. 38

    def forward(self, c, sample=True):
        mu = self.mu(c)
        logsig2 = self.logsig2(c).clamp(-10.0, 6.0)
        sig = torch.exp(0.5 * logsig2)
        z = mu + sig * torch.randn_like(sig) if sample else mu       # Eq. 39
        kl = 0.5 * (mu.pow(2) + sig.pow(2) - logsig2 - 1.0).sum(-1)  # Eq. 40
        return mu, sig, z, kl


# --------------------------------------------------------------------- main model
class LLMSEMRec(nn.Module):
    def __init__(self, cfg, n_items, n_rating_bins, d_img, d_txt,
                 has_img=None, has_txt=None, mean_rating_bin=None):
        super().__init__()
        d = cfg["d"]
        self.cfg = cfg
        self.d = d
        self.dz = cfg["dz"]
        self.l_max = cfg["l_max"]
        self.n_items = n_items
        self.inputs = ItemInputs(n_items, d, d_img, d_txt, n_rating_bins,
                                 use_img=cfg.get("use_img", True),
                                 use_txt=cfg.get("use_txt", True))
        self.fusion = GatedFusion(d, dropout=cfg["dropout"],
                                  use_gating=cfg.get("gated_fusion", True))
        self.encoder = HybridEncoder(d, cfg["l_max"], n_heads=cfg["n_heads"],
                                     n_layers=cfg["n_tf_layers"], ff=cfg["ff"],
                                     dropout=cfg["dropout"],
                                     dilations=cfg.get("dilations", (1, 2)))
        self.attn = TemporalAttention(d)
        if cfg.get("use_vae", True):
            self.vae = VariationalPreference(d, self.dz)
        else:
            self.vae = None
            self.det_proj = nn.Linear(d, self.dz)
        self.Wc = nn.Linear(d, self.dz)                 # Eq. 46
        self.Wmip = nn.Linear(d, self.dz)               # Eq. 24 head
        if has_img is not None:
            self.register_buffer("img_avail", has_img.float())
        if has_txt is not None:
            self.register_buffer("txt_avail", has_txt.float())
        if mean_rating_bin is not None:
            self.register_buffer("mean_rbin", torch.as_tensor(
                np.asarray(mean_rating_bin), dtype=torch.long))
        else:
            self.register_buffer("mean_rbin", torch.zeros(n_items, dtype=torch.long))
        self._cand_cache = None

    # ------------------------------------------------------------------ features
    def set_item_features(self, img_feat=None, txt_feat=None, img_avail=None):
        if img_feat is not None:
            self.inputs._imgfeat = img_feat
        if txt_feat is not None:
            self.inputs._txtfeat = txt_feat
        if img_avail is not None:
            self.inputs._imgavail = img_avail
        self.invalidate_candidates()

    def invalidate_candidates(self):
        self._cand_cache = None

    # ------------------------------------------------------------------ Eq. 15-20 on a sequence
    def fuse_sequence(self, idx, rbin, tbucket, mod_mask=None):
        idv, img, txt, a_id, a_img, a_txt = self.inputs.item_modalities(idx)
        if mod_mask is not None:
            # modality dropout 0.1 for image / text features (Sec. IV-H, view (iii))
            a_img = a_img * mod_mask[..., 0]
            a_txt = a_txt * mod_mask[..., 1]
            img = img * mod_mask[..., 0:1]
            txt = txt * mod_mask[..., 1:2]
        r = self.inputs.rating_emb(rbin)
        t = self.inputs.time_emb(tbucket)
        return self.fusion((idv, img, txt, r, t), (a_id, a_img, a_txt, a_id, a_id), idv)

    # ------------------------------------------------------------------ Eq. 45-46
    def candidate_vectors(self, use_cache=True):
        if use_cache and self._cand_cache is not None:
            return self._cand_cache
        dev = next(self.parameters()).device
        idx = torch.arange(self.n_items, device=dev).unsqueeze(0)         # [1,n]
        idv, img, txt, a_id, a_img, a_txt = self.inputs.item_modalities(idx)
        r = self.inputs.rating_emb(self.mean_rbin.to(dev)).unsqueeze(0)
        t = self.inputs.time_emb(torch.ones_like(idx))       # neutral recency bucket
        h = self.fusion((idv[0], img[0], txt[0], r[0], t[0]),
                        (a_id[0], a_img[0], a_txt[0], a_id[0], a_id[0]), idv[0])
        V = self.Wc(h)
        if use_cache and not self.training:
            self._cand_cache = V
        return V

    # ------------------------------------------------------------------ forward
    def encode_sequence(self, idx, rbin, tbucket, pad_mask, mod_mask=None):
        """Encode a batch of histories.

        Because TCN / Transformer / GRU are all causal, the prefix-pooled user
        representation is obtained for every position with a cumulative sum of
        the (unnormalised) temporal-attention weights of Eq. 34.  c[:, k] is
        therefore Eq. 35 restricted to the prefix 1..k, which allows one forward
        pass to supervise every position.  c[:, -1] is the full-history pooling.
        """
        h = self.fuse_sequence(idx, rbin, tbucket, mod_mask=mod_mask)
        c_tcn, g_tf, q = self.encoder(h, pad_mask=pad_mask)
        te = self.inputs.time_emb(tbucket)
        valid = ~pad_mask
        e = self.attn.wa(torch.tanh(self.attn.Wa(q) + self.attn.Wd(te) + self.attn.ba)).squeeze(-1)
        e = e.masked_fill(~valid, -1e9)
        w = torch.softmax(e, dim=-1) * valid.float()                      # Eq. 34 weights
        cw = torch.cumsum(w, dim=1).clamp_min(1e-9)
        num = torch.cumsum(w.unsqueeze(-1) * q, dim=1)
        cu_all = num / cw.unsqueeze(-1)                                   # [B,L,d] prefix poolings
        if self.vae is not None:
            mu, sig, z, kl = self.vae(cu_all, sample=self.training)
        else:
            mu = self.det_proj(cu_all)
            sig = torch.ones_like(mu)
            z = mu
            kl = torch.zeros(mu.shape[:2], device=idx.device)
        return {"mu": mu, "sig": sig, "z": z, "kl": kl, "cu": cu_all, "q": q,
                "alpha": w, "c": c_tcn, "g": g_tf, "h": h}

    def forward(self, idx, rbin, tbucket, pad_mask):
        return self.encode_sequence(idx, rbin, tbucket, pad_mask)

    # ------------------------------------------------------------------ Eq. 42-44
    @staticmethod
    def score(mu, sig, V, gamma=0.0):
        s_mean = mu @ V.t()                                    # Eq. 42
        if gamma == 0.0:
            return s_mean
        var = sig.pow(2).matmul(V.t().pow(2))                  # Eq. 43 (diagonal posterior)
        return s_mean - gamma * torch.sqrt(var.clamp_min(1e-12))   # Eq. 44


# --------------------------------------------------------------------- Eq. 24 : MIP
def mip_loss(model, q_seq, candV, targets, mask_pos, neg_items):
    h = q_seq[mask_pos]
    h = model.Wmip(h) if hasattr(model, "Wmip") else model.det_proj(h)
    pos = (h * candV[targets]).sum(-1)
    neg = torch.einsum("md,mkd->mk", h, candV[neg_items])
    logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
    y = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, y)


# --------------------------------------------------------------------- Eq. 25
def seqcl_loss(z1, z2, tau=0.07):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / tau
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# --------------------------------------------------------------------- Eq. 26
def itc_loss(v, l, tau=0.07):
    v = F.normalize(v, dim=-1)
    l = F.normalize(l, dim=-1)
    logits = v @ l.t() / tau
    labels = torch.arange(v.shape[0], device=v.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# --------------------------------------------------------------------- Eq. 27
def isc_loss(e_id, l, tau=0.07):
    e = F.normalize(e_id, dim=-1)
    l = F.normalize(l, dim=-1)
    logits = e @ l.t() / tau
    labels = torch.arange(e.shape[0], device=e.device)
    return F.cross_entropy(logits, labels)


# --------------------------------------------------------------------- Eq. 54
def sampled_softmax_rec(s_pos, s_neg):
    logits = torch.cat([s_pos.unsqueeze(1), s_neg], dim=1)
    y = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, y)
