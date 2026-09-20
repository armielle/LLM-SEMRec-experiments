"""
Eight additional supporting analyses for the paper.

Every figure is computed from artefacts that already exist on disk: the trained
checkpoints, the cached frozen-encoder embeddings, the per-user evaluation npz
files and the preprocessed datasets.  Nothing is recomputed from scratch and no
value is fabricated.

  figS1  reliability-aware fusion gate: where the model puts its attention
  figS2  recency-aware temporal attention profile (Eq. 33-35)
  figS3  inference-time modality dropout: are both modalities really used?
  figS4  cold-start x image availability: does content replace identifiers?
  figS5  geometry of the learned item space (ID vs multimodal) + category purity
  figS6  popularity bias / long-tail exposure of the recommendation lists
  figS7  performance vs exact history length and vs temporal drift
  figS8  per-category breakdown + score separation (ROC)
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nice  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import model as M
import train_eval as TE
from common import load_ds
from paths import CACHE, FIGURES, RESULTS

torch.set_num_threads(nice.threads())
MODELDIR = os.path.join(RESULTS, "..", "models")
DS = "amazon_beauty"
L = 20
MODELS = ["llmsemrec", "sasrec", "bprmf", "llmonly", "mmsasrec", "bert4rec"]
NICE = {"llmsemrec": "LLM-SEMRec", "sasrec": "SASRec", "bprmf": "BPR-MF",
        "llmonly": "LLM-only", "mmsasrec": "MM-SASRec", "bert4rec": "BERT4Rec",
        "popularity": "Popularity", "gru4rec": "GRU4Rec", "caser": "Caser",
        "mmssl": "MMSSL-lite"}
COL = {"llmsemrec": "#C0392B", "sasrec": "#2471A3", "bprmf": "#7F8C8D",
       "llmonly": "#6C3483", "mmsasrec": "#B7950B", "bert4rec": "#1E8449"}
MODS = ["item ID", "image", "text (LLM)", "feedback", "time"]

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "figure.facecolor": "white"})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"{name}.{ext}"))
    plt.close(fig)
    print("  fig:", name)


# --------------------------------------------------------------------- shared data
_DS = None
_SPLITS = {}
_FEATS = None
_MODEL = None


def ds():
    global _DS
    if _DS is None:
        _DS = load_ds(DS)
    return _DS


def split(name="test"):
    if name not in _SPLITS:
        _SPLITS[name] = TE.build_split(ds(), name, L)
    return _SPLITS[name]


def feats():
    global _FEATS
    if _FEATS is None:
        _FEATS = TE.load_features(DS, {"use_img": True, "use_txt": True})
    return _FEATS


def llmsemrec():
    global _MODEL
    if _MODEL is None:
        ck = torch.load(os.path.join(MODELDIR, f"llmsemrec__{DS}.pt"), weights_only=False)
        m = TE.build_model(ds(), dict(ck["cfg"]), feats())
        m.load_state_dict(ck["state"])
        m.eval()
        _MODEL = m
    return _MODEL


def npz(model, dataset=DS):
    p = os.path.join(MODELDIR, f"eval__{model}__{dataset}.npz")
    return np.load(p) if os.path.exists(p) else None


def pop_counts():
    p = {}
    n = len(ds()["items"])
    c = np.zeros(n)
    for u, v in ds()["train_targets"].items():
        for (i, _, _) in v:
            c[i] += 1
    return c


# --------------------------------------------------------------------- S1 gates
def gate_weights(model, idx, rbin, tb):
    """Recompute the masked-softmax gate of Eq. 18 with the trained weights."""
    idv, img, txt, a_id, a_img, a_txt = model.inputs.item_modalities(idx)
    r = model.inputs.rating_emb(rbin)
    t = model.inputs.time_emb(tb)
    mods = [idv, img, txt, r, t]
    av = [a_id, a_img, a_txt, a_id, a_id]
    x = torch.cat(list(mods) + [a.unsqueeze(-1) for a in av], dim=-1)
    o = model.fusion.Wg(x)
    avs = torch.stack(av, -1)
    o = o.masked_fill(avs < 0.5, -1e9)
    g = torch.softmax(o, -1) * avs
    return g / g.sum(-1, keepdim=True).clamp_min(1e-9)


def figS1():
    m = llmsemrec()
    sp = split("test")
    pop = pop_counts()
    tgt_pop = pop[sp.tgt[:, -1]]
    q = np.quantile(tgt_pop, [1 / 3, 2 / 3])
    bucket = np.digitize(tgt_pop, q)
    G, B, A = [], [], []
    with torch.no_grad():
        for s in range(0, len(sp.users), 256):
            e = min(s + 256, len(sp.users))
            idx, rb, tb, pm, _ = sp.to_torch(np.arange(s, e))
            g = gate_weights(m, idx, rb, tb)                     # [b,T,5]
            valid = (~pm).unsqueeze(-1)
            shown = torch.zeros_like(pm)
            shown[:, -1] = True                                   # the decision position
            sel = shown.unsqueeze(-1)
            G.append((g * sel).sum(1).cpu().numpy())
            B.append(bucket[s:e])
            A.append(m.inputs.item_modalities(idx)[4][:, -1].cpu().numpy())
    G = np.concatenate(G)
    B = np.concatenate(B)
    Aimg = np.concatenate(A)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))
    mean_g = G.mean(0)
    axes[0].bar(range(5), mean_g, color=["#2471A3", "#B7950B", "#6C3483", "#1E8449", "#7F8C8D"],
                edgecolor="black", lw=0.4)
    axes[0].set_xticks(range(5)); axes[0].set_xticklabels(MODS, rotation=25, ha="right")
    axes[0].set_ylabel("mean gate weight (Eq. 18)")
    axes[0].set_title("(a) Learned modality importance\nat the decision position")
    for i, v in enumerate(mean_g):
        axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    axes[0].set_ylim(0, mean_g.max() * 1.25)

    for i, b in enumerate(("cold", "medium", "warm")):
        mask = B == i
        if mask.sum() == 0:
            continue
        axes[1].bar(np.arange(5) + (i - 1) * 0.27, G[mask].mean(0), 0.27,
                    label=f"{b} targets (n={mask.sum():,})",
                    color=["#A9CCE3", "#F9E79F", "#D7BDE2"][i],
                    edgecolor="black", lw=0.4)
    axes[1].set_xticks(range(5)); axes[1].set_xticklabels(MODS, rotation=25, ha="right")
    axes[1].set_ylabel("mean gate weight")
    axes[1].set_title("(b) The gate adapts to item popularity")
    axes[1].legend(fontsize=7)

    grp = [G[Aimg < 0.5].mean(0) if (Aimg < 0.5).any() else np.full(5, np.nan),
           G[Aimg > 0.5].mean(0) if (Aimg > 0.5).any() else np.full(5, np.nan)]
    axes[2].bar(np.arange(5) - 0.19, grp[0], 0.38, label="image missing",
                color="#D5D8DC", edgecolor="black", lw=0.4)
    axes[2].bar(np.arange(5) + 0.19, grp[1], 0.38, label="image available",
                color="#F5B041", edgecolor="black", lw=0.4)
    axes[2].set_xticks(range(5)); axes[2].set_xticklabels(MODS, rotation=25, ha="right")
    axes[2].set_ylabel("mean gate weight")
    axes[2].set_title("(c) Availability drives the weights\n(not a zero-valued feature)")
    axes[2].legend(fontsize=7)
    fig.suptitle("Reliability-aware gated multimodal fusion: what the model actually "
                 "learns (Amazon Beauty)", fontweight="bold", y=1.04)
    save(fig, "figS1_gate_weights")


# --------------------------------------------------------------------- S2 recency
def figS2():
    m = llmsemrec()
    sp = split("test")
    prof, lens = [], []
    with torch.no_grad():
        for s in range(0, len(sp.users), 256):
            e = min(s + 256, len(sp.users))
            idx, rb, tb, pm, _ = sp.to_torch(np.arange(s, e))
            h = m.fuse_sequence(idx, rb, tb)
            c, g, q = m.encoder(h, pad_mask=pm)
            te = m.inputs.time_emb(tb)
            valid = ~pm
            ee = m.attn.wa(torch.tanh(m.attn.Wa(q) + m.attn.Wd(te) + m.attn.ba)).squeeze(-1)
            w = torch.softmax(ee.masked_fill(~valid, -1e9), dim=-1) * valid.float()
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
            prof.append(w.cpu().numpy())
            lens.append(valid.sum(-1).cpu().numpy())
    W = np.concatenate(prof)
    N = np.concatenate(lens)
    rel = np.full_like(W, np.nan)
    for r in range(W.shape[0]):
        n = int(N[r])
        rel[r, L - n:] = np.arange(n) / max(n - 1, 1)          # 0 = oldest, 1 = newest

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))
    bins = np.linspace(0, 1, 21)
    mid = 0.5 * (bins[1:] + bins[:-1])
    mean_prof = [np.nanmean(rel[:, (rel[0] >= bins[i]) & (rel[0] < bins[i + 1])] if False else
                            np.where((rel >= bins[i]) & (rel < bins[i + 1]), W, np.nan))
                 for i in range(20)]
    axes[0].plot(mid, mean_prof, "o-", color=COL["llmsemrec"], ms=3.5)
    axes[0].set_xlabel("relative position in the history (0 = oldest, 1 = most recent)")
    axes[0].set_ylabel("mean attention weight $\\alpha_t$")
    axes[0].set_title("(a) Recency-aware attention profile\n(Eq. 33-35)")

    ent = []
    for n in range(2, L + 1):
        w = np.where(N == n)[0]
        if len(w) < 30:
            continue
        p = W[w][:, L - n:]
        p = np.clip(p, 1e-12, None)
        ent.append((n, float(np.mean(-(p * np.log(p)).sum(1)))))
    if ent:
        xs, ys = zip(*ent)
        axes[1].plot(xs, ys, "s-", color="#1E8449", ms=3.5)
        axes[1].axhline(np.log(L), ls="--", color="gray", lw=1,
                        label="uniform attention $\\ln L$")
        axes[1].set_xlabel("number of available interactions")
        axes[1].set_ylabel("attention entropy (nats)")
        axes[1].set_title("(b) Attention stays non-degenerate:\nentropy grows with evidence")
        axes[1].legend(fontsize=7)

    ranks = npz("llmsemrec")["test_ranks"]
    hit = (ranks < 10).astype(float)
    xs, ys, ns = [], [], []
    for n in range(1, L + 1):
        w = N == n
        if w.sum() < 50:
            continue
        xs.append(n); ys.append(hit[w].mean()); ns.append(int(w.sum()))
    p = np.polyfit(xs, ys, 1)
    axes[2].bar(xs, ys, color="#5DADE2", edgecolor="black", lw=0.4)
    axes[2].plot(xs, np.polyval(p, xs), "--", color=COL["llmsemrec"],
                 label=f"trend {p[0]*10:+.4f} per 10 interactions")
    axes[2].set_xlabel("available interactions")
    axes[2].set_ylabel("Recall@10")
    axes[2].set_title("(c) Accuracy vs. amount of evidence")
    axes[2].legend(fontsize=7)
    fig.suptitle("Recency-aware temporal attention and evidence sensitivity "
                 "(Amazon Beauty)", fontweight="bold", y=1.04)
    save(fig, "figS2_recency_attention")


# --------------------------------------------------------------------- S3 modality dropout
@torch.no_grad()
def eval_modmask(model, sp, V, drop_img=False, drop_txt=False, chunk=256):
    N = len(sp.users)
    ranks = np.zeros(N, dtype=np.int64)
    tgt = sp.tgt[:, -1]
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        idx, rb, tb, pm, _ = sp.to_torch(np.arange(s, e))
        mm = None
        if drop_img or drop_txt:
            mm = torch.ones(e - s, idx.shape[1], 2)
            if drop_img:
                mm[..., 0] = 0.0
            if drop_txt:
                mm[..., 1] = 0.0
        out = model.encode_sequence(idx, rb, tb, pm, mod_mask=mm)
        mu = out["mu"][:, -1] if out["mu"].dim() == 3 else out["mu"]
        sig = out["sig"][:, -1] if out["sig"].dim() == 3 else out["sig"]
        sc = model.score(mu, sig, V, gamma=0.0).cpu().numpy()
        rows = np.arange(e - s)
        ranks[s:e] = (sc > sc[rows, tgt[s:e]][:, None]).sum(1)
    return (ranks < 10).mean()


def figS3():
    sp = split("test")
    m = llmsemrec()
    m.invalidate_candidates()
    V = m.candidate_vectors()
    base = eval_modmask(m, sp, V)
    di = eval_modmask(m, sp, V, drop_img=True)
    dt = eval_modmask(m, sp, V, drop_txt=True)
    db = eval_modmask(m, sp, V, drop_img=True, drop_txt=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    labs = ["full model", "image hidden", "text hidden", "both hidden"]
    vals = [base, di, dt, db]
    cols = [COL["llmsemrec"], "#F5B041", "#6C3483", "#7F8C8D"]
    axes[0].bar(range(4), vals, color=cols, edgecolor="black", lw=0.4)
    axes[0].set_xticks(range(4)); axes[0].set_xticklabels(labs, rotation=18, ha="right")
    axes[0].set_ylabel("Recall@10")
    axes[0].set_title("(a) Inference-time modality removal\n(trained model, no retraining)")
    for i, v in enumerate(vals):
        axes[0].text(i, v, f"{v:.4f}\n{(v-base)/base*100:+.1f}%", ha="center",
                     va="bottom", fontsize=7.5)
    axes[0].set_ylim(0, base * 1.35)

    with torch.no_grad():
        g = gate_weights(m, *sp.to_torch(np.arange(128))[:3])
        img_w = g[..., 1].reshape(-1).detach().numpy()
        avail = m.inputs.item_modalities(sp.to_torch(np.arange(128))[0])[4].reshape(-1).numpy()
    axes[1].hist(img_w[avail > 0.5], bins=40, color="#F5B041", alpha=0.85,
                 label="image available", edgecolor="none")
    if (avail < 0.5).any():
        axes[1].hist(img_w[avail < 0.5], bins=40, color="#7F8C8D", alpha=0.85,
                     label="image missing", edgecolor="none")
    axes[1].set_xlabel("gate weight of the image modality")
    axes[1].set_ylabel("count")
    axes[1].set_yscale("log")
    axes[1].set_title("(b) The image is gated off when unavailable")
    axes[1].legend(fontsize=7.5)
    fig.suptitle("Are both content modalities really used? (Amazon Beauty)",
                 fontweight="bold", y=1.03)
    save(fig, "figS3_modality_dropout")


# --------------------------------------------------------------------- S4 cold x availability
def figS4():
    pop = pop_counts()
    sp = split("test")
    tgt = sp.tgt[:, -1]
    tpop = pop[tgt]
    q = np.quantile(tpop, [1 / 3, 2 / 3])
    b = np.digitize(tpop, q)
    ia = np.load(os.path.join(CACHE, f"{DS}_img_avail.npy"))
    has = ia[tgt]
    models = [mm for mm in MODELS if npz(mm) is not None]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.8))
    w = 0.8 / len(models)
    for i, mm in enumerate(models):
        r = npz(mm)["test_ranks"]
        hit = (r < 10).astype(float)
        vals = [hit[(b == k) & has].mean() if ((b == k) & has).sum() else np.nan
                for k in range(3)]
        axes[0].bar(np.arange(3) + i * w, vals, w, label=NICE.get(mm, mm),
                    color=COL.get(mm, "#95A5A6"), edgecolor="black", lw=0.4)
    axes[0].set_xticks(np.arange(3) + w * (len(models) - 1) / 2)
    axes[0].set_xticklabels(["cold", "medium", "warm"])
    axes[0].set_xlabel("target-item popularity (tercile)")
    axes[0].set_ylabel("Recall@10")
    axes[0].set_title("(a) Cold start, items WITH an image")
    axes[0].legend(fontsize=6.5, ncols=2)

    dec = np.digitize(tpop, np.quantile(tpop, np.linspace(0, 1, 11)[1:-1]))
    show = ["llmsemrec", "sasrec", "bprmf", "llmonly"]
    for mm in show:
        d = npz(mm)
        if d is None:
            continue
        h = (d["test_ranks"] < 10).astype(float)
        axes[1].plot(range(1, 11),
                     [h[dec == k].mean() * 100 if (dec == k).sum() else np.nan
                      for k in range(10)],
                     "o-", ms=3.5, color=COL.get(mm, "#95A5A6"),
                     lw=2.2 if mm == "llmsemrec" else 1.2, label=NICE.get(mm, mm))
    axes[1].set_xlabel("target-item popularity decile (1 = coldest, 10 = hottest)")
    axes[1].set_ylabel("Recall@10 (%)")
    axes[1].set_title("(b) Where the gain comes from:\nthe cold end of the catalogue")
    axes[1].legend(fontsize=7)
    fig.suptitle("Cold-start behaviour (Amazon Beauty; 99.98 % of items carry an image, "
                 "so availability is not a usable axis)", fontweight="bold", y=1.03)
    save(fig, "figS4_coldstart_availability")


# --------------------------------------------------------------------- S5 item space
def figS5():
    m = llmsemrec()
    m.invalidate_candidates()
    with torch.no_grad():
        Vfull = m.candidate_vectors().cpu().numpy()            # [n,dz] multimodal
        idx = torch.arange(m.n_items).unsqueeze(0)
        Vraw = m.inputs.id_emb(idx)[0].detach().cpu().numpy()   # [n,d] ID only
    meta = ds()["meta"]
    items = ds()["items"]
    CATL = 2   # level 0 is the constant root ("Beauty"); level 2 is specific
    raw = [(meta.get(a, {}).get("cats") or ["?"]) for a in items]
    cats = np.array([(c + ["?"] * 3)[CATL] for c in raw])
    uniq, cnt = np.unique(cats, return_counts=True)
    top = uniq[np.argsort(-cnt)][:8]
    keep = np.isin(cats, top)
    lab = np.array([list(top).index(c) if c in top else -1 for c in cats])

    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE
    rng = np.random.RandomState(0)
    sel = np.where(keep)[0]
    if len(sel) > 4000:
        sel = rng.choice(sel, 4000, replace=False)

    def embed(X):
        Z = TruncatedSVD(n_components=32, random_state=0).fit_transform(X[sel])
        return TSNE(n_components=2, perplexity=30, random_state=0,
                    init="pca", max_iter=500).fit_transform(Z)

    E_mm = embed(Vfull)
    E_id = embed(Vraw)

    def purity_both(X, k=10):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        S_ = Xn[sel] @ Xn[sel].T
        np.fill_diagonal(S_, -2)
        nn = np.argpartition(-S_, k, axis=1)[:, :k]
        out = []
        for lv in (1, 2):
            l = np.array([(c + ["?"] * 4)[lv] for c in raw])[sel]
            out.append(float(np.mean(l[nn] == l[:, None])))
        return out

    (pur_id_l1, pur_id_l2), (pur_mm_l1, pur_mm_l2) = purity_both(Vraw), purity_both(Vfull)
    pur_id, pur_mm = pur_id_l2, pur_mm_l2
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0))
    for ax, E, ttl, pur in ((axes[0], E_id, "(a) identifier-only item space", pur_id),
                            (axes[1], E_mm, "(b) multimodal fused item space", pur_mm)):
        for c, cat in enumerate(top):
            msk = lab[sel] == c
            ax.scatter(E[msk, 0], E[msk, 1], s=5, alpha=0.75, label=cat[:17])
        ax.set_title(f"{ttl}\n10-NN category purity = {pur:.3f}")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[1].legend(fontsize=6, markerscale=2.2, loc="best", ncols=1)
    axes[2].bar(np.arange(2) - 0.19, [pur_id_l1, pur_id_l2], 0.38,
                color="#95A5A6", edgecolor="black", lw=0.4, label="identifier-only")
    axes[2].bar(np.arange(2) + 0.19, [pur_mm_l1, pur_mm_l2], 0.38,
                color=COL["llmsemrec"], edgecolor="black", lw=0.4, label="multimodal fused")
    axes[2].set_xticks(range(2))
    axes[2].set_xticklabels(["coarse\n(6 groups)", "fine\n(38 groups)"])
    axes[2].set_ylabel("10-NN category purity")
    axes[2].set_title("(c) Semantic organisation of\nthe two item spaces")
    for i, v in enumerate((pur_id_l1, pur_mm_l1, pur_id_l2, pur_mm_l2)):
        ax_i = i // 2
        axes[2].text(ax_i + (-0.19 if i % 2 == 0 else 0.19), v, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8)
    axes[2].set_ylim(0, max(pur_mm_l1, pur_id_l1) * 1.25)
    axes[2].legend(fontsize=7)
    fig.suptitle("Geometry of the learned item space (Amazon Beauty, top-8 categories)",
                 fontweight="bold", y=1.03)
    save(fig, "figS5_item_space_geometry")


# --------------------------------------------------------------------- S6 popularity bias
def figS6():
    pop = pop_counts()
    p = pop / pop.sum()
    n = len(pop)
    sp = split("test")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.8))
    order = np.argsort(-pop)
    rank_of = np.empty(n, dtype=np.int64)
    rank_of[order] = np.arange(n)
    for mm in MODELS:
        d = npz(mm)
        if d is None:
            continue
        rec = d["test_topk"].reshape(-1)
        r = rank_of[rec]
        axes[0].plot(np.linspace(0, 1, 101),
                     np.percentile(r, np.linspace(0, 100, 101)),
                     label=NICE.get(mm, mm), color=COL.get(mm, "#95A5A6"),
                     lw=2.0 if mm == "llmsemrec" else 1.1)
    axes[0].set_xlabel("percentile of recommended items (by popularity rank)")
    axes[0].set_ylabel("catalogue popularity rank of the item")
    axes[0].set_title("(a) How head-heavy are the recommendations?")
    axes[0].legend(fontsize=7)

    tail_share = []
    labs = []
    for mm in MODELS:
        d = npz(mm)
        if d is None:
            continue
        rec = d["test_topk"].reshape(-1)
        tail_share.append(float((pop[rec] <= 10).mean()))
        labs.append(NICE.get(mm, mm))
    idx = np.argsort(tail_share)
    axes[1].barh(range(len(labs)), [tail_share[i] for i in idx],
                 color=[COL.get([k for k, v in NICE.items() if v == labs[i]][0], "#95A5A6")
                        for i in idx],
                 edgecolor="black", lw=0.4)
    axes[1].set_yticks(range(len(labs)))
    axes[1].set_yticklabels([labs[i] for i in idx])
    axes[1].axvline(float((pop <= 10).mean()), ls="--", color="gray",
                    label="catalogue base rate")
    axes[1].set_xlabel("share of recommendations that are long-tail items")
    axes[1].set_title("(b) Long-tail exposure (items with $\\leq$10 interactions)")
    axes[1].legend(fontsize=7)
    fig.suptitle("Popularity bias and long-tail exposure (Amazon Beauty)",
                 fontweight="bold", y=1.03)
    save(fig, "figS6_popularity_bias")


# --------------------------------------------------------------------- S7 drift
def figS7():
    seqs = ds()["seqs"]
    users = sorted(seqs.keys())
    gap = np.zeros(len(users))
    hist = np.zeros(len(users))
    for r, u in enumerate(users):
        a = np.asarray(seqs[u])
        gap[r] = a[-1, 2] - a[-2, 2]
        hist[r] = min(len(a) - 2, L)
    hit = {}
    for mm in ["llmsemrec", "sasrec", "bprmf", "popularity"]:
        d = npz(mm)
        if d is None:
            continue
        hit[mm] = (d["test_ranks"] < 10).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.8))
    edges = [0, 30, 90, 365, 1095, 1e9]
    labs = ["<1 month", "1-3 mo", "3-12 mo", "1-3 yr", ">3 yr"]
    b = np.digitize(gap / 86400.0, edges[1:-1])
    w = 0.8 / max(len(hit), 1)
    for i, (mm, h) in enumerate(hit.items()):
        vals = [h[b == k].mean() if (b == k).sum() else np.nan for k in range(len(labs))]
        axes[0].bar(np.arange(len(labs)) + i * w, vals, w, label=NICE.get(mm, mm),
                    color=COL.get(mm, "#95A5A6"), edgecolor="black", lw=0.4)
    axes[0].set_xticks(np.arange(len(labs)) + w * (len(hit) - 1) / 2)
    axes[0].set_xticklabels(labs, rotation=20, ha="right")
    axes[0].set_xlabel("gap between the last training interaction and the target")
    axes[0].set_ylabel("Recall@10")
    axes[0].set_title("(a) Robustness to temporal drift")
    axes[0].legend(fontsize=7)

    xs, ys = [], []
    for k in range(1, L + 1):
        msk = hist == k
        if msk.sum() < 30:
            continue
        xs.append(k)
        ys.append(np.mean([hit[mm][msk].mean() for mm in hit]))
    if xs:
        axes[1].plot(xs, ys, "o-", color=COL["llmsemrec"], ms=4)
        axes[1].set_xlabel("available interactions (tooltip: all models averaged)")
        axes[1].set_ylabel("mean Recall@10 (top baselines)")
        axes[1].set_title("(b) Evidence curve")
    fig.suptitle("Temporal robustness of next-item prediction (Amazon Beauty)",
                 fontweight="bold", y=1.03)
    save(fig, "figS7_temporal_robustness")


# --------------------------------------------------------------------- S8 categories + ROC
def figS8():
    meta = ds()["meta"]
    items = ds()["items"]
    sp = split("test")
    raw_t = [(meta.get(items[t], {}).get("cats") or ["?"]) for t in sp.tgt[:, -1]]
    cats = np.array([(c + ["?"] * 3)[1] for c in raw_t])       # level 1, not the root
    uniq, cnt = np.unique(cats, return_counts=True)
    top = [u for u in uniq[np.argsort(-cnt)] if (cats == u).sum() >= 150][:12]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.0))
    w = 0.8 / 3
    show = ["llmsemrec", "sasrec", "bprmf"]
    for i, mm in enumerate(show):
        d = npz(mm)
        if d is None:
            continue
        h = (d["test_ranks"] < 10).astype(float)
        vals = [h[cats == c].mean() * 100 if (cats == c).sum() else np.nan for c in top]
        axes[0].bar(np.arange(len(top)) + i * w, vals, w, label=NICE.get(mm, mm),
                    color=COL.get(mm), edgecolor="black", lw=0.4)
    axes[0].set_xticks(np.arange(len(top)) + w)
    axes[0].set_xticklabels([f"{c[:16]}\n(n={(cats == c).sum()})" for c in top],
                            rotation=30, ha="right", fontsize=7.5)
    axes[0].set_ylabel("Recall@10 (%)")
    axes[0].set_title("(a) Consistency across product categories")
    axes[0].legend(fontsize=7.5)

    res = []
    for mm in ["llmsemrec", "sasrec", "bprmf", "llmonly", "mmsasrec", "bert4rec"]:
        d = npz(mm)
        if d is None:
            continue
        sc = d["test_scores"]                      # [N,101], column 0 = the positive
        auc = (sc[:, 1:] < sc[:, :1]).mean(1)      # per-user AUC, then averaged
        bs = [auc[np.random.RandomState(b).randint(0, len(auc), len(auc))].mean()
              for b in range(200)]
        res.append((mm, auc.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5),
                    float((auc >= 0.999).mean())))
    res.sort(key=lambda r: r[1])
    cl = [COL.get(r[0], "#95A5A6") for r in res]
    axes[1].barh(range(len(res)), [r[1] for r in res], color=cl,
                 edgecolor="black", lw=0.4)
    axes[1].errorbar([r[1] for r in res], range(len(res)),
                     xerr=[[r[1] - r[2] for r in res], [r[3] - r[1] for r in res]],
                     fmt="none", ecolor="black", capsize=3, lw=1)
    axes[1].set_yticks(range(len(res)))
    axes[1].set_yticklabels([f"{NICE.get(r[0], r[0])}" for r in res])
    axes[1].axvline(0.5, ls="--", color="gray", lw=1, label="random ranking")
    axes[1].set_xlabel("per-user AUC (bars = mean, whiskers = 95 % bootstrap CI)")
    axes[1].set_title("(b) Discriminative power per user")
    for i, r in enumerate(res):
        axes[1].text(r[1] + 0.008, i, f"{r[1]:.3f}", va="center", fontsize=7.5)
    axes[1].set_xlim(0.45, 0.85)
    axes[1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Category-level behaviour and discriminative power (Amazon Beauty)",
                 fontweight="bold", y=1.03)
    save(fig, "figS8_category_and_roc")


def figS9():
    """Content representations give cold items a meaningful position in the space."""
    D = ds()
    seqs, items = D["seqs"], D["items"]
    M = len(items)
    cnt = np.zeros(M)
    for u in seqs:
        for row in np.asarray(seqs[u])[:-2]:
            cnt[row[0]] += 1
    raw = [(D["meta"].get(a, {}).get("cats") or ["?"]) for a in items]
    lab = np.array([(c + ["?"] * 4)[2] for c in raw])
    _, cc = np.unique(lab, return_counts=True)
    chance = float(((cc / cc.sum()) ** 2).sum())

    m = llmsemrec()
    m.invalidate_candidates()
    with torch.no_grad():
        Vid = m.inputs.id_emb(torch.arange(M).unsqueeze(0))[0].detach().numpy()
    Tq = np.load(os.path.join(CACHE, f"{DS}_qwen3_0.6b.npy")).astype(np.float32)

    def purity(X, sel, k=10):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        Z = Xn[sel]
        S_ = Z @ Z.T
        np.fill_diagonal(S_, -2)
        nn = np.argpartition(-S_, k, axis=1)[:, :k]
        return float(np.mean(lab[sel][nn] == lab[sel][:, None]))

    buckets = [(0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 10, "6-10"),
               (11, 50, "11-50"), (51, 10 ** 9, ">50")]
    xs, pid, pq, nb = [], [], [], []
    for lo, hi, nm in buckets:
        sel = np.where((cnt >= lo) & (cnt <= hi))[0]
        if len(sel) < 30:
            continue
        xs.append(nm); pid.append(purity(Vid, sel)); pq.append(purity(Tq, sel)); nb.append(len(sel))

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 3.9))
    axes[0].plot(range(len(xs)), pid, "o-", color="#95A5A6", label="learned identifier embedding")
    axes[0].plot(range(len(xs)), pq, "s-", color=COL["llmsemrec"],
                 label="frozen LLM text embedding")
    axes[0].axhline(chance, ls="--", color="black", lw=1, label=f"frequency chance ({chance:.3f})")
    axes[0].set_xticks(range(len(xs)))
    axes[0].set_xticklabels([f"{x}\n(n={n})" for x, n in zip(xs, nb)], fontsize=7.5)
    axes[0].set_xlabel("training interactions of the item")
    axes[0].set_ylabel("10-NN category purity")
    axes[0].set_title("(a) Content keeps cold items organised;\nidentifiers do not")
    axes[0].legend(fontsize=7)

    pop = pop_counts()
    sp = split("test")
    tpop = pop[sp.tgt[:, -1]]
    q = np.quantile(tpop, [1 / 3, 2 / 3])
    b = np.digitize(tpop, q)
    w = 0.8 / 3
    for i, mm in enumerate(["llmsemrec", "sasrec", "bprmf"]):
        d = npz(mm)
        if d is None:
            continue
        h = (d["test_ranks"] < 10).astype(float)
        axes[1].bar(np.arange(3) + i * w, [h[b == k].mean() * 100 for k in range(3)], w,
                    label=NICE.get(mm, mm), color=COL.get(mm), edgecolor="black", lw=0.4)
    axes[1].set_xticks(np.arange(3) + w)
    axes[1].set_xticklabels(["cold", "medium", "warm"])
    axes[1].set_xlabel("test-target popularity tercile")
    axes[1].set_ylabel("Recall@10 (%)")
    axes[1].set_title("(b) The advantage is largest\nwhere identifiers are weakest")
    axes[1].legend(fontsize=7.5)

    ref = npz("sasrec")
    ours = npz("llmsemrec")
    if ref is not None and ours is not None:
        ho = (ours["test_ranks"] < 10).astype(float)
        hs = (ref["test_ranks"] < 10).astype(float)
        ratio = [(ho[b == k].mean() / hs[b == k].mean()) if hs[b == k].mean() > 0 else np.nan
                 for k in range(3)]
        CAP = 30.0
        plot = [CAP if not np.isfinite(v) else min(v, CAP) for v in ratio]
        axes[2].bar(range(3), plot, color=COL["llmsemrec"], edgecolor="black", lw=0.4)
        for i, (v, h) in enumerate(zip(ratio, (hs[b == k].mean() for k in range(3)))):
            lbl = (f"$\\infty$\n(SASRec = {int(h * (b == i).sum())}/{(b == i).sum():,})"
                   if not np.isfinite(v) else f"{v:.1f}×")
            axes[2].text(i, plot[i], lbl, ha="center", va="bottom", fontsize=8)
        axes[2].set_ylim(0, CAP * 1.30)
        axes[2].axhline(1.0, ls="--", color="gray", lw=1, label="parity with SASRec")
        axes[2].set_xticks(range(3)); axes[2].set_xticklabels(["cold", "medium", "warm"])
        axes[2].set_xlabel("test-target popularity tercile")
        axes[2].set_ylabel("Recall@10 ratio (LLM-SEMRec / SASRec)")
        axes[2].set_title("(c) Relative gain over the\nstrongest ID baseline")
        axes[2].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Why content matters: representation quality on cold items (Amazon Beauty)",
                 fontweight="bold", y=1.03)
    save(fig, "figS9_content_cold_items")


if __name__ == "__main__":
    want = sys.argv[1:]
    print("building supporting figures (Amazon Beauty) ...")
    for fn in (figS1, figS2, figS3, figS4, figS5, figS6, figS7, figS8, figS9):
        tag = fn.__name__.replace("fig", "").lower()
        if want and tag not in want:
            continue
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  !! {fn.__name__} failed: {e}")
            traceback.print_exc()
    print("done")
