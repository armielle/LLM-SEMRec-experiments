"""
Publication-quality figure generation for the LLM-SEMRec paper.

Every figure is written to figures/ as both PNG (300 dpi) and PDF (vector).
Run after run_main.py / run_ablation.py / run_analysis.py.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import FIGURES as FIG, RESULTS
from common import load_json

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.bbox": "tight",
})
C = {"ours": "#C0392B", "seq": "#2471A3", "ssl": "#1E8449", "mm": "#B7950B",
     "llm": "#6C3483", "base": "#7F8C8D", "acc": "#2E86C1", "acc2": "#E67E22",
     "g1": "#34495E", "g2": "#E74C3C", "g3": "#16A085"}
MODEL_ORDER = ["popularity", "bprmf", "gru4rec", "caser", "sasrec", "bert4rec",
               "cl4srec", "s3rec", "mmsasrec", "llmonly", "mmssl", "llmsemrec"]
NICE = {"popularity": "Popularity", "bprmf": "BPR-MF", "gru4rec": "GRU4Rec",
        "caser": "Caser", "sasrec": "SASRec", "bert4rec": "BERT4Rec",
        "cl4srec": "CL4SRec", "s3rec": "S3-Rec", "mmsasrec": "MM-SASRec",
        "llmonly": "LLM-only", "mmssl": "MMSSL-lite", "llmsemrec": "LLM-SEMRec"}
MCOL = {"popularity": C["base"], "bprmf": C["base"], "gru4rec": C["seq"], "caser": C["seq"],
        "sasrec": C["seq"], "bert4rec": C["seq"], "cl4srec": C["ssl"], "s3rec": C["ssl"],
        "mmsasrec": C["mm"], "llmonly": C["llm"], "mmssl": C["mm"], "llmsemrec": C["ours"]}
DATASETS = ["amazon_beauty", "amazon_fashion", "movielens_1m"]
DNAME = {"amazon_beauty": "Amazon Beauty", "amazon_fashion": "Amazon Fashion",
         "movielens_1m": "MovieLens-1M"}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)
    print("  fig:", name)


def try_load(p):
    try:
        return load_json(p)
    except Exception:
        return None


# ---------------------------------------------------------------- 1. architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")

    def box(x, y, w, h, text, fc, ec="black", fs=8, lw=1.1, tc="black", style="round,pad=0.35"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec,
                                    lw=lw, alpha=0.95, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3, color=tc, linespacing=1.35)

    def arrow(x1, y1, x2, y2, style="-|>", color="#333333", lw=1.1, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=11, lw=lw, color=color, ls=ls,
                                     connectionstyle=f"arc3,rad={rad}", zorder=1))

    box(1, 44, 15, 9, "Item-ID\n$e_i^{id}$", "#D6EAF8")
    box(1, 33.5, 15, 9, "Item image\n224×224", "#FCF3CF")
    box(1, 23, 15, 9, "Item text profile\n" + r"$x_i^{txt}$ (Eq. 12)", "#E8DAEF")
    box(1, 12.5, 15, 9, "Feedback $r_t$\nTime $\\Delta\\tau_t$", "#D5F5E3")

    box(20, 30, 17, 23.5, "FROZEN foundation encoders\n\nSigLIP 2 base\n(patch16-224, 768-d)\n\n"
        "Qwen3-Embedding-0.6B\n(1024-d)", "#FDEBD0", lw=1.4, fs=7.5)
    box(20, 8, 17, 19, "Trainable\nembeddings\n\n" + r"$E_r$, $E_{\Delta\tau}$" +
        "\n(ratings, time buckets)", "#EAF2F8", fs=7.5)

    box(41, 22, 15, 24, "Projection to shared space\n$d=128$ (this run)\n\n"
        + r"$\tilde{v}_i=\mathrm{LN}(W_v v_i)$" + "\n" + r"$\tilde{l}_i=\mathrm{LN}(W_l l_i)$",
        "#F4F6F6", fs=7.5)

    box(60, 24, 18, 23, "Reliability-aware\ngated multimodal fusion\n\n(Eq. 15–20)\n\n"
        "modalities: {id, img, txt, r, time}\nmasked softmax gate +\navailability indicators",
        "#FADBD8", lw=1.5, fs=7.5)

    box(60, 4, 18, 15, "Hybrid sequence encoder\n\nCausal TCN (dil. 1,2)\n"
        "→ Transformer ×2 (causal)\n→ GRU → temporal attention",
        "#D4E6F1", fs=7.5)

    box(82, 30, 16, 17, "Variational preference\n\n" + r"$\mu_u,\ \sigma_u^2$  (Eq. 36–39)" +
        "\n" + r"$z_u=\mu_u+\sigma_u\odot\epsilon$" + "\n" + r"$L_{KL}$ (Eq. 40)",
        "#D5F5E3", fs=7.5)
    box(82, 15.5, 16, 12.5, "Uncertainty-aware\nranking\n\n"
        + r"$s=\mu_u^{\top}v_j-\gamma\sqrt{v_j^{\top}\Sigma_u v_j}$"
        + "\n(Eq. 42–44)", "#FADBD8", fs=7.5)
    box(82, 4, 16, 8.5, "Top-K +\nconfidence", "#F9EBEA", fs=8)

    box(41, 4, 15, 13, "Auxiliary\nself-supervised\ntasks\n\nMIP (24)\nSeqCL (25)\n"
        "CMCL (26–28)", "#EAF7EE", fs=7.5)
    box(62, 49, 14, 5.5, r"Mixed negatives (Eq. 53)" + "\n1 in-batch + 4 pop + 5 semantic",
        "#FEF9E7", fs=6.4)

    arrow(16.2, 48.5, 19.8, 48.5)
    arrow(16.2, 38, 19.8, 38)
    arrow(16.2, 27.5, 19.8, 33)
    arrow(16.2, 17, 19.8, 17)
    arrow(37.2, 45, 40.8, 40)
    arrow(37.2, 20, 40.8, 26)
    arrow(56.2, 34, 59.8, 34)
    arrow(69, 23.8, 69, 19.2)
    arrow(78.2, 35.5, 81.8, 38.5)
    arrow(69, 48.8, 69, 47.2, ls="--", color="#B7950B")
    arrow(55.8, 10.5, 59.8, 11.5, ls="--", color=C["ssl"])
    arrow(48.5, 17.2, 48.5, 21.8, ls="--", color=C["ssl"])
    arrow(41, 10.5, 37.4, 13, ls="--", color=C["ssl"])

    ax.text(48.5, 56, "LLM-SEMRec: self-supervised LLM-enhanced multimodal sequential recommender",
            ha="center", fontsize=11, fontweight="bold")
    ax.plot([2, 98], [2.4, 2.4], color="white")
    ax.text(2, 0.6, "solid arrows: recommendation path     dashed arrows: auxiliary self-supervised objectives",
            fontsize=7.5, color="#555555")
    save(fig, "fig1_architecture")


# ---------------------------------------------------------------- 2. training dynamics
def fig_training():
    r = try_load(os.path.join(RESULTS, "llmsemrec__amazon_beauty.json"))
    if not r or not r["runs"][0]["history"]:
        return
    h = r["runs"][0]["history"]
    ep = [x["epoch"] for x in h]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for k, lab, c in (("loss_rec", r"$L_{Rec}$", C["ours"]), ("loss_mip", r"$L_{MIP}$", C["ssl"]),
                      ("loss_seqcl", r"$L_{SeqCL}$", C["llm"]), ("loss_cmcl", r"$L_{CMCL}$", C["mm"])):
        if k in h[0]:
            axes[0].plot(ep, [x[k] for x in h], marker="o", ms=2.5, label=lab, color=c)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
    axes[0].set_title("(a) Supervised + self-supervised objectives")
    if "loss_kl" in h[0]:
        axes[1].plot(ep, [x["loss_kl"] for x in h], marker="s", ms=2.5, color=C["g3"])
        axes[1].set_ylabel(r"$L_{KL}$ (nats)"); axes[1].set_title("(b) KL regulariser (Eq. 40)")
    axes[1].set_xlabel("epoch")
    axes[2].plot(ep, [x["val_NDCG@10"] for x in h], marker="o", ms=3.5,
                 color=C["ours"], label="val NDCG@10")
    axes[2].plot(ep, [x["val_Recall@10"] for x in h], marker="^", ms=3.5,
                 color=C["acc"], label="val Recall@10")
    best = int(np.argmax([x["val_NDCG@10"] for x in h]))
    axes[2].axvline(ep[best], ls=":", color="gray")
    top = max(x["val_NDCG@10"] for x in h)
    axes[2].set_ylim(0, top * 1.45)
    axes[2].annotate(f"best epoch {ep[best]}", (ep[best], top * 1.12), fontsize=7,
                     color="gray", ha="left")
    axes[2].set_xlabel("epoch"); axes[2].legend(loc="lower left")
    axes[2].set_title("(c) Validation performance")
    fig.suptitle("LLM-SEMRec training dynamics (Amazon Beauty)", fontweight="bold", y=1.02)
    save(fig, "fig2_training_dynamics")


# ---------------------------------------------------------------- 3. main results
def fig_main():
    from common import collect_summary
    data = {d: collect_summary(d) for d in DATASETS}
    data = {k: v for k, v in data.items() if v}
    if not data:
        return
    for metric, fname, title in (("NDCG@10", "fig3_main_ndcg10", "NDCG@10"),
                                 ("Recall@10", "fig3b_main_recall10", "Recall@10")):
        d0 = list(data)[0]
        models = [m for m in MODEL_ORDER if m in data[d0]]
        fig, ax = plt.subplots(figsize=(10.5, 3.9))
        w = 0.8 / max(len(data), 1)
        for i, (d, s) in enumerate(data.items()):
            vals = [s[m]["full"][metric]["mean"] if m in s else 0 for m in models]
            errs = [s[m]["full"][metric]["std"] if m in s else 0 for m in models]
            ax.bar(np.arange(len(models)) + i * w, vals, w, yerr=errs, capsize=2,
                   label=DNAME.get(d, d), alpha=0.92,
                   color=[MCOL[m] for m in models],
                   edgecolor="black", lw=0.4,
                   hatch=["", "//", ".."][i % 3])
        ax.set_xticks(np.arange(len(models)) + w * (len(data) - 1) / 2)
        ax.set_xticklabels([NICE.get(m, m) for m in models], rotation=35, ha="right")
        ax.set_ylabel(title)
        top = max([s[m]["full"][metric]["mean"] for s in data.values() for m in models
                   if m in s] + [1e-9])
        ax.set_ylim(0, top * 1.38)          # headroom for the legend
        ax.set_title(f"Overall recommendation performance — {title} (full-catalogue ranking)")
        ax.legend(title="dataset", ncols=2, loc="upper left", framealpha=0.95)
        save(fig, fname)


# ---------------------------------------------------------------- 4. ablation
def fig_ablation():
    from common import collect_ablation
    s = collect_ablation("amazon_beauty")
    if not s:
        return
    order = ["full", "wo_qwen3", "wo_siglip", "wo_ssl", "wo_vae", "simple_fusion", "uniform_neg"]
    labels = {"full": "Full LLM-SEMRec", "wo_qwen3": "w/o Qwen3 (text)", "wo_siglip": "w/o SigLIP (image)",
              "wo_ssl": "w/o SSL (MIP+SeqCL+CMCL)", "wo_vae": "w/o VAE", "simple_fusion": "Simple fusion",
              "uniform_neg": "Uniform negatives"}
    order = [k for k in order if k in s]
    base = s["full"]["full"]["Recall@10"]["mean"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8))
    y = np.arange(len(order))
    rec = [s[k]["full"]["Recall@10"]["mean"] for k in order]
    ndcg = [s[k]["full"]["NDCG@10"]["mean"] for k in order]
    axes[0].barh(y, rec, color=[C["ours"]] + [C["base"]] * (len(order) - 1),
                 edgecolor="black", lw=0.4)
    axes[0].set_yticks(y); axes[0].set_yticklabels([labels[k] for k in order])
    axes[0].invert_yaxis(); axes[0].set_xlabel("Recall@10")
    axes[0].set_title("(a) Ablation — Recall@10")
    for i, v in enumerate(rec):
        axes[0].text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=7)
    delta = [(v - base) / base * 100 for v in rec]
    cols = [C["ours"] if d >= 0 else C["g2"] for d in delta]
    axes[1].barh(y, delta, color=cols, edgecolor="black", lw=0.4)
    axes[1].set_yticks(y); axes[1].set_yticklabels([])
    axes[1].invert_yaxis()            # keep the rows aligned with panel (a)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Δ Recall@10 vs. full model (%)")
    axes[1].set_title("(b) Relative contribution of each component")
    for i, v in enumerate(delta):
        axes[1].text(v + (0.4 if v >= 0 else -0.4), i, f"{v:+.1f}%", va="center",
                     ha="left" if v >= 0 else "right", fontsize=7)
    fig.suptitle("Ablation study (Amazon Beauty)", fontweight="bold", y=1.03)
    save(fig, "fig4_ablation")
    # second panel: NDCG
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    mr = [s[k]["full"]["MRR@10"]["mean"] for k in order]
    ax.barh(y, ndcg, color=C["acc"], alpha=0.85, edgecolor="black", lw=0.4, label="NDCG@10")
    ax.barh(y + 0.0, mr, color=C["acc2"], alpha=0.6, height=0.4, edgecolor="black",
            lw=0.3, label="MRR@10")
    ax.set_yticks(y); ax.set_yticklabels([labels[k] for k in order]); ax.invert_yaxis()
    ax.legend(); ax.set_title("Ablation — NDCG@10 and MRR@10")
    save(fig, "fig4b_ablation_ndcg")


# ---------------------------------------------------------------- 5. sparsity / cold start
def fig_sparsity_coldstart():
    a = try_load(os.path.join(RESULTS, "analysis__amazon_beauty.json"))
    if not a:
        return
    ms = [m for m in ["llmsemrec", "sasrec", "llmonly"] if m in a["models"]]
    if not ms:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for ax, key, title, xl in (
            (axes[0], "sparsity", "(a) Sparse feedback: by history length", "history-length bucket"),
            (axes[1], "coldstart_popularity", "(b) Cold start: target-item popularity", "popularity bucket"),
            (axes[2], "coldstart_longtail", "(c) Long-tail items", "training interactions of target")):
        groups = a["models"][ms[0]][key]["groups"]
        n = len(groups)
        w = 0.8 / len(ms)
        for i, m in enumerate(ms):
            g = a["models"][m][key]["groups"]
            vals = [x.get("Recall@10", 0) for x in g]
            ax.bar(np.arange(n) + i * w, vals, w, label=NICE.get(m, m), color=MCOL.get(m),
                   edgecolor="black", lw=0.4)
            for j, v in enumerate(vals):
                ax.text(j + i * w, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.5)
        ax.set_xticks(np.arange(n) + w * (len(ms) - 1) / 2)
        ax.set_xticklabels(["short/cold", "medium", "long/warm"][:n])
        ax.set_ylabel("Recall@10"); ax.set_xlabel(xl); ax.set_title(title)
        top = max([x.get("Recall@10", 0) for m in ms
                   for x in a["models"][m][key]["groups"]] + [1e-6])
        ax.set_ylim(0, top * 1.28)          # headroom so value labels clear the title
        ax.legend(loc="upper left", framealpha=0.9)
    fig.suptitle("Sparsity and cold-start analyses (Amazon Beauty)", fontweight="bold", y=1.03)
    save(fig, "fig5_sparsity_coldstart")


# ---------------------------------------------------------------- 6. calibration
def fig_calibration():
    a = try_load(os.path.join(RESULTS, "analysis__amazon_beauty.json"))
    if not a or "llmsemrec" not in a["models"]:
        return
    r = a["models"]["llmsemrec"]
    cal = r["calibration"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))
    curve = [c for c in cal["curve"] if c["conf"] is not None]
    xs = [c["conf"] for c in curve]
    ys = [c["acc"] for c in curve]
    axes[0].plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="perfect calibration")
    axes[0].plot(xs, ys, "o-", color=C["ours"], ms=5, lw=1.6, label="LLM-SEMRec")
    axes[0].set_xlabel("predicted probability (temperature-scaled)")
    axes[0].set_ylabel("empirical accuracy@10")
    axes[0].set_title(f"(a) Reliability diagram\nT={cal['temperature']:.2f}, "
                      f"ECE={cal['ECE@10']:.4f}, Brier={cal['brier']:.4f}")
    axes[0].legend()

    dec = r["confidence_deciles"]
    d1 = [d for d in dec if d["accuracy"] is not None]
    axes[1].bar(range(len(d1)), [d["accuracy"] for d in d1], color=C["acc"],
                edgecolor="black", lw=0.4, label="empirical accuracy@10")
    axes[1].plot(range(len(d1)), [d["mean_prob"] for d in d1], "o--", color=C["ours"],
                 ms=4, label="mean predicted probability")
    axes[1].set_xticks(range(len(d1)))
    axes[1].set_xticklabels([str(d["decile"]) for d in d1])
    axes[1].set_xlabel("confidence decile"); axes[1].legend()
    axes[1].set_title("(b) Accuracy vs. predicted confidence")

    q = r["uncertainty_quartiles"]
    axes[2].bar(range(4), [x["Recall@10"] for x in q], color=[C["g3"], "#7DCEA0", "#F5B041", C["g2"]],
                edgecolor="black", lw=0.4)
    for i, x in enumerate(q):
        axes[2].text(i, x["Recall@10"] + 0.005, f"{x['Recall@10']:.3f}\nσ={x['sigma_mean']:.3f}",
                     ha="center", fontsize=6.5)
    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels(["Q1\n(most confident)", "Q2", "Q3", "Q4\n(least confident)"])
    axes[2].set_ylabel("Recall@10")
    ymax = max(x["Recall@10"] for x in q)
    axes[2].set_ylim(0, ymax * 1.30)
    axes[2].set_title("(c) Error vs. posterior uncertainty")
    fig.suptitle("Uncertainty and calibration analysis (Amazon Beauty)",
                 fontweight="bold", y=1.03)
    save(fig, "fig6_calibration")


# ---------------------------------------------------------------- 7. selective + gamma
def fig_selective_gamma():
    a = try_load(os.path.join(RESULTS, "analysis__amazon_beauty.json"))
    if not a or "llmsemrec" not in a["models"]:
        return
    r = a["models"]["llmsemrec"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    sel = r["selective"]
    axes[0].plot([s["coverage"] for s in sel], [s["Recall@10"] for s in sel], "o-",
                 color=C["ours"], ms=5)
    for s in sel:
        axes[0].annotate(f"σ·({s['saved_fraction']*100:.0f}% dropped)",
                         (s["coverage"], s["Recall@10"]), fontsize=6,
                         xytext=(4, -8), textcoords="offset points")
    axes[0].set_xlabel("coverage (fraction of users served)")
    axes[0].set_ylabel("Recall@10")
    axes[0].set_title("(a) Selective recommendation\n(abstain on the most uncertain users)")
    axes[0].invert_xaxis()

    g = r.get("gamma_sweep")
    if g:
        ax2 = axes[1]
        gs = [x["gamma"] for x in g]
        ax2.plot(gs, [x["NDCG@10"] for x in g], "o-", color=C["ours"], label="NDCG@10")
        ax2.plot(gs, [x["Recall@10"] for x in g], "s-", color=C["acc"], label="Recall@10")
        ax2.plot(gs, [x["MRR@10"] for x in g], "^-", color=C["ssl"], label="MRR@10")
        ax2.set_xlabel(r"risk-aversion $\gamma$ (Eq. 44)"); ax2.set_ylabel("metric")
        ax2.set_title("(b) Uncertainty-aware ranking:\neffect of the risk-aversion coefficient")
        ax2.legend()
    fig.suptitle("Uncertainty-aware ranking behaviour (Amazon Beauty)", fontweight="bold", y=1.03)
    save(fig, "fig7_selective_gamma")


# ---------------------------------------------------------------- 8. coverage / diversity
def fig_coverage_diversity():
    from common import collect_summary
    figs = {d: collect_summary(d) for d in DATASETS}
    figs = {k: v for k, v in figs.items() if v}
    if not figs:
        return
    d0 = list(figs)[0]
    models = [m for m in MODEL_ORDER if m in figs[d0] and figs[d0][m].get("list")]
    if not models:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for ax, key, title in ((axes[0], "coverage@10", "(a) Catalogue coverage@10"),
                           (axes[1], "diversity@10", "(b) Intra-list diversity@10"),
                           (axes[2], "novelty@10", "(c) Novelty@10 (self-information)")):
        vals = [figs[d0][m]["list"].get(key, 0) for m in models]
        ax.bar(range(len(models)), vals, color=[MCOL[m] for m in models],
               edgecolor="black", lw=0.4)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([NICE.get(m, m) for m in models], rotation=40, ha="right")
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
    fig.suptitle(f"Beyond accuracy: coverage, diversity and novelty ({DNAME[d0]})",
                 fontweight="bold", y=1.03)
    save(fig, "fig8_coverage_diversity")


# ---------------------------------------------------------------- 9. complexity
def fig_complexity():
    from common import collect_summary
    a = try_load(os.path.join(RESULTS, "analysis__amazon_beauty.json"))
    s = collect_summary("amazon_beauty") or None
    if not a or "llmsemrec" not in a["models"]:
        return
    r = a["models"]["llmsemrec"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    if s:
        models = [m for m in MODEL_ORDER if m in s and s[m].get("params")]
        vals = [max(s[m]["params"], 1) for m in models]
        axes[0].bar(range(len(models)), vals, color=[MCOL[m] for m in models],
                    edgecolor="black", lw=0.4)
        axes[0].set_yscale("log")
        axes[0].set_xticks(range(len(models)))
        axes[0].set_xticklabels([NICE.get(m, m) for m in models], rotation=40, ha="right")
        axes[0].set_ylabel("trainable parameters (log scale)")
        axes[0].set_title("(a) Trainable parameter count")
    lat = r.get("latency", {})
    if lat.get("seq_len"):
        axes[1].plot([x["L"] for x in lat["seq_len"]],
                     [x["ms_per_256_users"] for x in lat["seq_len"]], "o-", color=C["ours"])
        for x in lat["seq_len"]:
            axes[1].annotate(f"{x['ms_per_256_users']:.0f} ms",
                             (x["L"], x["ms_per_256_users"]), fontsize=7,
                             xytext=(3, 3), textcoords="offset points")
        axes[1].set_xlabel("maximum history length $L$")
        axes[1].set_ylabel("latency (ms / 256 users)")
        axes[1].set_title("(b) Inference latency vs. $L$")
    tt = [x["sec"] for x in try_load(os.path.join(RESULTS, "llmsemrec__amazon_beauty.json"))
          ["runs"][0]["history"]] if try_load(os.path.join(RESULTS, "llmsemrec__amazon_beauty.json")) else []
    if tt:
        axes[2].bar(range(1, len(tt) + 1), tt, color=C["g3"], edgecolor="black", lw=0.4)
        axes[2].axhline(np.mean(tt), ls="--", color=C["g2"],
                        label=f"mean {np.mean(tt):.0f}s")
        axes[2].set_xlabel("epoch"); axes[2].set_ylabel("seconds")
        axes[2].set_title("(c) Training time per epoch")
        axes[2].legend()
    fig.suptitle("Computational complexity (Amazon Beauty, CPU-only host)",
                 fontweight="bold", y=1.03)
    save(fig, "fig9_complexity")


# ---------------------------------------------------------------- 10. rank curves
def fig_rank_curves():
    from common import collect_summary
    r = try_load(os.path.join(RESULTS, "llmsemrec__amazon_beauty.json"))
    s = collect_summary("amazon_beauty")
    if not r or "llmsemrec" not in s:
        return
    ks = [1, 2, 5, 10, 20, 50, 100]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    for m in MODEL_ORDER:
        p = os.path.join(RESULTS, f"{m}__amazon_beauty.json")
        rr = try_load(p)
        if not rr:
            continue
        ranks = np.array(rr["runs"][0]["ranks_full"])
        axes[0].plot(ks, [(ranks < k).mean() for k in ks], "o-", ms=3,
                     label=NICE.get(m, m), color=MCOL[m],
                     lw=2.0 if m == "llmsemrec" else 1.0,
                     alpha=1.0 if m == "llmsemrec" else 0.7)
    axes[0].set_xscale("log"); axes[0].set_xticks(ks); axes[0].set_xticklabels(ks)
    axes[0].set_xlabel("@K"); axes[0].set_ylabel("Recall@K")
    axes[0].set_title("(a) Recall@K curves (full-catalogue)")
    axes[0].legend(fontsize=6.5, ncols=2)
    ranks = np.array(r["runs"][0]["ranks_full"])
    axes[1].hist(np.clip(ranks, 0, 200), bins=60, color=C["ours"], alpha=0.85,
                 edgecolor="white", lw=0.3)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("rank of the ground-truth item (clipped at 200)")
    axes[1].set_ylabel("number of test users (log)")
    axes[1].set_title(f"(b) Rank distribution — LLM-SEMRec\n"
                      f"MRR@10={s['llmsemrec']['full']['MRR@10']['mean']:.4f}")
    save(fig, "fig10_rank_curves")


if __name__ == "__main__":
    print("generating figures ->", FIG)
    fig_architecture()
    for f in (fig_training, fig_main, fig_ablation, fig_sparsity_coldstart,
              fig_calibration, fig_selective_gamma, fig_coverage_diversity,
              fig_complexity, fig_rank_curves):
        try:
            f()
        except Exception as e:
            print("  !!", f.__name__, "failed:", e)
    print("done")
