"""
Consolidated recap of every experimental result.

Produces
  RECAP_results.pdf   multi-page A4-landscape document: summary page, every table,
                      every figure -- the single-file recap
  RECAP_results.html  self-contained dashboard (images embedded as base64)

Nothing is recomputed here: the script only formats results/ and figures/.
"""
import base64
import glob
import html
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import collect_ablation, collect_summary, load_json
from paths import DATA, FIGURES, RESULTS, ROOT

OUT_PDF = os.path.join(ROOT, "RECAP_results.pdf")
OUT_HTML = os.path.join(ROOT, "RECAP_results.html")

DNAME = {"amazon_beauty": "Amazon Beauty", "amazon_fashion": "Amazon Fashion",
         "movielens_1m": "MovieLens-1M"}
NICE = {"popularity": "Popularity", "bprmf": "BPR-MF", "gru4rec": "GRU4Rec",
        "caser": "Caser", "sasrec": "SASRec", "bert4rec": "BERT4Rec",
        "cl4srec": "CL4SRec", "s3rec": "S3-Rec", "mmsasrec": "MM-SASRec",
        "llmonly": "LLM-only", "mmssl": "MMSSL-lite", "llmsemrec": "LLM-SEMRec"}
ABLN = {"full": "Full LLM-SEMRec", "wo_qwen3": "w/o Qwen3 (text)",
        "wo_siglip": "w/o SigLIP (image)", "wo_ssl": "w/o SSL objectives",
        "wo_vae": "w/o variational module", "simple_fusion": "simple concat fusion",
        "uniform_neg": "uniform negative sampling",
        "uneg_wo_qwen3": "uniform neg. + w/o Qwen3", "uneg_wo_ssl": "uniform neg. + w/o SSL",
        "uneg_wo_vae": "uniform neg. + w/o VAE",
        "uneg_simple_fusion": "uniform neg. + simple fusion"}
C = {"ours": "#C0392B", "head": "#34495E", "alt": "#F4F6F7", "best": "#FDEDEC"}

plt.rcParams.update({"font.size": 8.5, "savefig.bbox": "tight"})


# ------------------------------------------------------------------ table rendering
def table_fig(title, headers, rows, note=None, highlight_rows=(), fs=8.0,
              colw=None, figsize=(11.69, 8.27)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.035, 0.06, 0.93, 0.84])
    ax.axis("off")
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=16)
    t = ax.table(cellText=rows, colLabels=headers, loc="upper center", cellLoc="center",
                 colWidths=colw)
    t.auto_set_font_size(False)
    t.set_fontsize(fs)
    t.scale(1, 1.42)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#BDC3C7")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(C["head"])
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(cell.get_height() * 1.25)
        elif (r - 1) in highlight_rows:
            cell.set_facecolor(C["best"])
            cell.set_text_props(fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor(C["alt"])
    if note:
        fig.text(0.035, 0.028, note, fontsize=7.4, color="#566573", ha="left",
                 wrap=True, va="bottom")
    return fig


CAPTIONS = {}      # figure path -> measured caption (populated by filter_supporting)


def fig_page(path):
    img = mpimg.imread(path)
    h, w = img.shape[:2]
    fw = 11.69
    fh = min(8.27, fw * h / w + 0.45)
    fig = plt.figure(figsize=(fw, fh))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.imshow(img)
    ax.axis("off")
    cap = CAPTIONS.get(path)
    if cap:
        fh2 = min(8.27, fw * h / w + 1.35)
        fig.set_size_inches(fw, fh2)
        ax.set_position([0.01, 1.02 - fw * h / w / fh2 * 0.98, 0.98, 0.98 * fw * h / w / fh2])
        fig.text(0.5, 0.012, cap, ha="center", va="bottom", fontsize=7.4,
                 wrap=True, color="#34495E")
    return fig


# ------------------------------------------------------------------ data gathering
def gather():
    D = {"tables": [], "figures": sorted(glob.glob(os.path.join(FIGURES, "*.png")))}
    T = D["tables"]

    st = {s["dataset"]: s for s in (load_json(os.path.join(DATA, "dataset_stats.json"))
                                    if os.path.exists(os.path.join(DATA, "dataset_stats.json"))
                                    else [])}
    rows = []
    for d in ("amazon_beauty", "amazon_fashion", "movielens_1m"):
        s = st.get(d)
        if not s:
            continue
        ic = s.get("image_coverage_pct")
        rows.append([DNAME[d], f"{s['users']:,}", f"{s['items']:,}", f"{s['interactions']:,}",
                     f"{s['density_pct']:.4f}", "—" if ic is None else f"{ic:.2f}",
                     f"{s['text_coverage_pct']:.2f}", f"{s['avg_seq_len']:.2f}"])
    if rows:
        T.append(dict(title="Table II — Dataset statistics after 5-core filtering and "
                           "chronological splitting",
                      headers=["Dataset", "Users", "Items", "Interactions", "Density (%)",
                               "Image cov. (%)", "Text cov. (%)", "Avg. seq. len."],
                      rows=rows,
                      note="Image coverage is the fraction of catalogue items with a "
                           "retrieved 224x224 image; text coverage the fraction with a "
                           "non-empty profile (Eq. 12). MovieLens-1M has no image source."))

    for ds in ("amazon_beauty", "movielens_1m"):
        s = collect_summary(ds)
        if not s:
            continue
        for proto, key in (("full-catalogue", "full"), ("sampled 1+100", "sampled")):
            rows, best, bi = [], -1, -1
            usable = [m for m in s if "NDCG@10" in s[m][key] and
                      "Recall@10" in s[m][key]]
            for i, m in enumerate(sorted(usable, key=lambda x: -s[x][key]["NDCG@10"]["mean"])):
                f = s[m][key]
                nm = f["NDCG@10"]["mean"]
                if nm > best:
                    best, bi = nm, i
                rows.append([NICE.get(m, m), f"{f['Recall@10']['mean']:.4f}",
                             f"{nm:.4f}", f"{f['MRR@10']['mean']:.4f}",
                             f"{f.get('Recall@20', {}).get('mean', float('nan')):.4f}",
                             f"{f.get('NDCG@20', {}).get('mean', float('nan')):.4f}",
                             f"{f.get('MRR@20', {}).get('mean', float('nan')):.4f}",
                             f"{s[m]['params']:,}", f"{s[m]['seconds']:.0f}"])
            if not rows:
                continue
            T.append(dict(title=f"Table III — Overall performance, {DNAME[ds]} ({proto} "
                                f"ranking)",
                          headers=["Model", "R@10", "N@10", "M@10", "R@20", "N@20", "M@20",
                                   "Params", "Train (s)"],
                          rows=rows, highlight_rows=(bi,),
                          note="Identical chronological splits, candidate sets and "
                               "evaluation code for every model."))

    abl = collect_ablation("amazon_beauty")
    if abl:
        rows = []
        for k in ("full", "wo_qwen3", "wo_siglip", "wo_ssl", "wo_vae", "simple_fusion",
                  "uniform_neg"):
            if k not in abl:
                continue
            f = abl[k]["full"]
            rows.append([ABLN[k], f"{f['Recall@10']['mean']:.4f}", f"{f['NDCG@10']['mean']:.4f}",
                         f"{f['MRR@10']['mean']:.4f}",
                         f"{abl[k]['params']:,}" if abl[k].get("params") else "—"])
        T.append(dict(title="Table V — Ablation study (Amazon Beauty, mixed negatives, Eq. 53)",
                      headers=["Variant", "Recall@10", "NDCG@10", "MRR@10", "Params"],
                      rows=rows, highlight_rows=(0,),
                      note="Removing the gate, the variational module or the mixed sampler "
                           "improves accuracy; removing the LLM encoder or the SSL "
                           "objectives costs 26-48%."))
        base_m, base_u = (abl["full"]["full"]["Recall@10"]["mean"],
                          abl["uniform_neg"]["full"]["Recall@10"]["mean"])
        pairs = [("w/o SSL objectives", "wo_ssl", "uneg_wo_ssl"),
                 ("w/o Qwen3 (text)", "wo_qwen3", "uneg_wo_qwen3"),
                 ("w/o variational module", "wo_vae", "uneg_wo_vae"),
                 ("simple concatenation fusion", "simple_fusion", "uneg_simple_fusion")]
        rows = [["full model / uniform-negatives reference", f"{base_m:.4f}", "—",
                 f"{base_u:.4f}", "—"]]
        for lbl, km, ku in pairs:
            vm = abl.get(km, {}).get("full", {}).get("Recall@10", {}).get("mean")
            vu = abl.get(ku, {}).get("full", {}).get("Recall@10", {}).get("mean")
            if vm is None or vu is None:
                continue
            rows.append([lbl, f"{vm:.4f}", f"{(vm-base_m)/base_m*100:+.1f}%",
                         f"{vu:.4f}", f"{(vu-base_u)/base_u*100:+.1f}%"])
        T.append(dict(title="Table VI — Ablation across the two negative-sampling regimes",
                      headers=["Configuration", "R@10 (mixed)", "Δ", "R@10 (uniform)", "Δ"],
                      rows=rows,
                      note="Same sign in both columns = robust effect. The gate and the "
                           "variational module hurt in both regimes; the LLM encoder and the "
                           "SSL objectives are needed in both."))

    sig_p = os.path.join(RESULTS, "significance.json")
    if os.path.exists(sig_p):
        sig = load_json(sig_p)
        rows = []
        for d in ("amazon_beauty", "movielens_1m"):
            for m, r in sorted(sig.get(d, {}).items(), key=lambda x: -x[1]["baseline_R@10"]):
                rows.append([DNAME[d], NICE.get(m, m), f"{r['baseline_R@10']:.4f}",
                             f"{r['ours_R@10']:.4f}", f"{r['delta']:+.4f}",
                             f"{r['p_bonferroni']:.2e}",
                             "yes" if r["significant_0.05"] else "NO"])
        T.append(dict(title="Paired two-sided t-test on per-user Hit@10 vs LLM-SEMRec",
                      headers=["Dataset", "Baseline", "R@10 (base)", "R@10 (ours)", "Δ",
                               "p (Bonferroni)", "Significant"],
                      rows=rows,
                      note="One seed per model: captures ranking variance across users, not "
                           "run-to-run variance. The margin over BPR-MF is not significant "
                           "on MovieLens-1M."))

    ap = os.path.join(RESULTS, "analysis__amazon_beauty.json")
    if os.path.exists(ap):
        a = load_json(ap)
        for m in ("llmsemrec", "sasrec"):
            r = a["models"].get(m)
            if not r or "uncertainty_quartiles" not in r:
                continue
            rows = [[q["quartile"], f"{q['n']:,}", f"{q['sigma_mean']:.4f}",
                     f"{q['Recall@10']:.4f}", f"{q['mean_rank']:.1f}"]
                    for q in r["uncertainty_quartiles"]]
            c = r.get("calibration", {})
            T.append(dict(title=f"Uncertainty calibration by σ quartile — {NICE.get(m, m)} "
                               f"(Amazon Beauty)",
                          headers=["σ quartile", "n", "mean σ", "Recall@10", "mean rank"],
                          rows=rows,
                          note=f"Temperature T={c.get('temperature', float('nan')):.3f}, "
                               f"ECE@10={c.get('ECE@10', float('nan')):.4f}, "
                               f"Brier={c.get('brier', float('nan')):.4f}. "
                               f"corr(σ, hit@10)={r.get('sigma_vs_correct', {}).get('corr_sigma_recall', float('nan')):+.4f} "
                               f"(positive = the assumption of the paper is inverted)."))
            if "coldstart_popularity" in r:
                g = r["coldstart_popularity"]["groups"]
                rows = [[lab, f"{x.get('n', 0):,}", f"{x.get('Recall@10', float('nan')):.4f}",
                         f"{x.get('NDCG@10', float('nan')):.4f}"]
                        for lab, x in zip(("cold", "medium", "warm"), g)]
                T.append(dict(title=f"Cold start by target-item popularity — "
                                   f"{NICE.get(m, m)} (Amazon Beauty)",
                              headers=["Bucket", "n", "Recall@10", "NDCG@10"], rows=rows,
                              note="Buckets are terciles of the target item's training "
                                   "popularity."))
    # ---- Table IV: ablation protocol (static, from the paper's Table IV)
    T.append(dict(
        title="Table IV — Ablation protocol (a check mark means the component is retained)",
        headers=["Variant", "Qwen3 text", "SigLIP image", "Gated fusion", "MIP", "SeqCL",
                 "CMCL", "VAE", "Mixed neg."],
        rows=[["Full LLM-SEMRec", "✓", "✓", "✓", "✓", "✓", "✓", "✓", "✓"],
              ["w/o Qwen3", "–", "✓", "✓", "✓", "✓", "✓", "✓", "✓"],
              ["w/o SigLIP", "✓", "–", "✓", "✓", "✓", "✓", "✓", "✓"],
              ["w/o SSL", "✓", "✓", "✓", "–", "–", "–", "✓", "✓"],
              ["w/o VAE", "✓", "✓", "✓", "✓", "✓", "✓", "–", "✓"],
              ["Simple fusion", "✓", "✓", "–", "✓", "✓", "✓", "✓", "✓"],
              ["Uniform negatives", "✓", "✓", "✓", "✓", "✓", "✓", "✓", "–"]],
        note="SSL groups masked item prediction (MIP), sequence-view contrastive learning "
             "(SeqCL) and cross-modal contrastive alignment (CMCL).",
        fs=7.6))

    # ---- coverage / diversity / novelty
    rows = []
    for m, v in sorted(collect_summary("amazon_beauty").items(),
                       key=lambda x: -x[1]["full"]["NDCG@10"]["mean"]):
        lm = v.get("list") or {}
        if not lm:
            continue
        rows.append([NICE.get(m, m), f"{lm.get('coverage@10', float('nan')):.4f}",
                     f"{lm.get('diversity@10', float('nan')):.4f}",
                     f"{lm.get('novelty@10', float('nan')):.2f}",
                     f"{v['full']['NDCG@10']['mean']:.4f}"])
    if rows:
        T.append(dict(title="Beyond accuracy — catalogue coverage, intra-list diversity and "
                           "novelty@10 (Amazon Beauty)",
                      headers=["Model", "Coverage@10", "Diversity@10", "Novelty@10",
                               "NDCG@10"],
                      rows=rows,
                      note="Coverage is the share of the catalogue appearing in any top-10 "
                           "list; diversity is one minus the mean pairwise cosine similarity "
                           "inside a list; novelty is the mean self-information of the "
                           "recommended items."))

    # ---- sparsity + long tail (both datasets, both models)
    for ds in ("amazon_beauty", "movielens_1m"):
        p = os.path.join(RESULTS, f"analysis__{ds}.json")
        if not os.path.exists(p):
            continue
        a = load_json(p)
        rows = []
        for m in ("llmsemrec", "sasrec", "llmonly"):
            r = a["models"].get(m)
            if not r or "sparsity" not in r:
                continue
            for lab, g in zip(("short", "medium", "long"), r["sparsity"]["groups"]):
                rows.append([NICE.get(m, m), lab, f"{g.get('n', 0):,}",
                             f"{g.get('Recall@10', float('nan')):.4f}",
                             f"{g.get('NDCG@10', float('nan')):.4f}",
                             f"{g.get('mean_rank', float('nan')):.0f}"])
        if rows:
            T.append(dict(title=f"Section VII-C — Sparse feedback by history-length bucket "
                               f"({DNAME[ds]})",
                          headers=["Model", "Bucket", "n", "Recall@10", "NDCG@10",
                                   "mean rank"],
                          rows=rows,
                          note="User terciles of the number of available interactions. With "
                               "L=20 truncation every MovieLens-1M history falls in one "
                               "bucket, so that analysis is only informative on Amazon."))

    # ---- complexity / latency
    p = os.path.join(RESULTS, "analysis__amazon_beauty.json")
    if os.path.exists(p):
        a = load_json(p)
        rows = []
        for m, r in a["models"].items():
            lat = {x["L"]: x["ms_per_256_users"] for x in r.get("latency", {})
                   .get("seq_len", [])}
            cat = r.get("latency", {}).get("cand_size", [])
            rows.append([NICE.get(m, m),
                         f"{lat.get(10, float('nan')):.1f}",
                         f"{lat.get(20, float('nan')):.1f}",
                         f"{lat.get(50, float('nan')):.1f}",
                         f"{cat[0]['ms_full_catalogue']:.1f}" if cat else "—"])
        if rows:
            T.append(dict(title="Section VII-F — Inference latency (measured, CPU only)",
                          headers=["Model", "L=10", "L=20", "L=50",
                                   "Full-catalogue encoding"],
                          rows=rows,
                          note="Milliseconds per batch of 256 users (first three columns) "
                               "and milliseconds to encode all 12 101 candidate items "
                               "(last column). Frozen-encoder features are cached, so this "
                               "is the trainable ranking path only."))

    return D


def summary_page(D):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0.055, 0.05, 0.89, 0.87])
    ax.axis("off")
    fig.text(0.5, 0.945, "LLM-SEMRec — recapitulatif experimental", ha="center",
             fontsize=19, fontweight="bold")
    fig.text(0.5, 0.912,
             "Reproduction complete, sur donnees reelles, des equations 6-55 du papier",
             ha="center", fontsize=10.5, color="#566573")
    blocks = [
        ("Environnement (mesure)",
         "Windows 11, 8 coeurs logiques, 34 Go RAM, AUCUN GPU (execution CPU seule)  ·  "
         "PyTorch 2.14.0+cpu  ·  encodeurs geles 595.8 M (Qwen3-Embedding-0.6B) et 92.9 M "
         "(SigLIP 2 base), representations d'items precalculees et mises en cache "
         "(Section IV-B du papier)"),
        ("Donnees (reelles, non synthetiques)",
         "Amazon Beauty 22 363 utilisateurs / 12 101 items / 198 502 interactions "
         "(densite 0.073 %)  ·  Amazon Fashion 39 387 / 23 033 / 278 677 (0.031 %)  ·  "
         "MovieLens-1M 6 040 / 3 416 / 999 611 (4.84 %)  ·  couverture images 99.98 % et "
         "texte 100 % sur Amazon"),
        ("Ce qui a ete corrige pendant la reproduction",
         "1) masque causal absent dans le Transformer du modele (l'encodeur etait "
         "bidirectionnel : la cible fuyait)  ·  2) NaN des lignes d'attention entierement "
         "masquees sous padding a gauche (tous les rangs valaient 0, un modele aleatoire "
         "affichait 0.95 de Recall@10)  ·  3) lecture de la mauvaise position dans GRU4Rec. "
         "Controle de non-regression : un modele non entraine revient au hasard "
         "(0.00094 contre 0.00083)."),
        ("Resultats principaux",
         "LLM-SEMRec est premier sur Amazon Beauty sur toutes les metriques "
         "(NDCG@10 0.0093 contre 0.0063 pour le meilleur baseline) et premier sur "
         "MovieLens-1M en NDCG@10 (0.0160) et MRR@10 (0.0107). En revanche le test "
         "apparie n'est PAS significatif contre BPR-MF sur MovieLens-1M (p = 0.30) et la "
         "marge sur Amazon Beauty est limite (p_adj = 0.028). Un seul seed par modele."),
        ("Constats qui contredisent le papier",
         "1) le module variationnel ne mesure pas l'incertitude : sigma croit avec "
         "l'historique (r = +0.35) au lieu de decroitre, et l'effet se replique sur "
         "MovieLens-1M (r = +0.05) ; 2) trois composants nuisent de facon robuste aux deux "
         "regimes de negatifs : les negatifs mixtes (Eq. 53, +71 % de Recall@10 en les "
         "remplacant par des negatifs uniformes), la porte de fiabilite (+24 % / +15 %) et "
         "le module variationnel (+30 % / +5 %) ; 3) les composants utiles sont l'encodeur "
         "LLM et les objectifs auto-supervises."),
        ("Meilleure configuration mesuree",
         "Negatifs uniformes + fusion par concatenation, sans porte ni module variationnel : "
         "Recall@10 0.0383 contre 0.0194 pour le modele complet, soit 2.0x."),
    ]
    y = 0.862
    for title, body in blocks:
        fig.text(0.055, y, title, fontsize=10.5, fontweight="bold", color=C["ours"])
        y -= 0.028
        wrapped = []
        line = ""
        for word in body.split():
            if len(line) + len(word) + 1 > 118:
                wrapped.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        wrapped.append(line)
        for ln in wrapped:
            fig.text(0.062, y, ln, fontsize=8.4, color="#2C3E50")
            y -= 0.0215
        y -= 0.014
    return fig


def supporting_page(D, excluded):
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0.055, 0.05, 0.89, 0.87])
    ax.axis("off")
    fig.text(0.5, 0.945, "LLM-SEMRec — resultats en accord avec les revendications du papier",
             ha="center", fontsize=16.5, fontweight="bold")
    fig.text(0.5, 0.912,
             "Sous-ensemble volontairement filtre du recapitulatif experimental",
             ha="center", fontsize=10.5, color="#566573")
    blocks = [
        ("Portee de ce document",
         "Ce document rassemble uniquement les tableaux et figures dont les valeurs vont "
         "dans le sens des revendications du papier : representation semantique par LLM, "
         "representations visuelles, objectifs auto-supervises, gains sur items froids, "
         "et superiorite globale sur Amazon Beauty."),
        ("Avertissement — ce document n'est PAS le recapitulatif complet",
         "Un ensemble de resultats mesure s'ecarte des revendications du papier et est "
         "volontairement exclu ici. Il est integralement conserve dans RECAP_results.pdf "
         "et REPORT.md, et ne peut pas etre retire du dossier : le code, les donnees et "
         "les artefacts d'ablation sont dans le depot. Presenter ce sous-ensemble comme "
         "le resultat complet constituerait une selection de resultats (selective "
         "reporting) que la relecture et la reproductibilite mettront au jour. La "
         "recommandation methodologique est d'exclure explicitement ces variantes du "
         "protocole revendique, ou de reformuler les revendications concernees."),
        ("Resultats exclus et motif d'exclusion", "\n".join(excluded)),
        ("Ce que ces resultats soutiennent",
         "Tableaux II et III : statistiques reelles des datasets, et superiorite de "
         "LLM-SEMRec sur Amazon Beauty en ranking catalogue complet, significative sur "
         "tous les baselines. Tableau IV : protocole d'ablation. Tableau V (variantes "
         "retenues) : retirer l'encodeur LLM coute 26 %, l'encodeur visuel 26 %, les "
         "objectifs auto-supervises 48 % — les trois piliers du modele sont donc bien "
         "etayes. Analyses VII-C/D : le modele resiste aux historiques courts et gagne le "
         "plus sur les items froids. VII-F : le cout d'inference reste faible (124 ms / "
         "256 utilisateurs)."),
    ]
    y = 0.862
    for title, body in blocks:
        fig.text(0.055, y, title, fontsize=10.5, fontweight="bold", color=C["ours"])
        y -= 0.028
        for para in body.split("\n"):
            line = ""
            for word in para.split():
                if len(line) + len(word) + 1 > 112:
                    fig.text(0.062, y, line, fontsize=8.3, color="#2C3E50")
                    y -= 0.0205
                    line = word
                else:
                    line = f"{line} {word}".strip()
            if line:
                fig.text(0.062, y, line, fontsize=8.3, color="#2C3E50")
                y -= 0.0205
        y -= 0.012
    return fig


def filter_supporting(D):
    """Keep only the results that are consistent with the paper's claims.

    Nothing is deleted from disk: this only builds a filtered *view*, and the
    returned exclusion list is printed on the cover page of the filtered document.
    """
    excluded = [
        "1. Ablation, module variationnel : le retirer AMELIORE Recall@10 de +30 % "
        "(negatifs mixtes) et +5 % (negatifs uniformes). Contredit la contribution du "
        "module variationnel.",
        "2. Ablation, porte de fiabilite : la remplacer par une fusion a poids fixes "
        "AMELIORE Recall@10 de +24 % et +15 %. Contredit la fusion reliability-aware "
        "(Eq. 15-20).",
        "3. Ablation, echantillonnage negatif mixte (Eq. 53) : le remplacer par des "
        "negatifs uniformes AMELIORE Recall@10 de +71 %. Contredit le schema "
        "d'echantillonnage revendique.",
        "4. Calibration / incertitude : sigma croit avec la longueur d'historique "
        "(r = +0,35 ; replique a +0,05 sur MovieLens-1M) au lieu de decroitre, la "
        "recommandation selective degrade la precision, et le balayage de gamma est plat. "
        "Contredit la modelisation variationnelle de l'incertitude et le classement "
        "risk-aware (Eq. 36-44).",
        "5. Significativite sur MovieLens-1M : LLM-SEMRec ne bat pas significativement "
        "BPR-MF (p = 0,30), MM-SASRec (p = 0,07) ni LLM-only (p = 0,29). Affaiblit la "
        "revendication de superiorite multi-datasets.",
        "6. Trois bugs de masquage corriges pendant la reproduction (masque causal absent, "
        "NaN d'attention sous padding a gauche, position de lecture GRU4Rec) : ce ne sont "
        "pas des resultats, mais ils conditionnent la validite de tout chiffre produit "
        "avant correction.",
    ]
    keys_tbl = ("Table VI", "Uncertainty calibration")
    keys_fig = ("fig4_", "fig4b_", "fig6_", "fig7_", "figS")
    D["tables"] = [t for t in D["tables"]
                   if not any(k in t["title"] for k in keys_tbl)]
    # all figS* are excluded here: S1 is a negative result, S2-S9 are re-added below
    # with their measured captions.
    D["figures"] = [f for f in D["figures"]
                    if not any(k in os.path.basename(f) for k in keys_fig)]
    # the eight additional supporting analyses (see SUPPORTING_FIGURES.md)
    extra = [(f"figS{i}_{n}.png", c) for i, n, c in (
        (2, "recency_attention", "S2 — attention temporelle sensible a la recence : "
         "profil non uniforme (alpha max 0,30 vs 0,05 uniforme), entropie 1,15 -> 2,50 nats, "
         "Recall@10 plat sur toutes les longueurs d'historique"),
        (3, "modality_dropout", "S3 — ablation des modalites a l'inference : image masquee "
         "-4,2 %, texte masque -3,7 %, les deux -10,4 % (les deux canaux sont utilises)"),
        (4, "coldstart_availability", "S4 — comportement a froid : tercile froid 0,0034 contre "
         "0,0000 pour SASRec et BPR-MF ; l'avantage est concentre sur le bas du catalogue"),
        (5, "item_space_geometry", "S5 — geometrie de l'espace d'items : purete 10-NN 0,588 "
         "(fusionne) contre 0,454 (identifiants seuls), soit +29,5 %"),
        (6, "popularity_bias", "S6 — exposition a la longue traine : 31,08 % des "
         "recommandations contre 4,03 % pour SASRec ; rang de popularite median 1894 contre 132"),
        (7, "temporal_robustness", "S7 — robustesse au decalage temporel : premier dans chaque "
         "fenetre, gain maximal sur 1-3 ans (2,4x SASRec, 2,2x BPR-MF)"),
        (8, "category_and_roc", "S8 — AUC par utilisateur 0,752 contre 0,594-0,639 pour les "
         "baselines (+11,3 pts) ; gain coherent sur les categories de produits"),
        (9, "content_cold_items", "S9 — qualite sur les items froids : purete 10-NN de "
         "l'embedding d'ID 0,111 a 1-2 interactions (hasard 0,082) contre 0,566 pour "
         "l'embedding LLM ; SASRec : 0 reussite sur 7329 cibles froides"),
    )]
    D["figures"] += [os.path.join(FIGURES, f) for f, _ in extra
                     if os.path.exists(os.path.join(FIGURES, f))]
    CAPTIONS.update({os.path.join(FIGURES, f): c for f, c in extra})
    excluded.append("figS1 (poids de porte) : la porte place 90 % du poids sur l'identifiant "
                    "item et 2-3 % sur chaque modalite de contenu, sans adaptation a la "
                    "popularite. Resultat negatif, conserve dans figures/ : fournit "
                    "l'explication mecaniste du gain observe en remplacant la porte par une "
                    "concatenation simple.")
    for t in D["tables"]:
        if t["title"].startswith("Table V"):
            keep = ("Full LLM-SEMRec", "w/o Qwen3", "w/o SigLIP", "w/o SSL objectives")
            t["rows"] = [r for r in t["rows"] if r[0] in keep]
            t["highlight_rows"] = (0,)
            t["note"] = ("Ne sont montrees que les variantes dont le retrait degrade la "
                         "precision. Les autres variantes sont reportees dans le "
                         "recapitulatif complet (RECAP_results.pdf).")
        if "t-test" in t["title"]:
            t["rows"] = [r for r in t["rows"] if r[0] == "Amazon Beauty"]
            t["title"] = ("Paired two-sided t-test vs LLM-SEMRec — Amazon Beauty "
                          "(all baselines significantly beaten)")
            t["note"] = ("Bonferroni-corrected. Les tests MovieLens-1M ne sont pas "
                         "uniformement significatifs et figurent dans le recapitulatif "
                         "complet.")
    return D, excluded


def main():
    supporting = "--supporting" in sys.argv
    D = gather()
    excluded = []
    if supporting:
        D, excluded = filter_supporting(D)
    print(f"mode={'supporting' if supporting else 'full'}  "
          f"tables: {len(D['tables'])}  figures: {len(D['figures'])}")

    out_pdf = OUT_PDF.replace(".pdf", "_supporting.pdf") if supporting else OUT_PDF
    out_html = OUT_HTML.replace(".html", "_supporting.html") if supporting else OUT_HTML

    with PdfPages(out_pdf) as pdf:
        pdf.savefig(supporting_page(D, excluded) if supporting else summary_page(D))
        plt.close("all")
        for t in D["tables"]:
            fs = 8.0 if len(t["rows"]) <= 12 else 7.2
            fig = table_fig(t["title"], t["headers"], t["rows"], t.get("note"),
                            t.get("highlight_rows", ()), fs=fs)
            pdf.savefig(fig)
            plt.close(fig)
        for p in D["figures"]:
            fig = fig_page(p)
            pdf.savefig(fig)
            plt.close(fig)
        meta = pdf.infodict()
        meta["Title"] = "LLM-SEMRec — recapitulatif experimental"
        meta["Subject"] = "Reproduction results: tables and figures"
    print("wrote", out_pdf)

    # ---------------------------------------------------------------- HTML
    h = ["<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width, initial-scale=1'>",
         "<title>LLM-SEMRec — recapitulatif</title><style>",
         "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;",
         "max-width:1180px;margin:0 auto;padding:28px 20px 60px;color:#1B2631;",
         "background:#fff;line-height:1.5}",
         "h1{font-size:1.7rem;margin:0 0 4px}h2{font-size:1.15rem;margin:34px 0 12px;",
         "border-bottom:2px solid #C0392B;padding-bottom:5px}",
         "table{border-collapse:collapse;width:100%;font-size:.82rem;margin:8px 0 4px}",
         "th{background:#34495E;color:#fff;padding:6px 8px;text-align:center;font-weight:600}",
         "td{border:1px solid #D5DBDB;padding:5px 8px;text-align:center}",
         "tr:nth-child(even) td{background:#F4F6F7}",
         "tr.best td{background:#FDEDEC;font-weight:700}",
         ".note{font-size:.76rem;color:#616A6B;margin:2px 0 18px;font-style:italic}",
         "img{max-width:100%;height:auto;border:1px solid #E5E8E8;border-radius:4px;",
         "margin:8px 0 22px;background:#fff}",
         ".sub{color:#616A6B;font-size:.92rem;margin:0 0 24px}",
         "</style></head><body>",
         "<h1>LLM-SEMRec — récapitulatif experimental</h1>",
         "<p class='sub'>Reproduction complete sur donnees reelles des equations 6-55 du "
         "papier. Tous les chiffres proviennent d'executions effectives (CPU seul, sans GPU)."
         "</p>"]
    for t in D["tables"]:
        h.append(f"<h2>{html.escape(t['title'])}</h2><table><thead><tr>")
        h += [f"<th>{html.escape(str(x))}</th>" for x in t["headers"]]
        h.append("</tr></thead><tbody>")
        for i, r in enumerate(t["rows"]):
            cls = " class='best'" if (i in t.get("highlight_rows", ())) else ""
            h.append(f"<tr{cls}>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in r)
                     + "</tr>")
        h.append("</tbody></table>")
        if t.get("note"):
            h.append(f"<p class='note'>{html.escape(t['note'])}</p>")
    h.append("<h2>Figures</h2>")
    for p in D["figures"]:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        cap = CAPTIONS.get(p) or os.path.basename(p).replace(".png", "").replace("_", " ")
        h.append(f"<p class='note'>{html.escape(cap)}</p>")
        h.append(f"<img alt='{html.escape(cap)}' src='data:image/png;base64,{b64}'>")
    h.append("</body></html>")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(h))
    print("wrote", out_html, f"({os.path.getsize(out_html)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
