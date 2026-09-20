"""
Emit LaTeX tables for the paper from the result JSONs.

Writes tables/*.tex, ready to \input{} into main.tex:
  table2_dataset_stats.tex     Table II   dataset statistics
  table3_overall.tex           Table III  overall performance
  table4_ablation_protocol.tex Table IV   ablation protocol
  table5_ablation.tex          Table V    ablation study
  table6_calibration.tex       Sec. VII-E uncertainty / calibration
  table7_sparsity.tex          Sec. VII-C sparsity buckets
  table8_coldstart.tex         Sec. VII-D cold start
  table9_complexity.tex        Sec. VII-F computational complexity
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_json, load_ds
from paths import DATA, RESULTS, TABLES

MODEL_ORDER = ["popularity", "bprmf", "gru4rec", "caser", "sasrec", "bert4rec",
               "cl4srec", "s3rec", "mmsasrec", "llmonly", "mmssl", "llmsemrec"]
NICE = {"popularity": "Popularity", "bprmf": "BPR-MF", "gru4rec": "GRU4Rec",
        "caser": "Caser", "sasrec": "SASRec", "bert4rec": "BERT4Rec",
        "cl4srec": "CL4SRec", "s3rec": "S3-Rec", "mmsasrec": "MM-SASRec",
        "llmonly": "LLM-only", "mmssl": "MMSSL-lite", "llmsemrec": "\\model{}"}
ABL_ORDER = ["full", "wo_qwen3", "wo_siglip", "wo_ssl", "wo_vae", "simple_fusion", "uniform_neg"]
DNAME = {"amazon_beauty": "Amazon Beauty", "amazon_fashion": "Amazon Fashion",
         "movielens_1m": "MovieLens-1M"}
DATASETS = ["amazon_beauty", "amazon_fashion", "movielens_1m"]


def w(name, txt):
    p = os.path.join(TABLES, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(txt)
    print("wrote", p)


def tload(p):
    try:
        return load_json(p)
    except Exception:
        return None


# ---------------------------------------------------------------- Table II
def table2():
    s = tload(os.path.join(DATA, "dataset_stats.json"))
    if not s:
        return
    have = {x["dataset"]: x for x in s}
    lines = [
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Dataset statistics after iterative 5-core filtering and chronological "
        "splitting. Density is the number of interactions divided by "
        "$|\\mathcal{U}|\\cdot|\\mathcal{I}|$. Image coverage is the fraction of catalogue "
        "items for which a product image could be retrieved; text coverage is the fraction "
        "with a non-empty textual profile (Eq.~\\ref{eq:textprofile}).}",
        "\\label{tab:datasets}", "\\small",
        "\\begin{tabular}{lrrrrrrr}", "\\toprule",
        "Dataset & Users & Items & Interactions & Density (\\%) & Image cov. (\\%) & "
        "Text cov. (\\%) & Avg. seq. len. \\\\", "\\midrule"]
    for d in DATASETS:
        x = have.get(d)
        if not x:
            continue
        lines.append(
            f"{DNAME[d]} & {x['users']:,} & {x['items']:,} & {x['interactions']:,} & "
            f"{x['density_pct']:.4f} & {x.get('image_coverage_pct', float('nan')):.2f} & "
            f"{x['text_coverage_pct']:.2f} & {x['avg_seq_len']:.2f} \\\\".replace(",", "\\,"))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    w("table2_dataset_stats.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------- Table III
def table3():
    from common import collect_summary
    data = {d: collect_summary(d) for d in DATASETS}
    data = {k: v for k, v in data.items() if v}
    if not data:
        return
    best = {}
    for d, s in data.items():
        for m in s:
            best.setdefault(d, {})
            for k in ("Recall@10", "NDCG@10", "MRR@10", "Recall@20", "NDCG@20", "MRR@20"):
                v = s[m]["full"][k]["mean"] if k in s[m]["full"] else None
                if v is not None and v > best[d].get(k, -1):
                    best[d][k] = v
    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{Overall recommendation performance under full-catalogue ranking "
             "(chronological leave-two-out split, identical candidate sets for all models). "
             "Best result per dataset and column in bold. Mean $\\pm$ std over "
             "seeds where available.}",
             "\\label{tab:overall}", "\\small", "\\begin{tabular}{llrrrrrr}", "\\toprule",
             "Dataset & Model & R@10 & N@10 & M@10 & R@20 & N@20 & M@20 \\\\", "\\midrule"]
    for d, s in data.items():
        models = [m for m in MODEL_ORDER if m in s]
        lines.append("\\multirow{%d}{*}{%s}" % (len(models), DNAME[d]))
        for m in models:
            row = []
            for k in ("Recall@10", "NDCG@10", "MRR@10", "Recall@20", "NDCG@20", "MRR@20"):
                v = s[m]["full"].get(k, {}).get("mean")
                if v is None:
                    row.append("--")
                else:
                    t = f"{v:.4f}"
                    if abs(v - best[d].get(k, -1)) < 1e-9:
                        t = f"\\textbf{{{t}}}"
                    row.append(t)
            lines.append(f" & {NICE.get(m, m)} & " + " & ".join(row) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table*}"]
    w("table3_overall.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------- Table IV
def table4():
    rows = [("Full \\model{}", "\\cmark & \\cmark & \\cmark & \\cmark & \\cmark & \\cmark & \\cmark & \\cmark"),
            ("w/o Qwen3", "-- & \\cmark & \\cmark & \\cmark & \\cmark & \\cmark & \\cmark & \\cmark"),
            ("w/o SigLIP", "\\cmark & -- & \\cmark & \\cmark & \\cmark & \\cmark & \\cmark & \\cmark"),
            ("w/o SSL", "\\cmark & \\cmark & \\cmark & -- & -- & -- & \\cmark & \\cmark"),
            ("w/o VAE", "\\cmark & \\cmark & \\cmark & \\cmark & \\cmark & \\cmark & -- & \\cmark"),
            ("Simple fusion", "\\cmark & \\cmark & -- & \\cmark & \\cmark & \\cmark & \\cmark & \\cmark"),
            ("Uniform negatives", "\\cmark & \\cmark & \\cmark & \\cmark & \\cmark & \\cmark & \\cmark & --")]
    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{Ablation protocol for \\model{}. A check mark indicates that the "
             "component is retained. SSL denotes masked item prediction, sequence-view "
             "contrastive learning and cross-modal contrastive alignment.}",
             "\\label{tab:ablation_protocol}", "\\small",
             "\\begin{tabular}{lcccccccc}", "\\toprule",
             "Variant & Qwen3 text & SigLIP image & Gated fusion & MIP & SeqCL & CMCL & "
             "VAE & Mixed neg. \\\\", "\\midrule"]
    for a, b in rows:
        lines.append(f"{a} & {b} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    w("table4_ablation_protocol.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------- Table V
def table5():
    from common import collect_ablation
    s = collect_ablation("amazon_beauty")
    if not s:
        return
    labels = {"full": "Full \\model{}", "wo_qwen3": "Without LLM representations",
              "wo_siglip": "Without visual representations",
              "wo_ssl": "Without self-supervised objectives",
              "wo_vae": "Without variational module",
              "simple_fusion": "Simple concatenation fusion",
              "uniform_neg": "Uniform negative sampling"}
    base = s.get("full", {}).get("full", {}).get("Recall@10", {}).get("mean")
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Ablation study on Amazon Beauty (full-catalogue ranking). "
             "$\\Delta$ is relative to the full model.}",
             "\\label{tab:ablation_results}", "\\small",
             "\\begin{tabular}{lrrrr}", "\\toprule",
             "Variant & R@10 & N@10 & M@10 & $\\Delta$R@10 \\\\", "\\midrule"]
    for k in ABL_ORDER:
        if k not in s:
            continue
        f = s[k]["full"]
        r = f["Recall@10"]["mean"]
        d = f"{(r-base)/base*100:+.1f}\\%" if base and k != "full" else "--"
        bold = "\\textbf{" if k == "full" else ""
        endb = "}" if k == "full" else ""
        lines.append(f"{labels.get(k,k)} & {bold}{r:.4f}{endb} & {bold}{f['NDCG@10']['mean']:.4f}{endb} "
                     f"& {bold}{f['MRR@10']['mean']:.4f}{endb} & {d} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    w("table5_ablation.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------- VII tables
def table_calibration():
    a = tload(os.path.join(RESULTS, "analysis__amazon_beauty.json"))
    if not a:
        return
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Uncertainty and calibration analysis on Amazon Beauty. "
             "Users are sorted by the posterior uncertainty $\\sigma_u$ (Eq.~38) into "
             "quartiles; the temperature is fitted post hoc on the validation split.}",
             "\\label{tab:calibration}", "\\small",
             "\\begin{tabular}{lrrrr}", "\\toprule",
             "Quartile & $n$ & mean $\\sigma_u$ & R@10 & Mean rank \\\\", "\\midrule"]
    for m, r in a["models"].items():
        if "uncertainty_quartiles" not in r:
            continue
        lines.append("\\multicolumn{5}{l}{\\textit{%s}} \\\\" % NICE.get(m, m))
        for q in r["uncertainty_quartiles"]:
            lines.append(f"\\quad {q['quartile']} & {q['n']:,} & {q['sigma_mean']:.4f} & "
                         f"{q['Recall@10']:.4f} & {q['mean_rank']:.1f} \\\\".replace(",", "\\,"))
        c = r["calibration"]
        lines.append(f"\\quad ECE & \\multicolumn{{3}}{{l}}{{{c['ECE@10']:.4f} "
                     f"(Brier {c['brier']:.4f}, $T={c['temperature']:.2f}$)}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    w("table6_calibration.tex", "\n".join(lines) + "\n")


def table_sparsity():
    a = tload(os.path.join(RESULTS, "analysis__amazon_beauty.json"))
    if not a:
        return
    names = ["short", "medium", "long"]
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Performance by history-length bucket (Amazon Beauty). "
             "Buckets are user terciles of the number of available interactions.}",
             "\\label{tab:sparsity}", "\\small",
             "\\begin{tabular}{llrrr}", "\\toprule",
             "Model & Bucket & $n$ & R@10 & N@10 \\\\", "\\midrule"]
    for m, r in a["models"].items():
        if "sparsity" not in r:
            continue
        for i, g in enumerate(r["sparsity"]["groups"]):
            lines.append(f"{NICE.get(m,m)} & {names[i]} & {g['n']:,} & {g['Recall@10']:.4f} "
                         f"& {g['NDCG@10']:.4f} \\\\".replace(",", "\\,"))
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}"]
    w("table7_sparsity.tex", "\n".join(lines) + "\n")


def table_coldstart():
    a = tload(os.path.join(RESULTS, "analysis__amazon_beauty.json"))
    if not a:
        return
    names = ["cold", "medium", "warm"]
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Cold-start analysis (Amazon Beauty). Buckets are terciles of the "
             "target item's training popularity.}",
             "\\label{tab:coldstart}", "\\small",
             "\\begin{tabular}{llrrr}", "\\toprule",
             "Model & Bucket & $n$ & R@10 & N@10 \\\\", "\\midrule"]
    for m, r in a["models"].items():
        if "coldstart_popularity" not in r:
            continue
        for i, g in enumerate(r["coldstart_popularity"]["groups"]):
            lines.append(f"{NICE.get(m,m)} & {names[i]} & {g['n']:,} & {g['Recall@10']:.4f} "
                         f"& {g['NDCG@10']:.4f} \\\\".replace(",", "\\,"))
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}"]
    w("table8_coldstart.tex", "\n".join(lines) + "\n")


def table_complexity():
    from common import collect_summary
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Computational complexity. Trainable parameters exclude the frozen "
             "foundation encoders (Qwen3-Embedding-0.6B: 595.8\\,M, SigLIP~2 base vision "
             "tower: 92.9\\,M), whose item representations are precomputed and cached.}",
             "\\label{tab:complexity}", "\\small",
             "\\begin{tabular}{lrrr}", "\\toprule",
             "Model & Trainable params & s/epoch & Latency \\\\", "\\midrule"]
    for d in DATASETS:
        s = collect_summary(d)
        if not s:
            continue
        for m in MODEL_ORDER:
            if m not in s:
                continue
            tr = tload(os.path.join(RESULTS, f"{m}__{d}.json"))
            spe = "--"
            if tr and tr["runs"] and tr["runs"][0]["history"]:
                hist = [h for h in tr["runs"][0]["history"] if h.get("sec")]
                if hist:
                    spe = f"{sum(h['sec'] for h in hist)/len(hist):.1f}"
            lines.append(f"{NICE.get(m,m)} ({DNAME[d].split()[0]}) & "
                         f"{s[m]['params']:,} & {spe} & -- \\\\".replace(",", "\\,"))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    w("table9_complexity.tex", "\n".join(lines) + "\n")


def table_significance():
    """Paired two-sided t-test of each baseline against LLM-SEMRec (Sec. VI-E)."""
    p = os.path.join(RESULTS, "significance.json")
    if not os.path.exists(p):
        return
    sig = load_json(p)
    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{Paired two-sided $t$-test on per-user Hit@10 against \\model{} "
             "(same users, same candidate sets, one training seed per model). "
             "$p_{\\mathrm{adj}}$ is Bonferroni-corrected over the baselines of that "
             "dataset.}",
             "\\label{tab:significance}", "\\small",
             "\\begin{tabular}{llrrrr}", "\\toprule",
             "Dataset & Baseline & R@10 (base) & R@10 (ours) & $p_{\\mathrm{adj}}$ & Sig. \\\\",
             "\\midrule"]
    for d in DATASETS:
        if d not in sig:
            continue
        nm = DNAME.get(d, d)
        rows = sorted(sig[d].items(), key=lambda x: -x[1]["baseline_R@10"])
        for m, r in rows:
            star = "yes" if r["significant_0.05"] else "no"
            lines.append(f"{nm} & {NICE.get(m, m)} & {r['baseline_R@10']:.4f} & "
                         f"{r['ours_R@10']:.4f} & {r['p_bonferroni']:.2e} & {star} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table*}"]
    w("table10_significance.tex", "\n".join(lines) + "\n")


if __name__ == "__main__":
    table2(); table3(); table4(); table5()
    table_calibration(); table_sparsity(); table_coldstart(); table_complexity()
    table_significance()
    print("done")
