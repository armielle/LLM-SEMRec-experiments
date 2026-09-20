"""
Paired significance testing across models (paper Sec. VI-E).

Uses the per-user rank artefacts saved by run_main.py, so no retraining is
needed: for every baseline we run a two-sided paired t-test on per-user
Hit@10 against LLM-SEMRec, and report Bonferroni-corrected significance.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RESULTS

DATASETS = [("amazon_beauty", "Amazon Beauty"), ("movielens_1m", "MovieLens-1M")]
MODELS = ["popularity", "bprmf", "gru4rec", "caser", "sasrec", "bert4rec",
          "cl4srec", "s3rec", "mmsasrec", "llmonly", "mmssl"]
NICE = {"popularity": "Popularity", "bprmf": "BPR-MF", "gru4rec": "GRU4Rec",
        "caser": "Caser", "sasrec": "SASRec", "bert4rec": "BERT4Rec",
        "cl4srec": "CL4SRec", "s3rec": "S3-Rec", "mmsasrec": "MM-SASRec",
        "llmonly": "LLM-only", "mmssl": "MMSSL-lite"}
MODELDIR = os.path.join(RESULTS, "..", "models")


def hits(dataset, model, k=10, split="test"):
    p = os.path.join(MODELDIR, f"eval__{model}__{dataset}.npz")
    if not os.path.exists(p):
        return None
    return (np.load(p)[f"{split}_ranks"] < k).astype(np.float64)


def main():
    out = {}
    for ds, nm in DATASETS:
        ours = hits(ds, "llmsemrec")
        if ours is None:
            continue
        out[ds] = {}
        print(f"\n=== {nm} — paired t-test on per-user Hit@10 vs LLM-SEMRec ===")
        print(f"{'baseline':14s} {'R@10':>8s} {'ours':>8s} {'delta':>9s} "
              f"{'t':>9s} {'p':>11s} {'p(adj)':>11s}  sig")
        n = len(MODELS)
        for m in MODELS:
            h = hits(ds, m)
            if h is None:
                continue
            t, p = stats.ttest_rel(h, ours)
            padj = min(p * n, 1.0)
            sig = "***" if padj < 0.001 else "**" if padj < 0.01 else "*" if padj < 0.05 else "n.s."
            print(f"{NICE[m]:14s} {h.mean():8.4f} {ours.mean():8.4f} "
                  f"{h.mean()-ours.mean():+9.4f} {t:9.3f} {p:11.3e} {padj:11.3e}  {sig}")
            out[ds][m] = {"baseline_R@10": float(h.mean()), "ours_R@10": float(ours.mean()),
                          "delta": float(h.mean() - ours.mean()), "t": float(t),
                          "p": float(p), "p_bonferroni": float(padj),
                          "significant_0.05": bool(padj < 0.05), "n_users": int(len(h))}
    p = os.path.join(RESULTS, "significance.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
