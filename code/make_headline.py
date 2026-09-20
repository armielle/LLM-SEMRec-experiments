"""
Extract the compact headline metrics from the bulky per-run result files.

The per-run json files in results/ carry per-user rank arrays and therefore
weigh 1.4-5.7 MB each (~127 MB in total). This script distils the aggregate
numbers needed to reproduce every table and figure into a single small file,
so the repository can ship the results without the bulk.

    python code/make_headline.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nice  # noqa: F401

from common import collect_ablation, collect_summary
from paths import RESULTS

DATASETS = ["amazon_beauty", "amazon_fashion", "movielens_1m"]
KEYS = ["Recall@5", "Recall@10", "Recall@20", "Recall@50",
        "NDCG@5", "NDCG@10", "NDCG@20", "NDCG@50",
        "MRR@5", "MRR@10", "MRR@20", "MRR@50",
        "Hit@10", "Precision@10"]


def compact(block):
    """Keep only the mean of each aggregate metric."""
    out = {}
    for k in KEYS:
        v = block.get(k)
        if isinstance(v, dict) and "mean" in v:
            out[k] = round(float(v["mean"]), 6)
        elif isinstance(v, (int, float)):
            out[k] = round(float(v), 6)
    return out


def main():
    out = {"_note": "Aggregate metrics distilled from the per-run result files. "
                    "Regenerate the full per-run files with code/run_main.py and "
                    "the ablation chains before re-running the figure scripts.",
           "datasets": {}}

    for ds in DATASETS:
        s = collect_summary(ds)
        if not s:
            continue
        entry = {"models": {}}
        for model, d in sorted(s.items()):
            rec = {}
            if d.get("full"):
                rec["full_catalogue"] = compact(d["full"])
            if d.get("sampled"):
                rec["sampled_1_plus_100"] = compact(d["sampled"])
            if d.get("list"):
                rec["beyond_accuracy"] = {k: round(float(v), 6)
                                          for k, v in d["list"].items()}
            if d.get("params"):
                rec["trainable_params"] = int(d["params"])
            if rec:
                entry["models"][model] = rec
        a = collect_ablation(ds)
        if a:
            entry["ablation"] = {k: compact(v.get("full", {}))
                                 for k, v in sorted(a.items()) if v.get("full")}
        out["datasets"][ds] = entry

    for extra in ("analysis__amazon_beauty", "analysis__movielens_1m",
                  "significance", "dataset_stats"):
        p = os.path.join(RESULTS, extra + ".json")
        if os.path.exists(p):
            out[extra] = json.load(open(p, encoding="utf-8"))

    dst = os.path.join(RESULTS, "headline_metrics.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("wrote %s (%.1f KB)" % (dst, os.path.getsize(dst) / 1024))
    for ds, e in out["datasets"].items():
        print("  %-16s %d models, %d ablation variants"
              % (ds, len(e.get("models", {})), len(e.get("ablation", {}))))


if __name__ == "__main__":
    main()
