"""Shared utilities: dataset loading, evaluation protocol, metrics."""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CACHE, DATA, FIGURES, RESULTS, TABLES, ROOT  # noqa: F401

L_MAX = 50


def load_ds(name):
    with open(os.path.join(DATA, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------- evaluation protocol
def build_eval_candidates(ds, n_neg=100, seed=42):
    """Sampled evaluation: 1 positive + n_neg negatives drawn from the TRAINING catalog.

    Candidate sets are frozen so every model is evaluated on identical items.
    """
    rng = np.random.RandomState(seed)
    train_items = set()
    for u, v in ds["train_targets"].items():
        for (i, _, _) in v:
            train_items.add(i)
    train_items = np.array(sorted(train_items), dtype=np.int64)
    seen = {u: set(x[0] for x in v) for u, v in ds["seqs"].items()}

    out = {}
    for split, key in (("val", "val_targets"), ("test", "test_targets")):
        users = sorted(ds[key].keys())
        items = np.zeros((len(users), n_neg + 1), dtype=np.int64)
        for r, u in enumerate(users):
            tgt = ds[key][u][0]
            s = seen[u] | {tgt}
            negs = []
            while len(negs) < n_neg:
                cand = train_items[rng.randint(0, len(train_items))]
                if cand not in s and cand not in negs:
                    negs.append(int(cand))
            items[r, 0] = tgt
            items[r, 1:] = negs
        out[split] = {"users": users, "items": items}
    return out


def metrics_from_ranks(ranks, ks=(5, 10, 20)):
    """Ranking metrics from the 0-indexed rank of the ground-truth item."""
    m = {}
    ranks = np.asarray(ranks)
    for k in ks:
        hit = (ranks < k).astype(np.float64)
        m[f"Recall@{k}"] = float(hit.mean())
        m[f"HR@{k}"] = float(hit.mean())
        m[f"Precision@{k}"] = float(hit.mean() / k)
        m[f"NDCG@{k}"] = float(np.where(ranks < k, 1.0 / np.log2(ranks + 2.0), 0.0).mean())
        m[f"MRR@{k}"] = float(np.where(ranks < k, 1.0 / (ranks + 1.0), 0.0).mean())
    return m


def rank_curves(ranks, ks=(1, 2, 5, 10, 20, 50)):
    r = np.asarray(ranks)
    return {f"Recall@{k}": float((r < k).mean()) for k in ks}


def bootstrap_ci(values, n_boot=1000, seed=0, alpha=0.05):
    if len(values) == 0:
        return 0.0, 0.0
    rng = np.random.RandomState(seed)
    v = np.asarray(values, dtype=np.float64)
    means = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_ttest(a, b):
    """Two-sided paired t-test over user-level metric values (paper Sec. VI-E)."""
    from scipy import stats
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 2:
        return {"t": 0.0, "p": 1.0}
    t, p = stats.ttest_rel(a, b)
    return {"t": float(t), "p": float(p)}


def per_user_hits(ranks, k=10):
    return (np.asarray(ranks) < k).astype(np.float64)


# ------------------------------------------------------------------- list metrics
def list_metrics(topk_items, pop_counts, n_items, item_repr=None, k=10):
    """Coverage / intra-list diversity / novelty of the recommended top-K lists."""
    flat = np.concatenate([np.asarray(t)[:k] for t in topk_items])
    coverage = len(np.unique(flat)) / float(n_items)
    total = float(np.maximum(pop_counts, 0).sum())
    p = np.maximum(pop_counts, 0) / max(total, 1e-12)
    novelty = float(np.mean(-np.log2(np.clip(p[flat], 1e-12, None))))
    diversity = 0.0
    if item_repr is not None:
        sims = []
        for t in topk_items:
            v = np.asarray(item_repr)[np.asarray(t)[:k]]
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
            s = v @ v.T
            iu = np.triu_indices(len(v), 1)
            if len(iu[0]):
                sims.append(s[iu].mean())
        diversity = float(1.0 - np.mean(sims)) if sims else 0.0
    return {f"coverage@{k}": coverage, f"diversity@{k}": diversity,
            f"novelty@{k}": novelty}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print("wrote", path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def collect_ablation(dataset, results_dir=None):
    """Aggregate every ablation__<variant>__<dataset>.json into one mapping."""
    import glob
    rd = results_dir or RESULTS
    out = {}
    for p in sorted(glob.glob(os.path.join(rd, f"ablation__*__{dataset}.json"))):
        try:
            r = load_json(p)
        except Exception:
            continue
        v = r.get("variant") or os.path.basename(p).split("__")[1]
        out[v] = {"full": r["aggregate_full"], "list": r.get("list_metrics", {}),
                  "params": r.get("n_params", 0), "overrides": r.get("overrides", {})}
    return out


def collect_summary(dataset, results_dir=None):
    """Aggregate every <model>__<dataset>.json into the summary structure.

    More robust than relying on a single summary file, which is rewritten by
    every partial run.
    """
    import glob
    rd = results_dir or RESULTS
    out = {}
    for p in sorted(glob.glob(os.path.join(rd, f"*__{dataset}.json"))):
        base = os.path.basename(p)
        if base.startswith("_") or base.startswith("ablation__"):
            continue        # ablation variants are aggregated by collect_ablation()
        model = base.split("__")[0]
        try:
            r = load_json(p)
        except Exception:
            continue
        if "aggregate_full" not in r:
            continue
        out[model] = {"full": r["aggregate_full"],
                      "sampled": r.get("aggregate_sampled", {}),
                      "params": r.get("n_params", 0),
                      "seconds": r.get("wall_seconds", 0.0),
                      "list": r.get("list_metrics", {})}
    return out
