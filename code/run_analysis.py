"""
Section VII analyses:
  C  sparsity / history-length buckets
  D  cold-start (target-item popularity and modality availability)
  E  uncertainty and calibration (temperature scaling, reliability, ECE,
     error-vs-uncertainty, selective recommendation, gamma sweep)
  F  computational complexity

Reloads the models and per-user artefacts saved by run_main.py.
Usage: python run_analysis.py <dataset> [--models llmsemrec,sasrec]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nice  # noqa: F401

import train_eval as TE
from common import (RESULTS, build_eval_candidates, load_ds, metrics_from_ranks,
                    save_json)
from paths import RESULTS as RES_DIR
from run_main import build

torch.set_num_threads(nice.threads())
MODELDIR = os.path.join(RES_DIR, "..", "models")


# ---------------------------------------------------------------- helpers
def tercile_labels(values, names=("short", "medium", "long")):
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    lab = np.where(values <= q1, 0, np.where(values <= q2, 1, 2))
    return lab, names


def group_metrics(ranks, labels, n_groups=3):
    out = []
    for g in range(n_groups):
        m = labels == g
        if m.sum() == 0:
            out.append({"n": 0})
            continue
        r = ranks[m]
        out.append({"n": int(m.sum()),
                    "Recall@10": float((r < 10).mean()),
                    "Recall@20": float((r < 20).mean()),
                    "NDCG@10": float(np.where(r < 10, 1 / np.log2(r + 2), 0).mean()),
                    "MRR@10": float(np.where(r < 10, 1 / (r + 1), 0).mean()),
                    "mean_rank": float(r.mean())})
    return out


def ece(p, y, n_bins=10):
    """Expected calibration error with equal-width probability bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    tot, n = 0.0, len(p)
    curve = []
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            curve.append({"bin": (edges[i] + edges[i + 1]) / 2, "conf": None,
                          "acc": None, "n": 0})
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        tot += m.sum() * abs(conf - acc)
        curve.append({"bin": float((edges[i] + edges[i + 1]) / 2), "conf": conf,
                      "acc": acc, "n": int(m.sum())})
    return tot / max(n, 1), curve


def softmax_rows(x, T=1.0):
    x = x / max(T, 1e-6)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(scores, n_grid=60):
    """Post-hoc temperature scaling fitted by NLL on the validation split."""
    y = np.zeros(len(scores))
    best, bestT = 1e18, 1.0
    for T in np.linspace(0.05, 5.0, n_grid):
        p = softmax_rows(scores, T)[:, 0]
        nll = -np.mean(np.log(np.clip(p, 1e-12, 1)))
        if nll < best:
            best, bestT = nll, T
    return float(bestT), float(best)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--models", default="llmsemrec,sasrec")
    ap.add_argument("--lmax", type=int, default=20)
    ap.add_argument("--d", type=int, default=128)
    a = ap.parse_args()

    ds = load_ds(a.dataset)
    names = a.models.split(",")
    L = a.lmax
    cfg0 = dict(TE.DEFAULT_CFG)
    cfg0.update(dict(l_max=L, d=a.d, dz=max(a.d // 2, 32)))
    splits = (TE.build_split(ds, "train", L), TE.build_split(ds, "val", L),
              TE.build_split(ds, "test", L))
    cands = build_eval_candidates(ds, n_neg=100, seed=42)
    n_items = len(ds["items"])

    pop = np.zeros(n_items)
    for u, v in ds["train_targets"].items():
        for (i, _, _) in v:
            pop[i] += 1

    out = {"dataset": a.dataset, "l_max": L, "models": {}}
    test_hist_len = (~splits[2].padmask).sum(1)
    test_targets = splits[2].tgt[:, -1]
    seen = {u: set(x[0] for x in v) for u, v in ds["seqs"].items()}
    test_users = splits[2].users

    for name in names:
        npz_p = os.path.join(MODELDIR, f"eval__{name}__{a.dataset}.npz")
        if not os.path.exists(npz_p):
            print(f"  [skip] {name}: {npz_p} missing")
            continue
        d = np.load(npz_p)
        ranks = d["test_ranks"]
        sigma = d["test_sigma"]
        scores = d["test_scores"]           # [N,101] sampled candidate scores
        vranks, vsigma, vscores = d["val_ranks"], d["val_sigma"], d["val_scores"]
        print(f"\n=== analysis {name} ({a.dataset}) ===", flush=True)
        res = {}

        # ---------------- VII.C  sparsity
        lab, _ = tercile_labels(test_hist_len)
        res["sparsity"] = {"buckets": [int((lab == g).sum()) for g in range(3)],
                           "groups": group_metrics(ranks, lab)}
        print("  VII.C sparsity:", json.dumps(res["sparsity"]["groups"]))

        # ---------------- VII.D  cold-start
        tgt_pop = pop[test_targets]
        lab, _ = tercile_labels(tgt_pop)
        res["coldstart_popularity"] = {"groups": group_metrics(ranks, lab)}
        avail_p = os.path.join(RES_DIR, "..", "cache", f"{a.dataset}_img_avail.npy")
        if os.path.exists(avail_p):
            ia = np.load(avail_p)
            has = ia[test_targets]
            res["coldstart_image"] = {
                "with_image": group_metrics(ranks, np.where(has, 0, 1))[0],
                "without_image": group_metrics(ranks, np.where(has, 0, 1))[1]}
        # long-tail: items with <= 10 training interactions
        tail = np.where(tgt_pop <= 10, 0, np.where(tgt_pop <= 50, 1, 2))
        res["coldstart_longtail"] = {"groups": group_metrics(ranks, tail)}
        print("  VII.D cold-start (pop):", json.dumps(res["coldstart_popularity"]["groups"]))

        # ---------------- VII.E  uncertainty / calibration
        y = (ranks < 10).astype(float)
        T, nll = fit_temperature(vscores)
        p = softmax_rows(scores, T)[:, 0]
        e, curve = ece(p, y, 10)
        res["calibration"] = {
            "temperature": T, "val_nll": nll,
            "ECE@10": float(e), "AUC_like_acc": float(y.mean()),
            "curve": curve,
            "mean_prob": float(p.mean()),
            "brier": float(np.mean((p - y) ** 2)),
            "nll_test": float(-np.mean(np.log(np.clip(p, 1e-12, 1))))}
        q = np.quantile(sigma, [0.25, 0.5, 0.75])
        qlab = np.digitize(sigma, q)
        res["uncertainty_quartiles"] = [{
            "quartile": f"Q{i+1}", "n": int((qlab == i).sum()),
            "sigma_mean": float(sigma[qlab == i].mean()),
            "Recall@10": float(y[qlab == i].mean()),
            "mean_rank": float(ranks[qlab == i].mean())} for i in range(4)]
        res["sigma_vs_correct"] = {
            "corr_sigma_recall": float(np.corrcoef(sigma, y)[0, 1]),
            "sigma_correct": float(sigma[y > 0].mean()) if (y > 0).any() else None,
            "sigma_incorrect": float(sigma[y == 0].mean()) if (y == 0).any() else None}
        # selective recommendation: abstain on the most uncertain users
        order = np.argsort(sigma)
        sel = []
        for frac in (1.0, 0.9, 0.75, 0.5, 0.25, 0.1):
            k = max(int(frac * len(order)), 1)
            idx = order[:k]
            sel.append({"coverage": float(k / len(order)),
                        "Recall@10": float(y[idx].mean()),
                        "saved_fraction": float(1 - k / len(order))})
        res["selective"] = sel
        # calibration-conditioned: accuracy by predicted-confidence decile
        plab = np.digitize(p, np.quantile(p, np.linspace(0, 1, 11)[1:-1]))
        res["confidence_deciles"] = [{
            "decile": i + 1, "n": int((plab == i).sum()),
            "mean_prob": float(p[plab == i].mean()) if (plab == i).any() else None,
            "accuracy": float(y[plab == i].mean()) if (plab == i).any() else None}
            for i in range(10)]
        print(f"  VII.E calibration: T={T:.3f} ECE={e:.4f} "
              f"| acc(Q1)={res['uncertainty_quartiles'][0]['Recall@10']:.3f} "
              f"acc(Q4)={res['uncertainty_quartiles'][3]['Recall@10']:.3f}")

        # ---------------- VII.F  gamma sweep (needs the trained model)
        mp = os.path.join(MODELDIR, f"{name}__{a.dataset}.pt")
        if os.path.exists(mp):
            ck = torch.load(mp, weights_only=False)
            cfg = dict(ck["cfg"])
            feats = TE.load_features(a.dataset, cfg)
            if not cfg.get("use_img", True):
                feats = (None, feats[1], None, 768, feats[4])
            if not cfg.get("use_txt", True):
                feats = (feats[0], None, feats[2], feats[3], 1024)
            if name == "llmsemrec":
                m = TE.build_model(ds, cfg, feats)
            else:
                m, _, _ = build(name, ds, cfg, feats)
                if hasattr(m, "set_item_features"):
                    m.set_item_features(img_feat=feats[0], txt_feat=feats[1],
                                        img_avail=feats[2])
            m.load_state_dict(ck["state"])
            m.eval()
            sweep = []
            t0 = time.time()
            for g in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
                r = TE.evaluate(m, splits[2], ds, cand_sets=cands["test"]["items"], gamma=g)
                sweep.append({"gamma": g, **{k: v for k, v in r["full"].items()
                                             if k in ("Recall@10", "NDCG@10", "MRR@10",
                                                      "Recall@20", "NDCG@20")}})
            res["gamma_sweep"] = sweep
            res["gamma_eval_seconds"] = round(time.time() - t0, 1)
            print("  VII.F gamma sweep:", json.dumps(
                [{"g": s["gamma"], "NDCG@10": round(s["NDCG@10"], 4)} for s in sweep]))

            # ---------------- VII.F  latency vs L and candidate-set size
            lat = {"seq_len": [], "cand_size": []}
            for L2 in (10, 20, 50):
                # a fresh model with the same architecture but a longer position
                # table: latency does not depend on the weight values
                c2 = dict(cfg)
                c2["l_max"] = L2
                if name == "llmsemrec":
                    m2 = TE.build_model(ds, c2, feats)
                else:
                    m2 = build(name, ds, c2, feats)[0]
                    if hasattr(m2, "set_item_features"):
                        m2.set_item_features(img_feat=feats[0], txt_feat=feats[1],
                                             img_avail=feats[2])
                m2.eval()
                sp = TE.build_split(ds, "val", L2)
                ii, rr, tt, pp, _ = sp.to_torch(np.arange(min(256, len(sp.users))))
                t0 = time.time()
                with torch.no_grad():
                    for _ in range(3):
                        m2.encode_sequence(ii, rr, tt, pp)
                lat["seq_len"].append({"L": L2,
                                       "ms_per_256_users": round((time.time() - t0) / 3 * 1000, 1)})
            V = m.candidate_vectors()
            t0 = time.time()
            with torch.no_grad():
                for _ in range(5):
                    m.candidate_vectors(use_cache=False)
            lat["cand_size"].append(
                {"n_candidates": int(m.n_items),
                 "ms_full_catalogue": round((time.time() - t0) / 5 * 1000, 1)})
            res["latency"] = lat
            print("  VII.F latency:", json.dumps(lat))

        out["models"][name] = res
        # persist after every model so a later failure cannot lose earlier work
        save_json(out, os.path.join(RESULTS, f"analysis__{a.dataset}.json"))

    save_json(out, os.path.join(RESULTS, f"analysis__{a.dataset}.json"))
    print("\nwrote analysis")


if __name__ == "__main__":
    main()
