"""
Ablation study -- Tables IV and V of the paper.

Variants (Table IV):
  full            Qwen3 + SigLIP + gated fusion + MIP + SeqCL + CMCL + VAE + mixed negatives
  wo_qwen3        remove the LLM semantic modality
  wo_siglip       remove the visual modality
  wo_ssl          remove MIP, SeqCL and CMCL
  wo_vae          remove the variational module (deterministic preference)
  simple_fusion   replace reliability-aware gating with fixed weights (Eq. 18 disabled)
  uniform_neg     replace mixed negative sampling with uniform negatives

Usage: python run_ablation.py <dataset> [--budget S] [--epochs N]
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
from common import RESULTS, build_eval_candidates, list_metrics, load_ds, save_json
from paths import RESULTS

torch.set_num_threads(nice.threads())

VARIANTS = {
    "full": {},
    "wo_qwen3": {"use_txt": False},
    "wo_siglip": {"use_img": False},
    "wo_ssl": {"lam_mip": 0.0, "lam_seqcl": 0.0, "lam_cmcl": 0.0},
    "wo_vae": {"use_vae": False},
    "simple_fusion": {"gated_fusion": False},
    "uniform_neg": {"uniform_negatives": True},
    # second round: ablate *on top of* uniform negatives, to test whether the
    # components that looked harmful above only look harmful because the mixed
    # sampler of Eq. 53 dominates the training signal
    "uneg_wo_vae": {"uniform_negatives": True, "use_vae": False},
    "uneg_simple_fusion": {"uniform_negatives": True, "gated_fusion": False},
    "uneg_wo_ssl": {"uniform_negatives": True, "lam_mip": 0.0, "lam_seqcl": 0.0,
                    "lam_cmcl": 0.0},
    "uneg_wo_qwen3": {"uniform_negatives": True, "use_txt": False},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--variants", default="all")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--budget", type=int, default=2700)
    ap.add_argument("--lmax", type=int, default=20)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    variants = list(VARIANTS) if a.variants == "all" else a.variants.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    ds = load_ds(a.dataset)
    cfg0 = dict(TE.DEFAULT_CFG)
    cfg0.update(dict(l_max=a.lmax, d=a.d, dz=max(a.d // 2, 32), epochs=a.epochs,
                     patience=a.patience, batch=a.batch))

    print(f"### ABLATION {a.dataset} variants={variants}", flush=True)
    splits = (TE.build_split(ds, "train", cfg0["l_max"]),
              TE.build_split(ds, "val", cfg0["l_max"]),
              TE.build_split(ds, "test", cfg0["l_max"]))
    cands = build_eval_candidates(ds, n_neg=100, seed=42)
    pop = np.zeros(len(ds["items"]))
    for u, v in ds["train_targets"].items():
        for (i, _, _) in v:
            pop[i] += 1

    summary = {}
    for vname in variants:
        over = VARIANTS[vname]
        cfg = dict(cfg0)
        cfg.update(over)
        feats = TE.load_features(a.dataset, cfg)
        # a removed modality must not be silently replaced by zeros
        if not cfg.get("use_img", True):
            feats = (None, feats[1], None, 768, feats[4])
        if not cfg.get("use_txt", True):
            feats = (feats[0], None, feats[2], feats[3], 1024)
        neg = TE.MixedNegatives(ds, splits[0], k_pop=cfg["k_pop"], k_sem=cfg["k_sem"],
                                uniform=bool(cfg.get("uniform_negatives")))

        runs, n_params = [], 0
        for sd in seeds:
            c = dict(cfg)
            c["seed"] = sd
            print(f"\n-- variant {vname} seed={sd} --", flush=True)
            m, info = TE.train(ds, c, feats=feats, verbose=True, max_seconds=a.budget,
                               neg_sampler=neg, train_split=splits[0], val_split=splits[1])
            n_params = info["n_params"]
            res = TE.evaluate(m, splits[2], ds, cand_sets=cands["test"]["items"],
                              return_lists=True)
            V = m.candidate_vectors().detach().cpu().numpy()
            lm = list_metrics(res["topk"], pop, len(ds["items"]), item_repr=V, k=10)
            runs.append({"seed": sd, "test_full": res["full"], "test_sampled": res["sampled"],
                         "list_metrics": lm, "history": info["history"],
                         "n_params": info["n_params"], "train_seconds": info["train_seconds"],
                         "ranks_full": res["ranks_full"].tolist(),
                         "sigma": res["sigma"].tolist(), "topk": res["topk"].tolist(),
                         "test_users": splits[2].users})
            print(f"  >> {vname} s{sd}: R@10={res['full']['Recall@10']:.4f} "
                  f"NDCG@10={res['full']['NDCG@10']:.4f} MRR@10={res['full']['MRR@10']:.4f}",
                  flush=True)
        agg = {k: {"mean": float(np.mean([r["test_full"][k] for r in runs])),
                   "std": float(np.std([r["test_full"][k] for r in runs]))}
               for k in runs[0]["test_full"]}
        lm_agg = {k: float(np.mean([r["list_metrics"][k] for r in runs]))
                  for k in runs[0]["list_metrics"]}
        rec = {"variant": vname, "overrides": over, "dataset": a.dataset, "config": cfg,
               "seeds": seeds, "runs": runs, "aggregate_full": agg, "list_metrics": lm_agg,
               "n_params": n_params}
        save_json(rec, os.path.join(RESULTS, f"ablation__{vname}__{a.dataset}.json"))
        summary[vname] = {"full": agg, "list": lm_agg, "params": n_params}
        print(f"== {vname}: R@10={agg['Recall@10']['mean']:.4f} "
              f"NDCG@10={agg['NDCG@10']['mean']:.4f}", flush=True)

    save_json(summary, os.path.join(RESULTS, f"_summary_ablation__{a.dataset}.json"))
    print(f"\n=== ABLATION SUMMARY {a.dataset} ===")
    base = summary.get("full", {}).get("full", {}).get("Recall@10", {}).get("mean")
    for k, v in summary.items():
        r = v["full"]["Recall@10"]["mean"]
        d = (r - base) / base * 100 if base else 0.0
        print(f"{k:15s} R@10={r:.4f} NDCG@10={v['full']['NDCG@10']['mean']:.4f} "
              f"MRR@10={v['full']['MRR@10']['mean']:.4f} ({d:+.2f}% vs full)")


if __name__ == "__main__":
    main()
