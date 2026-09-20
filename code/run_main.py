"""
Main experiment driver -- Table III (overall performance) of the paper.

Usage:
  python run_main.py <dataset> <model[,model...]|all> [--seeds 0,1,2] [--epochs N]
                     [--budget SECONDS_PER_MODEL] [--lmax 20] [--d 128]

Each model writes results/<model>__<dataset>.json as soon as it finishes, so the
driver is resumable and models that are already computed are skipped.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nice  # noqa: F401  (low OS priority + bounded BLAS threads)

import baselines as B
import train_eval as TE
from common import (RESULTS, build_eval_candidates, list_metrics, load_ds,
                    paired_ttest, per_user_hits, save_json)
from paths import RESULTS as RES_DIR

torch.set_num_threads(nice.threads())

ALL_MODELS = ["popularity", "bprmf", "gru4rec", "caser", "sasrec", "bert4rec",
              "cl4srec", "s3rec", "mmsasrec", "llmonly", "mmssl", "llmsemrec"]


def base_cfg(a, **over):
    c = dict(TE.DEFAULT_CFG)
    c.update(dict(l_max=a.lmax, d=a.d, dz=max(a.d // 2, 32), epochs=a.epochs,
                  patience=a.patience, batch=a.batch))
    c.update(over)
    return c


def build(name, ds, cfg, feats):
    n_items = len(ds["items"])
    n_users = len(ds["seqs"])
    n_bins = int(ds["stats"]["n_bins"])
    img, txt, avail, d_img, d_txt = feats
    common = dict(n_items=n_items, n_bins=n_bins, dz=cfg["dz"], d=cfg["d"],
                  l_max=cfg["l_max"], dropout=cfg["dropout"])
    if name == "gru4rec":
        return B.GRU4Rec(**common), "next", ()
    if name == "caser":
        return B.Caser(**common), "next", ()
    if name == "sasrec":
        return B.SASRec(**common, n_heads=cfg["n_heads"], n_layers=cfg["n_tf_layers"],
                        ff=cfg["ff"]), "next", ()
    if name == "bert4rec":
        return B.BERT4Rec(**common, n_heads=cfg["n_heads"], n_layers=cfg["n_tf_layers"],
                          ff=cfg["ff"]), "mip", ()
    if name == "cl4srec":
        return B.SASRec(**common, n_heads=cfg["n_heads"], n_layers=cfg["n_tf_layers"],
                        ff=cfg["ff"]), "next", ("seqcl",)
    if name == "s3rec":
        return B.SASRec(**common, n_heads=cfg["n_heads"], n_layers=cfg["n_tf_layers"],
                        ff=cfg["ff"]), "next", ("mip", "seqcl")
    if name == "mmsasrec":
        return B.SASRec(**common, n_heads=cfg["n_heads"], n_layers=cfg["n_tf_layers"],
                        ff=cfg["ff"], content="concat", d_img=d_img, d_txt=d_txt), "next", ()
    if name == "llmonly":
        return B.SASRec(**common, n_heads=cfg["n_heads"], n_layers=cfg["n_tf_layers"],
                        ff=cfg["ff"], content="txt", d_img=d_img, d_txt=d_txt), "next", ()
    if name == "mmssl":
        return B.SASRec(**common, n_heads=cfg["n_heads"], n_layers=cfg["n_tf_layers"],
                        ff=cfg["ff"], content="gated", d_img=d_img, d_txt=d_txt), "next", ("cmcl",)
    if name == "bprmf":
        return B.BPRMF(n_items=n_items, n_users=n_users, dz=cfg["dz"]), "bpr", ()
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("models", default="all")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--budget", type=int, default=2700, help="seconds per model")
    ap.add_argument("--lmax", type=int, default=20)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--k-sem", type=int, default=5)
    ap.add_argument("--seqcl-every", type=int, default=2)
    ap.add_argument("--lam-mip", type=float, default=None)
    ap.add_argument("--lam-seqcl", type=float, default=None)
    ap.add_argument("--lam-cmcl", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--min-hist", type=int, default=0,
                    help="supervise L_Rec only from prefixes with >= this many interactions")
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    models = ALL_MODELS if a.models == "all" else a.models.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    ds = load_ds(a.dataset)
    cfg = base_cfg(a)
    for k in ("lr", "dropout", "wd", "seqcl_every", "k_sem"):
        cfg[k] = getattr(a, k if k != "k_sem" else "k_sem")
    for k, v in (("lam_mip", a.lam_mip), ("lam_seqcl", a.lam_seqcl),
                 ("lam_cmcl", a.lam_cmcl), ("beta", a.beta)):
        if v is not None:
            cfg[k] = v
    cfg["min_hist"] = a.min_hist
    L = cfg["l_max"]

    print(f"### MAIN {a.dataset}: {len(ds['items']):,} items, "
          f"{len(ds['seqs']):,} users, L={L}, d={cfg['d']}, dz={cfg['dz']}", flush=True)
    t0 = time.time()
    splits = (TE.build_split(ds, "train", L), TE.build_split(ds, "val", L),
              TE.build_split(ds, "test", L))
    feats = TE.load_features(a.dataset, cfg)
    print(f"  features: img={None if feats[0] is None else tuple(feats[0].shape)} "
          f"txt={None if feats[1] is None else tuple(feats[1].shape)}", flush=True)
    neg = TE.MixedNegatives(ds, splits[0], k_pop=cfg["k_pop"], k_sem=cfg["k_sem"])
    cands = build_eval_candidates(ds, n_neg=100, seed=42)
    assert cands["test"]["users"] == splits[2].users
    pop = np.zeros(len(ds["items"]))
    for u, v in ds["train_targets"].items():
        for (i, _, _) in v:
            pop[i] += 1
    print(f"  data ready in {time.time()-t0:.0f}s", flush=True)

    summary = {}
    suffix = f"__{a.tag}" if a.tag else ""
    for name in models:
        outpath = os.path.join(RES_DIR, f"{name}__{a.dataset}{suffix}.json")
        if os.path.exists(outpath) and not a.force:
            print(f"  [skip] {name} (already done)", flush=True)
            try:
                r = json.load(open(outpath))
                summary[name] = {"full": r["aggregate_full"], "sampled": r["aggregate_sampled"],
                                 "seconds": r["wall_seconds"], "params": r["n_params"]}
            except Exception:
                pass
            continue
        print(f"\n-- {name} --", flush=True)
        t1 = time.time()
        if name == "popularity":
            res = TE.evaluate(PopularityModel(pop), splits[2], ds, cand_sets=cands["test"]["items"],
                              return_lists=True)
            pops = ~np.isfinite(pop)
            V = None
            runs = [{"seed": 0, "test_full": res["full"], "test_sampled": res["sampled"],
                     "sigma_mean": 0.0, "list_metrics": {}, "history": [],
                     "n_params": 0, "train_seconds": 0.0,
                     "ranks_full": res["ranks_full"].tolist(),
                     "sigma": res["sigma"].tolist(), "topk": res["topk"].tolist(),
                     "test_users": splits[2].users}]
            n_params = 0
            wall = time.time() - t1
        else:
            runs, n_params = [], 0
            for sd in seeds:
                c = dict(cfg)
                c["seed"] = sd
                if name == "llmsemrec":
                    m = TE.build_model(ds, c, feats)
                    m, info = TE.train(ds, c, feats=feats, verbose=True,
                                       max_seconds=a.budget, neg_sampler=neg,
                                       train_split=splits[0], val_split=splits[1])
                    objective = "next"
                else:
                    m, objective, aux = build(name, ds, c, feats)
                    if objective == "bpr":
                        m, info = B.fit_bpr(m, ds, (splits[0], splits[1]), c, name=name,
                                            max_seconds=a.budget, verbose=True)
                    else:
                        m, info = B.fit(m, ds, c, (splits[0], splits[1]), feats=feats,
                                        neg_sampler=neg, objective=objective, aux=aux,
                                        name=name, max_seconds=a.budget, verbose=True)
                n_params = info["n_params"]
                res = TE.evaluate(m, splits[2], ds, cand_sets=cands["test"]["items"],
                                  return_lists=True)
                V = m.candidate_vectors().detach().cpu().numpy()
                lm = list_metrics(res["topk"], pop, len(ds["items"]), item_repr=V, k=10)
                if sd == seeds[0]:
                    mdir = os.path.join(RES_DIR, "..", "models")
                    os.makedirs(mdir, exist_ok=True)
                    torch.save({"state": m.state_dict(), "cfg": c, "name": name,
                                "dataset": a.dataset},
                               os.path.join(mdir, f"{name}__{a.dataset}{suffix}.pt"))
                    val_res = TE.evaluate(m, splits[1], ds, cand_sets=cands["val"]["items"],
                                          return_lists=True)
                    np.savez_compressed(
                        os.path.join(mdir, f"eval__{name}__{a.dataset}{suffix}.npz"),
                        test_ranks=res["ranks_full"], test_sigma=res["sigma"],
                        test_scores=res["sampled_scores"],
                        val_ranks=val_res["ranks_full"], val_sigma=val_res["sigma"],
                        val_scores=val_res["sampled_scores"],
                        test_topk=res["topk"], test_targets=splits[2].tgt[:, -1],
                        test_items=cands["test"]["items"], val_items=cands["val"]["items"])
                runs.append({"seed": sd, "test_full": res["full"],
                             "test_sampled": res["sampled"],
                             "sigma_mean": res["sigma_mean"], "list_metrics": lm,
                             "history": info["history"], "n_params": info["n_params"],
                             "train_seconds": info["train_seconds"],
                             "ranks_full": res["ranks_full"].tolist(),
                             "sigma": res["sigma"].tolist(),
                             "topk": res["topk"].tolist(),
                             "test_users": splits[2].users})
                print(f"  >> {name} seed{sd} full R@10={res['full']['Recall@10']:.4f} "
                      f"NDCG@10={res['full']['NDCG@10']:.4f}", flush=True)
            wall = time.time() - t1
        agg_f = {k: {"mean": float(np.mean([r["test_full"][k] for r in runs])),
                     "std": float(np.std([r["test_full"][k] for r in runs]))}
                 for k in runs[0]["test_full"]}
        agg_s = {k: {"mean": float(np.mean([r["test_sampled"][k] for r in runs])),
                     "std": float(np.std([r["test_sampled"][k] for r in runs]))}
                 for k in runs[0]["test_sampled"]}
        lm_agg = {}
        if runs[0]["list_metrics"]:
            for k in runs[0]["list_metrics"]:
                lm_agg[k] = float(np.mean([r["list_metrics"][k] for r in runs]))
        rec = {"model": name, "dataset": a.dataset, "config": cfg, "seeds": seeds,
               "n_params": n_params, "wall_seconds": round(wall, 1),
               "runs": runs, "aggregate_full": agg_f, "aggregate_sampled": agg_s,
               "list_metrics": lm_agg}
        save_json(rec, outpath)
        summary[name] = {"full": agg_f, "sampled": agg_s, "seconds": wall,
                         "params": n_params, "list": lm_agg}
        print(f"== {name}: R@10={agg_f['Recall@10']['mean']:.4f} "
              f"NDCG@10={agg_f['NDCG@10']['mean']:.4f} MRR@10={agg_f['MRR@10']['mean']:.4f} "
              f"({wall:.0f}s)", flush=True)

    save_json(summary, os.path.join(RES_DIR, f"_summary_main__{a.dataset}.json"))
    print("\n=== SUMMARY", a.dataset, "===")
    for k, v in sorted(summary.items(), key=lambda x: -x[1]["full"]["NDCG@10"]["mean"]):
        print(f"{k:12s} R@10={v['full']['Recall@10']['mean']:.4f} "
              f"NDCG@10={v['full']['NDCG@10']['mean']:.4f} "
              f"MRR@10={v['full']['MRR@10']['mean']:.4f} "
              f"R@20={v['full']['Recall@20']['mean']:.4f} "
              f"NDCG@20={v['full']['NDCG@20']['mean']:.4f} "
              f"params={v['params']:,}")


class PopularityModel:
    """Non-personalized most-popular baseline (ranked by training frequency)."""

    def __init__(self, pop):
        self.pop = pop
        self.n_items = len(pop)
        self.dz = 1

    def eval(self):
        return self

    def train(self):
        return self

    def invalidate_candidates(self):
        pass

    def candidate_vectors(self, use_cache=True):
        return torch.from_numpy(self.pop.reshape(-1, 1).astype(np.float32))

    def encode_sequence(self, idx, rbin, tb, padmask):
        B = idx.shape[0]
        return {"mu": torch.ones(B, 1), "sig": torch.ones(B, 1), "z": torch.ones(B, 1),
                "kl": torch.zeros(B)}

    @staticmethod
    def score(mu, sig, V, gamma=0.0):
        return mu[:, :1] @ V.t()


if __name__ == "__main__":
    main()
