"""Single-step runner: trains ONE model on ONE dataset, then exits.

Designed for a low-impact workflow: one model at a time, bounded threads,
below-normal OS priority, everything appended to results/ so a restart simply
skips whatever is already done.

  python run_step.py <dataset> <model> [--epochs N] [--budget S] [--seeds 0]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nice  # noqa: F401  (sets thread caps + low OS priority)

import numpy as np
import torch

torch.set_num_threads(nice.threads())

import train_eval as TE
from common import build_eval_candidates, list_metrics, load_ds, save_json
from paths import RESULTS

ALL = ["popularity", "bprmf", "gru4rec", "caser", "sasrec", "bert4rec", "cl4srec",
       "s3rec", "mmsasrec", "llmonly", "mmssl", "llmsemrec"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("model")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--budget", type=int, default=2400)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--lmax", type=int, default=20)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    outpath = os.path.join(RESULTS, f"{a.model}__{a.dataset}.json")
    if os.path.exists(outpath) and not a.force:
        print(f"[skip] {a.model} on {a.dataset} already done")
        return

    # run_main owns the model registry and the training loop
    sys.argv = ["run_main.py", a.dataset, a.model, "--seeds", a.seeds,
                "--epochs", str(a.epochs), "--patience", str(a.patience),
                "--budget", str(a.budget), "--lmax", str(a.lmax), "--d", str(a.d)]
    import run_main
    run_main.main()


if __name__ == "__main__":
    main()
