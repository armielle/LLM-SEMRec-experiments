#!/usr/bin/env bash
# Small hyperparameter search for LLM-SEMRec on Amazon Beauty (paper Eq. 59-60).
#
# The first full run showed the training loss collapsing while validation NDCG@10
# degraded after epoch 1 -- classic overfitting of the 10-negative sampled
# softmax by the collaborative ID path.  This sweep varies the two knobs that
# control it (learning rate and dropout) and reports validation NDCG@10 per
# epoch so the best setting can be picked before the long runs.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export NTHREADS=4

run() {  # run <lr> <dropout> <epochs>
  local lr="$1" dp="$2" ep="$3"
  local tag="sweep_lr${lr}_dp${dp}"
  echo "### sweep lr=$lr dropout=$dp epochs=$ep"
  $PY -u code/run_main.py amazon_beauty llmsemrec --seeds 0 --epochs "$ep" --patience "$ep" \
      --budget 1500 --lr "$lr" --dropout "$dp" --tag "$tag" --force \
      > "logs/sweep_${tag}.log" 2>&1
  echo "--- $tag ---"
  grep -E "^  ep" "logs/sweep_${tag}.log" | tail -6
}

run 1e-3 0.2 5
run 3e-4 0.3 5
run 1e-4 0.4 5
echo "### SWEEP DONE"
