#!/usr/bin/env bash
# Round 2 of the experiments:
#   A. MovieLens-1M  -- completes a second dataset for Table III
#   B. uniform-negative ablation round -- tests whether the components that looked
#      harmful in Table V only look harmful because the Eq. 53 sampler dominates
#   C. refresh figures / LaTeX tables / report after every stage
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export NTHREADS=4
log() { echo "[$(date +%H:%M:%S)] $*"; }
emit() {
  $PY -u code/make_figures.py >> logs/figures.log 2>&1
  $PY -u code/make_latex.py   >> logs/latex.log   2>&1
  $PY -u code/make_report.py  >> logs/report.log  2>&1
  log "artefacts refreshed"
}

# ------------------------------------------------------------ A. MovieLens-1M
log "=== A1. Qwen3 text embeddings (MovieLens-1M) ==="
if [ ! -s cache/movielens_1m_qwen3_0.6b.npy ]; then
  $PY -u code/encode_text.py --dataset movielens_1m --max-tokens 96 --batch 64 \
      --threads $NTHREADS >> logs/encode_text_ml1m.log 2>&1
fi
log "text embeddings ready"

log "=== A2. all models on MovieLens-1M ==="
$PY -u code/run_main.py movielens_1m all --seeds 0 --epochs 20 --patience 5 \
    --budget 1200 --lr 1e-3 >> logs/main_ml1m.log 2>&1
emit

log "=== A3. section VII analyses (MovieLens-1M) ==="
$PY -u code/run_analysis.py movielens_1m --models llmsemrec,sasrec >> logs/analysis_ml1m.log 2>&1
emit

# ------------------------------------------------------------ B. uniform-neg ablation
log "=== B. ablation on top of uniform negatives ==="
for v in uneg_wo_qwen3 uneg_wo_ssl uneg_wo_vae uneg_simple_fusion; do
  log "variant $v"
  $PY -u code/run_ablation.py amazon_beauty --variants "$v" --seeds 0 \
      --epochs 8 --patience 3 --budget 1500 >> logs/ablation_beauty_round2.log 2>&1
  grep -E "^== " logs/ablation_beauty_round2.log | tail -1
  emit
done
log "=== ROUND 2 COMPLETE ==="
