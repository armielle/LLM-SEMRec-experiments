#!/usr/bin/env bash
# Catch-up: the multimodal/LLM models of MovieLens-1M failed on the first pass
# because a dataset without images produced a width-1 placeholder against a
# 768-wide concat layer (fixed in baselines.SASRec).  Re-run only those models.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export NTHREADS=4
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== MovieLens-1M: multimodal + LLM models ==="
$PY -u code/run_main.py movielens_1m mmsasrec,llmonly,mmssl,llmsemrec --seeds 0 \
    --epochs 20 --patience 5 --budget 1500 --lr 1e-3 >> logs/main_ml1m.log 2>&1
grep -E "^== " logs/main_ml1m.log | tail -4

log "=== analyses (MovieLens-1M) ==="
$PY -u code/run_analysis.py movielens_1m --models llmsemrec,sasrec >> logs/analysis_ml1m.log 2>&1

log "=== deliverables ==="
$PY -u code/make_figures.py >> logs/figures.log 2>&1
$PY -u code/make_latex.py   >> logs/latex.log   2>&1
$PY -u code/make_report.py  >> logs/report.log  2>&1
log "=== CATCH-UP COMPLETE ==="
