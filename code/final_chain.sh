#!/usr/bin/env bash
# Final chain: repaired GRU4Rec baseline, section VII analyses, deliverables.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export NTHREADS=4
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== GRU4Rec (fixed readout) ==="
$PY -u code/run_main.py amazon_beauty gru4rec --seeds 0 --epochs 18 --patience 5 \
    --budget 900 --lr 1e-3 --force >> logs/grubaseline.log 2>&1
grep -E "^== " logs/grubaseline.log | tail -1

log "=== Section VII analyses (Amazon Beauty) ==="
$PY -u code/run_analysis.py amazon_beauty --models llmsemrec,sasrec,llmonly \
    >> logs/analysis_beauty.log 2>&1
ls -la results/analysis__amazon_beauty.json 2>/dev/null || log "analysis FAILED"

log "=== figures / LaTeX tables / report ==="
$PY -u code/make_figures.py >> logs/figures.log 2>&1
$PY -u code/make_latex.py   >> logs/latex.log   2>&1
$PY -u code/make_report.py  >> logs/report.log  2>&1
log "=== FINAL CHAIN COMPLETE ==="
