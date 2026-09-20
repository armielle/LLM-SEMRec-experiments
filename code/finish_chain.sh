#!/usr/bin/env bash
# Finishing chain: repair the GRU4Rec baseline, run the section VII analyses,
# then refresh figures / LaTeX tables / report.  One process at a time.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export NTHREADS=4
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== GRU4Rec (fixed last-valid-position readout) ==="
$PY -u code/run_main.py amazon_beauty gru4rec --seeds 0 --epochs 18 --patience 5 \
    --budget 900 --lr 1e-3 >> logs/grubaseline.log 2>&1

log "=== Section VII analyses (Amazon Beauty) ==="
$PY -u code/run_analysis.py amazon_beauty --models llmsemrec,sasrec \
    >> logs/analysis_beauty.log 2>&1

log "=== figures / tables / report ==="
$PY -u code/make_figures.py >> logs/figures.log 2>&1
$PY -u code/make_latex.py   >> logs/latex.log   2>&1
$PY -u code/make_report.py  >> logs/report.log  2>&1
log "artefacts refreshed"

log "=== multimodal + LLM baselines (Amazon Beauty) ==="
$PY -u code/run_main.py amazon_beauty mmsasrec,llmonly --seeds 0 --epochs 12 --patience 4 \
    --budget 1800 --lr 3e-4 >> logs/mm_baselines.log 2>&1
$PY -u code/run_analysis.py amazon_beauty --models llmsemrec,sasrec,llmonly \
    >> logs/analysis_beauty.log 2>&1
$PY -u code/make_figures.py >> logs/figures.log 2>&1
$PY -u code/make_latex.py   >> logs/latex.log   2>&1
$PY -u code/make_report.py  >> logs/report.log  2>&1
log "=== FINISH CHAIN COMPLETE ==="
