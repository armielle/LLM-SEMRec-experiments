#!/usr/bin/env bash
# Ablation study (Tables IV-V), then refresh the deliverables.
# Each variant is written to results/ as soon as it finishes.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export NTHREADS=4
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== ablation (Amazon Beauty) ==="
for v in full wo_qwen3 wo_siglip wo_ssl wo_vae simple_fusion uniform_neg; do
  log "variant $v"
  $PY -u code/run_ablation.py amazon_beauty --variants "$v" --seeds 0 \
      --epochs 8 --patience 3 --budget 1500 >> logs/ablation_beauty.log 2>&1
  grep -E "^== " logs/ablation_beauty.log | tail -1
  $PY -u code/make_figures.py >> logs/figures.log 2>&1
  $PY -u code/make_latex.py   >> logs/latex.log   2>&1
  $PY -u code/make_report.py  >> logs/report.log  2>&1
done
log "=== ABLATION COMPLETE ==="
