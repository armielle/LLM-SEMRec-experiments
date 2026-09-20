#!/usr/bin/env bash
# Sequential, low-impact execution plan for the LLM-SEMRec experiments.
#
# ONE heavy process at a time, bounded thread count, artefacts written
# incrementally so an interruption never loses completed work.  Figures, LaTeX
# tables and the report are regenerated after every stage: whatever has been
# computed is always available as a deliverable.
#
# Usage: bash code/run_plan_light.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export NTHREADS=6
mkdir -p logs results figures tables models

log() { echo "[$(date +%H:%M:%S)] $*"; }

wait_for() {   # wait_for <file> <label> [max_minutes]
  local f="$1" lbl="$2" mx="${3:-240}" i=0
  while [ ! -s "$f" ]; do
    i=$((i+1)); sleep 30
    if [ $((i % 10)) -eq 0 ]; then log "waiting for $lbl ($((i/2)) min)"; fi
    if [ $i -gt $((mx*2)) ]; then log "TIMEOUT waiting for $lbl"; return 1; fi
  done
  log "$lbl ready"
}

emit() {   # regenerate the deliverables from whatever results exist
  $PY -u code/make_figures.py >> logs/figures.log 2>&1
  $PY -u code/make_latex.py   >> logs/latex.log   2>&1
  $PY -u code/make_report.py  >> logs/report.log  2>&1
  log "artefacts refreshed (figures/, tables/, REPORT.md)"
}

# ------------------------------------------------- stage 3a: Qwen3 text (Beauty)
wait_for cache/amazon_beauty_qwen3_0.6b.npy "Qwen3 embeddings (Beauty)" 240

# ------------------------------------------------- stage 3b: SigLIP2 images (Beauty)
if [ ! -s cache/amazon_beauty_siglip2_base.npy ]; then
  log "encoding SigLIP2 images (Beauty)"
  $PY -u code/encode_image.py --dataset amazon_beauty --batch 32 --threads $NTHREADS \
      >> logs/encode_image_beauty.log 2>&1
fi
log "frozen encoders for Amazon Beauty are ready"

# ------------------------------------------------- stage 4a: the proposed model (headline)
log "=== stage 4a: LLM-SEMRec on Amazon Beauty ==="
$PY -u code/run_main.py amazon_beauty llmsemrec --seeds 0,1 --epochs 25 --patience 5 \
    --budget 4200 >> logs/main_llmsemrec_beauty.log 2>&1
emit

# ------------------------------------------------- stage 4b: cheap baselines
log "=== stage 4b: ID-only baselines (Amazon Beauty) ==="
$PY -u code/run_main.py amazon_beauty popularity,bprmf,gru4rec,caser,sasrec,bert4rec \
    --seeds 0 --epochs 25 --patience 5 --budget 1200 >> logs/main_id_beauty.log 2>&1
$PY -u code/run_main.py amazon_beauty cl4srec,s3rec --seeds 0 --epochs 20 --patience 5 \
    --budget 1500 >> logs/main_ssl_beauty.log 2>&1
emit

# ------------------------------------------------- stage 4c: multimodal / LLM baselines
log "=== stage 4c: multimodal baselines (Amazon Beauty) ==="
$PY -u code/run_main.py amazon_beauty mmsasrec,llmonly,mmssl --seeds 0 --epochs 18 \
    --patience 5 --budget 2000 >> logs/main_mm_beauty.log 2>&1
emit

# ------------------------------------------------- stage 6a: analyses on Beauty
log "=== stage 6a: VII-C..VII-F analyses (Amazon Beauty) ==="
$PY -u code/run_analysis.py amazon_beauty --models llmsemrec,sasrec,llmonly \
    >> logs/analysis_beauty.log 2>&1
emit

# ------------------------------------------------- stage 4d: MovieLens-1M (cheap dataset)
log "=== stage 4d: MovieLens-1M ==="
if [ ! -s cache/movielens_1m_qwen3_0.6b.npy ]; then
  $PY -u code/encode_text.py --dataset movielens_1m --max-tokens 96 --batch 64 \
      --threads $NTHREADS >> logs/encode_text_ml1m.log 2>&1
fi
$PY -u code/run_main.py movielens_1m all --seeds 0 --epochs 25 --patience 5 --budget 1500 \
    >> logs/main_ml1m.log 2>&1
$PY -u code/run_analysis.py movielens_1m --models llmsemrec,sasrec >> logs/analysis_ml1m.log 2>&1
emit

# ------------------------------------------------- stage 5: ablation
log "=== stage 5: ablation (Amazon Beauty) ==="
$PY -u code/run_ablation.py amazon_beauty --seeds 0 --epochs 12 --patience 4 --budget 1500 \
    >> logs/ablation_beauty.log 2>&1
emit

# ------------------------------------------------- stage 4e: Amazon Fashion (optional)
log "=== stage 4e: Amazon Fashion ==="
if [ ! -s cache/amazon_fashion_siglip2_base.npy ] && [ -s cache/amazon_fashion_qwen3_0.6b.npy ]; then
  $PY -u code/encode_image.py --dataset amazon_fashion --batch 32 --threads $NTHREADS \
      >> logs/encode_image_fashion.log 2>&1
fi
if [ -s cache/amazon_fashion_qwen3_0.6b.npy ]; then
  $PY -u code/run_main.py amazon_fashion llmsemrec,sasrec,popularity --seeds 0 --epochs 20 \
      --patience 5 --budget 3600 >> logs/main_fashion.log 2>&1
fi
emit

log "=== PLAN COMPLETE ==="
