#!/usr/bin/env bash
# End-to-end reproduction of the LLM-SEMRec experiments.
#
# Usage:  bash code/run_plan.sh [dataset ...]
#
# Stage 0  raw downloads            (SNAP Amazon 5-core, MovieLens-1M)
# Stage 1  preprocessing            5-core + leave-two-out + item text profiles
# Stage 2  item images              legacy Amazon image endpoint + validation
# Stage 3  frozen encoders          Qwen3-Embedding-0.6B + SigLIP 2 base -> cached
# Stage 4  main comparison          Table III
# Stage 5  ablation                 Tables IV-V
# Stage 6  analyses VII-C..VII-F
# Stage 7  figures + LaTeX tables
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
DATASETS="${*:-amazon_beauty amazon_fashion movielens_1m}"
mkdir -p logs results figures tables

raw_urls=(
  "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz:beauty_reviews.json.gz"
  "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz:beauty_meta.json.gz"
  "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz:cloth_reviews.json.gz"
  "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz:cloth_meta.json.gz"
)
echo "### Stage 0: raw data"
for f in "${raw_urls[@]}"; do
  src="${f%%:*}"; dst="raw/${f##*:}"
  [ -s "$dst" ] || curl -sL --retry 3 -o "$dst" "$src"
  echo "  $dst $(du -h "$dst" | cut -f1)"
done
[ -s raw/ml-1m.zip ] || curl -sL --retry 3 -o raw/ml-1m.zip \
  https://files.grouplens.org/datasets/movielens/ml-1m.zip

echo "### Stage 1: preprocessing"
for d in $DATASETS; do
  case $d in
    amazon_beauty) key=beauty;; amazon_fashion) key=fashion;; movielens_1m) key=ml1m;;
  esac
  [ -s "data/$d.pkl" ] || $PY -u code/prep.py "$key" 2>&1 | tee "logs/prep_$d.log"
done

echo "### Stage 2: item images"
for d in $DATASETS; do
  [ -d "data/images/$d" ] && [ "$(ls data/images/$d | wc -l)" -gt 100 ] || \
    $PY -u code/download_images.py "$d" 24 2>&1 | tee "logs/images_$d.log"
done

echo "### Stage 3: frozen encoders"
for d in $DATASETS; do
  [ -s "cache/${d}_qwen3_0.6b.npy" ] || \
    $PY -u code/encode_text.py --dataset "$d" --max-tokens 96 --batch 48 --threads 8 \
        2>&1 | tee "logs/encode_text_$d.log"
  [ -s "cache/${d}_siglip2_base.npy" ] || \
    $PY -u code/encode_image.py --dataset "$d" --batch 32 --threads 8 \
        2>&1 | tee "logs/encode_image_$d.log"
done

echo "### Stage 4: main comparison (Table III)"
for d in $DATASETS; do
  NTHREADS=8 $PY -u code/run_main.py "$d" all --seeds 0 --epochs 30 --patience 6 \
    --budget 2400 2>&1 | tee "logs/main_$d.log"
done

echo "### Stage 5: ablation (Tables IV-V)"
NTHREADS=8 $PY -u code/run_ablation.py amazon_beauty --seeds 0 --epochs 20 --patience 5 \
  --budget 1800 2>&1 | tee logs/ablation_beauty.log

echo "### Stage 6: analyses (VII-C..VII-F)"
for d in $DATASETS; do
  NTHREADS=8 $PY -u code/run_analysis.py "$d" --models llmsemrec,sasrec,llmonly \
    2>&1 | tee "logs/analysis_$d.log"
done

echo "### Stage 7: figures + LaTeX tables"
$PY -u code/make_figures.py 2>&1 | tee logs/figures.log
$PY -u code/make_latex.py 2>&1 | tee logs/latex.log
$PY -u code/make_report.py 2>&1 | tee logs/report.log
echo "### DONE"
