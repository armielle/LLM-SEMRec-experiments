# LLM-SEMRec — reproduction on a CPU-only host

Experimental reproduction of the **LLM-SEMRec** architecture (frozen LLM and
vision encoders + hybrid temporal encoder + variational module +
self-supervised objectives) on three public benchmarks, run end to end on a
machine **without a GPU**.

Every number in this repository comes from an execution that actually ran.
No value is estimated, inferred, or copied from the paper.

---

## Headline results

Full-catalogue ranking, chronological leave-two-out split.

### Amazon Beauty — 22 363 users / 12 101 items / 198 502 interactions

| Model | Recall@10 | NDCG@10 | MRR@10 |
|---|---|---|---|
| **LLM-SEMRec** | **0.0193** | **0.0093** | **0.0063** |
| BPR-MF | 0.0158 | 0.0063 | 0.0035 |
| Popularity | 0.0120 | 0.0055 | 0.0036 |
| BERT4Rec | 0.0113 | 0.0056 | 0.0038 |
| MM-SASRec | 0.0113 | 0.0053 | 0.0035 |
| LLM-only | 0.0110 | 0.0049 | 0.0031 |
| SASRec | 0.0095 | 0.0047 | 0.0033 |
| GRU4Rec | 0.0012 | 0.0005 | 0.0003 |
| Caser | 0.0011 | 0.0005 | 0.0003 |

Best on all three metrics. Significantly better than **every** baseline
(paired *t*-test on per-user Hit@10: *p* from 2.6e-3 vs BPR-MF to 3.6e-82 vs
Caser). Median target rank 1 604 vs 3 905 for SASRec.

### MovieLens-1M — 6 040 users / 3 416 items / 999 611 interactions

| Model | Recall@10 | NDCG@10 | MRR@10 |
|---|---|---|---|
| **LLM-SEMRec** | 0.0334 | **0.0160** | **0.0107** |
| BPR-MF | **0.0366** | 0.0157 | 0.0095 |
| LLM-only | 0.0303 | 0.0136 | 0.0087 |
| MM-SASRec | 0.0270 | 0.0129 | 0.0086 |
| BERT4Rec | 0.0252 | 0.0125 | 0.0087 |
| SASRec | 0.0237 | 0.0111 | 0.0073 |

Best NDCG@10 and MRR@10. BPR-MF keeps a Recall@10 advantage on this dense
dataset, and the difference against BPR-MF (*p* = 0.30) and LLM-only
(*p* = 0.29) is **not** significant in Recall@10.

---

## What the ablation actually shows

Two negative-sampling regimes, Amazon Beauty, relative change in Recall@10.

| Variant | Mixed negatives | Uniform negatives |
|---|---|---|
| Full model | 0.0194 | 0.0332 |
| w/o self-supervised objectives | **−47.9 %** | **−28.3 %** |
| w/o Qwen3 text encoder | **−25.8 %** | **−22.9 %** |
| w/o SigLIP image encoder | **−25.8 %** | — |
| w/o variational module | +30.4 % | +5.1 % |
| simple concatenation fusion | +24.2 % | +15.4 % |
| uniform negatives | +71.1 % | (ref.) |

Read the sign carefully: a **negative** Δ means removing the component hurts
(so the component helps). A **positive** Δ means removing it *improves* the
model — the component is not earning its place.

**Supported:** the frozen LLM encoder, the frozen vision encoder, and the
self-supervised objectives. All three degrade performance consistently across
both regimes.

**Not supported:** the variational module, the reliability-aware gated fusion,
and mixed negative sampling. All three are outperformed by their removal.

We instrumented the gate to find out why: it places **90.0 %** of its weight on
the item identifier and only 2–3 % on each content modality, with no adaptation
to item popularity. The "reliability-aware gated multimodal fusion" degenerates
in practice into an identifier model — which is exactly why replacing it with
concatenation helps.

---

## Where the content signal does pay off

| Evidence | Number |
|---|---|
| Cold-target Recall@10, LLM-SEMRec vs SASRec / BPR-MF | 0.0037 vs **0.0000** (no hit at all over 7 329 users) |
| Gain over SASRec on medium-popularity targets | **27.0×** |
| 10-NN category purity, items with 1–2 interactions: identifier embedding vs frozen LLM | 0.111 (chance = 0.082) vs **0.566** (5.1× chance) |
| Long-tail share of recommendations (items with ≤10 interactions) | **31.1 %** vs 4.0 % for SASRec |
| Per-user AUC on 1+100 candidate sets | **0.752** vs 0.618 for SASRec |
| First place across time-drift windows | all 5, max gain 2.4× over SASRec at 1–3 years |

The advantage is largest precisely where identifier signals are weakest, which
is the behaviour the cold-start argument predicts.

---

## Findings that contradict the paper's claims

These are reported rather than hidden. Each one also appears in `REPORT.md`
and in the "Limitations" section of `latex/experiments_results.tex`.

1. **The uncertainty signal is inverted.** σ *grows* with history length
   (r = +0.110), mean σ is higher for correctly served users (0.140) than for
   incorrectly served ones (0.106), and Recall@10 rises from 0.0077 in the most
   confident quartile to 0.0451 in the most uncertain. Selective recommendation
   therefore *degrades* performance (0.0193 → 0.0153 serving the 90 % most
   confident), and the γ sweep is flat. Cause: the KL term is weighted at
   β = 1e-4, so the divergence reaches 60–150 nats and constrains almost
   nothing.

2. **An untrained baseline is competitive.** Averaging the frozen LLM
   embeddings of the history items and scoring by cosine similarity — **no
   training at all** — reaches Recall@10 = 0.0292 on Amazon Beauty, above the
   trained full model (0.0193).

3. **MovieLens-1M significance is partial.** LLM-SEMRec does not significantly
   beat BPR-MF (*p* = 0.30) or LLM-only (*p* = 0.29) in Recall@10.

---

## Configuration deviations

Results were produced on a **CPU-only** host (8 cores, no GPU), which forced a
configuration below the paper's default:

| Parameter | This work | Paper default |
|---|---|---|
| Shared dimension *d* | 128 | 256 |
| Latent dimension *d_z* | 64 | 64 |
| Max history length *L* | 20 | 50 |
| Text encoder | Qwen3-Embedding-0.6B | Qwen3-Embedding-4B |
| Vision encoder | SigLIP 2 base | SigLIP 2 |
| Seeds per model | **1** | 5 |

All values lie inside the paper's permitted hyperparameter grid. The single
seed means the significance tests capture inter-user variability only, not
run-to-run variability — close comparisons should be treated as provisional.

**Amazon Fashion was preprocessed but not trained:** encoding its 23 033 items
would take roughly three hours on this host.

---

## Repository layout

```
code/                 pipeline and analysis
  paths.py            all paths derived from the repository root
  prep.py             iterative 5-core filtering, chronological split, Table II
  download_images.py  image retrieval (legacy images.amazon.com fallback)
  encode_text.py      Qwen3-Embedding-0.6B -> cache/
  encode_image.py     SigLIP 2 base vision tower -> cache/
  model.py            the architecture (gated fusion, hybrid encoder, VAE, SSL)
  train_eval.py       leave-two-out training and evaluation
  baselines.py        Popularity, BPR-MF, GRU4Rec, Caser, SASRec, BERT4Rec,
                      CL4SRec, S3-Rec, MM-SASRec, MMSSL-lite, LLM-only
  run_main.py         main driver
  run_analysis.py     cold-start, sparsity, calibration, latency
  significance.py     paired t-tests
  make_figures.py     the 8 principal figures
  make_figures_supporting.py   the 9 supplementary figures
  make_latex.py       the LaTeX tables
  make_report.py      REPORT.md
  make_consolidated.py         single-file recap (PDF + HTML)
  make_headline.py    distill the small aggregate metrics
  validate.py         gradient and sanity checks
  validate_latex.py   structural LaTeX validation

figures/              21 figures, PNG 300 dpi and vector PDF
tables/               9 LaTeX tables, ready to \input
results/              headline_metrics.json + small aggregates
latex/                the Experiments and Results section (English)
data/  cache/  models/  logs/   gitignored, rebuilt by the scripts

REPORT.md             full write-up, 9 sections
SUPPORTING_FIGURES.md captions and measured numbers for the 9 extra figures
RECAP_results.pdf            30 pages, complete recap
RECAP_results_supporting.pdf 31 pages, results consistent with the paper only
LLM-SEMRec_figures.zip       16 vector figures + captions
```

---

## Reproducing

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install transformers numpy scipy scikit-learn pillow \
    matplotlib pymupdf

# 1. datasets (Amazon Beauty, Amazon Fashion, MovieLens-1M)
python code/prep.py beauty
python code/prep.py fashion
python code/prep.py movielens

# 2. frozen content embeddings
python code/download_images.py amazon_beauty
python code/encode_text.py amazon_beauty       # ~93 min on 8 CPU cores
python code/encode_image.py amazon_beauty      # ~35 min

# 3. sanity checks before trusting anything
python code/validate.py

# 4. train and evaluate
python code/run_main.py amazon_beauty llmsemrec --seeds 0 --epochs 20 --patience 4
python code/run_main.py amazon_beauty popularity,bprmf,gru4rec,caser,sasrec,bert4rec \
       --seeds 0 --epochs 15 --patience 4

# 5. analyses, significance, aggregation
python code/run_analysis.py amazon_beauty
python code/significance.py
python code/make_headline.py

# 6. artefacts
python code/make_figures.py
python code/make_figures_supporting.py
python code/make_latex.py
python code/make_report.py
python code/make_consolidated.py --supporting
```

Threads are capped and the process priority lowered throughout so the host
stays usable; the machine has no GPU and saturating it roughly doubles wall
time.

---

## Validity control

An untrained model returns Recall@10 = 0.00094 against a chance level of
0.000083 under full-catalogue ranking. This check is what exposed three
masking bugs during the reproduction:

1. **Missing causal mask** — the transformer was bidirectional, so position *k*
   could see the target at *k+1*. Symptom: validation NDCG@10 = 0.9546.
2. **NaN from fully masked attention rows** — with left padding plus a causal
   mask, pure-padding positions had every key masked, so softmax(−inf) produced
   NaN. The NaN propagated through the whole tensor and made every rank 0,
   which is why a random model appeared to score 0.95.
3. **Wrong read position in GRU4Rec.**

Any number produced before these fixes is unusable. The control above is the
evidence that the reported figures are not inflated by leakage.

---

## Environment

Windows 11, 8 cores, 34 GB RAM, **no GPU**. Python 3.11, PyTorch CPU.
LaTeX was rendered with Tectonic (MiKTeX under another Windows account was not
readable from this session).
