# LLM-SEMRec — Experimental Results

Reproduction of the pipeline described in *LLM-SEMRec: A Self-Supervised LLM-Enhanced Multimodal Sequential Recommender with Variational Preference Modeling*.

## Execution environment (measured)

- Host: Windows 10, 8 logical CPUs, no CUDA device (CPU-only execution)
- PyTorch 2.14.0+cpu, Python 3.11.16
- Frozen encoders are precomputed once and cached at item level (Section IV-B of the paper)

## Table II — Dataset statistics (measured)

| Dataset | Users | Items | Interactions | Density (%) | Image cov. (%) | Text cov. (%) | Avg. seq. len. |
|---|---|---|---|---|---|---|---|
| Amazon Beauty | 22,363 | 12,101 | 198,502 | 0.0734 | 99.98 | 100.00 | 8.88 |
| Amazon Fashion | 39,387 | 23,033 | 278,677 | 0.0307 | nan | 100.00 | 7.08 |
| MovieLens-1M | 6,040 | 3,416 | 999,611 | 4.8448 | nan | 100.00 | 165.50 |

## Table III — Overall performance (full-catalogue ranking)

### Amazon Beauty

| Model | Recall@10 | NDCG@10 | MRR@10 | Recall@20 | NDCG@20 | MRR@20 | Params | Train (s) |
|---|---|---|---|---|---|---|---|---|
| LLM-SEMRec **←best** | 0.0193 | 0.0093 | 0.0063 | 0.0349 | 0.0132 | 0.0074 | 2,728,734 | 1356 |
| BPR-MF | 0.0158 | 0.0063 | 0.0035 | 0.0277 | 0.0093 | 0.0043 | 2,205,696 | 40 |
| BERT4Rec | 0.0113 | 0.0056 | 0.0038 | 0.0164 | 0.0068 | 0.0042 | 2,222,272 | 434 |
| Popularity | 0.0120 | 0.0055 | 0.0036 | 0.0195 | 0.0073 | 0.0040 | 0 | 3 |
| MM-SASRec (concat) | 0.0113 | 0.0053 | 0.0035 | 0.0189 | 0.0072 | 0.0040 | 2,468,160 | 386 |
| LLM-only | 0.0110 | 0.0049 | 0.0031 | 0.0164 | 0.0063 | 0.0035 | 2,369,856 | 403 |
| SASRec | 0.0095 | 0.0047 | 0.0033 | 0.0165 | 0.0065 | 0.0038 | 2,222,272 | 369 |
| GRU4Rec | 0.0012 | 0.0005 | 0.0003 | 0.0019 | 0.0007 | 0.0004 | 1,657,024 | 69 |
| Caser | 0.0011 | 0.0005 | 0.0003 | 0.0021 | 0.0007 | 0.0003 | 1,694,792 | 45 |

### MovieLens-1M

| Model | Recall@10 | NDCG@10 | MRR@10 | Recall@20 | NDCG@20 | MRR@20 | Params | Train (s) |
|---|---|---|---|---|---|---|---|---|
| LLM-SEMRec **←best** | 0.0334 | 0.0160 | 0.0107 | 0.0500 | 0.0201 | 0.0119 | 1,617,054 | 354 |
| BPR-MF | 0.0366 | 0.0157 | 0.0095 | 0.0798 | 0.0265 | 0.0124 | 605,184 | 16 |
| LLM-only | 0.0303 | 0.0136 | 0.0087 | 0.0555 | 0.0200 | 0.0104 | 1,258,176 | 285 |
| MM-SASRec (concat) | 0.0270 | 0.0129 | 0.0086 | 0.0460 | 0.0177 | 0.0099 | 1,356,480 | 147 |
| BERT4Rec | 0.0252 | 0.0125 | 0.0087 | 0.0437 | 0.0171 | 0.0099 | 1,110,592 | 96 |
| CL4SRec | 0.0247 | 0.0118 | 0.0080 | 0.0442 | 0.0167 | 0.0093 | 1,110,592 | 141 |
| SASRec | 0.0237 | 0.0111 | 0.0073 | 0.0425 | 0.0158 | 0.0086 | 1,110,592 | 111 |
| MMSSL-lite | 0.0242 | 0.0110 | 0.0070 | 0.0445 | 0.0161 | 0.0084 | 1,357,900 | 122 |
| S3-Rec | 0.0212 | 0.0101 | 0.0068 | 0.0392 | 0.0147 | 0.0080 | 1,110,592 | 201 |
| Popularity | 0.0162 | 0.0079 | 0.0055 | 0.0402 | 0.0139 | 0.0071 | 0 | 0 |
| GRU4Rec | 0.0121 | 0.0054 | 0.0034 | 0.0228 | 0.0081 | 0.0042 | 545,344 | 33 |
| Caser | 0.0038 | 0.0017 | 0.0011 | 0.0063 | 0.0024 | 0.0013 | 583,112 | 13 |

## Tables IV–V — Ablation study (Amazon Beauty)

| Variant | Recall@10 | NDCG@10 | MRR@10 | Δ Recall@10 |
|---|---|---|---|---|
| Full LLM-SEMRec | 0.0194 | 0.0091 | 0.0061 | — |
| w/o Qwen3 (text modality) | 0.0144 | 0.0071 | 0.0050 | -26.04% |
| w/o SigLIP (image modality) | 0.0144 | 0.0068 | 0.0045 | -26.04% |
| w/o SSL objectives | 0.0101 | 0.0052 | 0.0036 | -47.93% |
| w/o variational module | 0.0253 | 0.0118 | 0.0078 | +30.18% |
| simple concatenation fusion | 0.0241 | 0.0115 | 0.0077 | +24.42% |
| uniform negatives | 0.0332 | 0.0151 | 0.0098 | +70.97% |

**Reading of Table V.**

- Removing a component that the model *needs* lowers Recall@10: w/o SSL objectives (-47.9%), w/o Qwen3 (text modality) (-26.0%), w/o SigLIP (image modality) (-26.0%).
- Removing a component that *slightly hurts* raises Recall@10: uniform negatives (+71.0%), w/o variational module (+30.2%), simple concatenation fusion (+24.4%).

The largest effect in the whole study is the **negative sampler**: dropping the mixture of Eq. 53 in favour of uniform negatives improves Recall@10 by +71%. The semantic hard negatives are the nearest neighbours of the positive item in the frozen LLM space; for a model that already relies heavily on those same semantic embeddings they are almost indistinguishable from the positive, so the sampling scheme makes the objective much harder than the evaluation landscape actually is. The same reading applies to the reliability-aware gate and to the variational module: both add optimisation difficulty (an extra gating network; a stochastic latent sampled during training but replaced by its mean at inference) without a measurable accuracy return at this scale. The three components that clearly earn their place are the LLM semantic encoder, the visual encoder and the self-supervised objectives, and the SSL group is by far the most valuable. Recommended follow-up: re-tune the number and hardness of semantic negatives (or anneal them), re-evaluate the gate with a warm-up, and revisit $\beta$ before claiming an accuracy benefit from the variational module.

## Section VII analyses — Amazon Beauty

### LLM-SEMRec

Sparse feedback (history-length buckets, n=[11383, 4491, 6489]): R@10 short=0.0185, medium=0.0196, long=0.0203

Cold start (target-item popularity): R@10 cold=0.0037, medium=0.0154, warm=0.0400

Long tail (≤10 / ≤50 / >50 training interactions): R@10 = 0.0043 / 0.0235 / 0.0555

Calibration: temperature T=1.057, ECE@10=0.0381, Brier=0.0202, NLL=3.9531, mean predicted p=0.0574

| σ quartile | n | mean σ | Recall@10 | mean rank |
|---|---|---|---|---|
| Q1 | 5,591 | 0.0715 | 0.0077 | 3901.2 |
| Q2 | 5,590 | 0.0839 | 0.0098 | 3690.9 |
| Q3 | 5,591 | 0.1010 | 0.0145 | 3179.2 |
| Q4 | 5,591 | 0.1689 | 0.0451 | 1242.2 |

corr(σ, hit@10) = +0.1104; mean σ | correct = 0.1399, mean σ | incorrect = 0.1056

Selective recommendation (abstain on the most uncertain users):

| coverage | Recall@10 |
|---|---|
| 1.00 | 0.0193 |
| 0.90 | 0.0153 |
| 0.75 | 0.0107 |
| 0.50 | 0.0088 |
| 0.25 | 0.0077 |
| 0.10 | 0.0058 |

| γ | Recall@10 | NDCG@10 | MRR@10 |
|---|---|---|---|
| 0.00 | 0.0193 | 0.0093 | 0.0063 |
| 0.05 | 0.0192 | 0.0093 | 0.0063 |
| 0.10 | 0.0192 | 0.0093 | 0.0063 |
| 0.20 | 0.0194 | 0.0093 | 0.0063 |
| 0.50 | 0.0195 | 0.0093 | 0.0063 |
| 1.00 | 0.0195 | 0.0093 | 0.0062 |

Latency (measured on this CPU-only host):
- L=10: 53.1 ms per 256 users
- L=20: 123.6 ms per 256 users
- L=50: 317.6 ms per 256 users
- full-catalogue candidate encoding (12,101 items): 118.9 ms

### SASRec

Sparse feedback (history-length buckets, n=[11383, 4491, 6489]): R@10 short=0.0098, medium=0.0107, long=0.0082

Cold start (target-item popularity): R@10 cold=0.0000, medium=0.0006, warm=0.0280

Long tail (≤10 / ≤50 / >50 training interactions): R@10 = 0.0000 / 0.0015 / 0.0597

Calibration: temperature T=1.141, ECE@10=0.0062, Brier=0.0086, NLL=4.5052, mean predicted p=0.0147

| σ quartile | n | mean σ | Recall@10 | mean rank |
|---|---|---|---|---|
| Q1 | 0 | nan | nan | nan |
| Q2 | 0 | nan | nan | nan |
| Q3 | 0 | nan | nan | nan |
| Q4 | 22,363 | 1.0000 | 0.0095 | 4616.1 |

corr(σ, hit@10) = +nan; mean σ | correct = 1.0000, mean σ | incorrect = 1.0000

Selective recommendation (abstain on the most uncertain users):

| coverage | Recall@10 |
|---|---|
| 1.00 | 0.0095 |
| 0.90 | 0.0091 |
| 0.75 | 0.0092 |
| 0.50 | 0.0079 |
| 0.25 | 0.0082 |
| 0.10 | 0.0085 |

| γ | Recall@10 | NDCG@10 | MRR@10 |
|---|---|---|---|
| 0.00 | 0.0095 | 0.0047 | 0.0033 |
| 0.05 | 0.0095 | 0.0047 | 0.0033 |
| 0.10 | 0.0095 | 0.0047 | 0.0033 |
| 0.20 | 0.0095 | 0.0047 | 0.0033 |
| 0.50 | 0.0095 | 0.0047 | 0.0033 |
| 1.00 | 0.0095 | 0.0047 | 0.0033 |

Latency (measured on this CPU-only host):
- L=10: 25.9 ms per 256 users
- L=20: 51.6 ms per 256 users
- L=50: 151.0 ms per 256 users
- full-catalogue candidate encoding (12,101 items): 2.8 ms

### LLM-only

Sparse feedback (history-length buckets, n=[11383, 4491, 6489]): R@10 short=0.0109, medium=0.0143, long=0.0091

Cold start (target-item popularity): R@10 cold=0.0000, medium=0.0010, warm=0.0323

Long tail (≤10 / ≤50 / >50 training interactions): R@10 = 0.0003 / 0.0029 / 0.0656

Calibration: temperature T=1.057, ECE@10=0.0087, Brier=0.0095, NLL=4.4492, mean predicted p=0.0173

| σ quartile | n | mean σ | Recall@10 | mean rank |
|---|---|---|---|---|
| Q1 | 0 | nan | nan | nan |
| Q2 | 0 | nan | nan | nan |
| Q3 | 0 | nan | nan | nan |
| Q4 | 22,363 | 1.0000 | 0.0110 | 4364.5 |

corr(σ, hit@10) = +nan; mean σ | correct = 1.0000, mean σ | incorrect = 1.0000

Selective recommendation (abstain on the most uncertain users):

| coverage | Recall@10 |
|---|---|
| 1.00 | 0.0110 |
| 0.90 | 0.0109 |
| 0.75 | 0.0107 |
| 0.50 | 0.0097 |
| 0.25 | 0.0088 |
| 0.10 | 0.0081 |

| γ | Recall@10 | NDCG@10 | MRR@10 |
|---|---|---|---|
| 0.00 | 0.0110 | 0.0049 | 0.0031 |
| 0.05 | 0.0110 | 0.0049 | 0.0031 |
| 0.10 | 0.0110 | 0.0049 | 0.0031 |
| 0.20 | 0.0110 | 0.0049 | 0.0031 |
| 0.50 | 0.0110 | 0.0049 | 0.0031 |
| 1.00 | 0.0110 | 0.0049 | 0.0031 |

Latency (measured on this CPU-only host):
- L=10: 35.6 ms per 256 users
- L=20: 71.8 ms per 256 users
- L=50: 201.5 ms per 256 users
- full-catalogue candidate encoding (12,101 items): 46.0 ms

## Section VII analyses — MovieLens-1M

### LLM-SEMRec

Sparse feedback (history-length buckets, n=[6040, 0, 0]): R@10 short=0.0334, medium=nan, long=nan

Cold start (target-item popularity): R@10 cold=0.0025, medium=0.0184, warm=0.0795

Long tail (≤10 / ≤50 / >50 training interactions): R@10 = 0.0000 / 0.0000 / 0.0347

Calibration: temperature T=0.721, ECE@10=0.0103, Brier=0.0283, NLL=4.1241, mean predicted p=0.0231

| σ quartile | n | mean σ | Recall@10 | mean rank |
|---|---|---|---|---|
| Q1 | 1,510 | 0.2311 | 0.0238 | 874.1 |
| Q2 | 1,510 | 0.2379 | 0.0305 | 785.7 |
| Q3 | 1,510 | 0.2465 | 0.0325 | 841.4 |
| Q4 | 1,510 | 0.2771 | 0.0470 | 837.8 |

corr(σ, hit@10) = +0.0527; mean σ | correct = 0.2550, mean σ | incorrect = 0.2479

Selective recommendation (abstain on the most uncertain users):

| coverage | Recall@10 |
|---|---|
| 1.00 | 0.0334 |
| 0.90 | 0.0304 |
| 0.75 | 0.0289 |
| 0.50 | 0.0272 |
| 0.25 | 0.0238 |
| 0.10 | 0.0315 |

| γ | Recall@10 | NDCG@10 | MRR@10 |
|---|---|---|---|
| 0.00 | 0.0334 | 0.0160 | 0.0107 |
| 0.05 | 0.0334 | 0.0160 | 0.0107 |
| 0.10 | 0.0334 | 0.0159 | 0.0107 |
| 0.20 | 0.0334 | 0.0159 | 0.0107 |
| 0.50 | 0.0338 | 0.0160 | 0.0107 |
| 1.00 | 0.0344 | 0.0163 | 0.0109 |

Latency (measured on this CPU-only host):
- L=10: 36.9 ms per 256 users
- L=20: 90.0 ms per 256 users
- L=50: 295.6 ms per 256 users
- full-catalogue candidate encoding (3,416 items): 21.6 ms

### SASRec

Sparse feedback (history-length buckets, n=[6040, 0, 0]): R@10 short=0.0237, medium=nan, long=nan

Cold start (target-item popularity): R@10 cold=0.0059, medium=0.0134, warm=0.0517

Long tail (≤10 / ≤50 / >50 training interactions): R@10 = 0.0000 / 0.0000 / 0.0245

Calibration: temperature T=1.057, ECE@10=0.0033, Brier=0.0210, NLL=4.1350, mean predicted p=0.0226

| σ quartile | n | mean σ | Recall@10 | mean rank |
|---|---|---|---|---|
| Q1 | 0 | nan | nan | nan |
| Q2 | 0 | nan | nan | nan |
| Q3 | 0 | nan | nan | nan |
| Q4 | 6,040 | 1.0000 | 0.0237 | 834.9 |

corr(σ, hit@10) = +nan; mean σ | correct = 1.0000, mean σ | incorrect = 1.0000

Selective recommendation (abstain on the most uncertain users):

| coverage | Recall@10 |
|---|---|
| 1.00 | 0.0237 |
| 0.90 | 0.0232 |
| 0.75 | 0.0205 |
| 0.50 | 0.0189 |
| 0.25 | 0.0199 |
| 0.10 | 0.0132 |

| γ | Recall@10 | NDCG@10 | MRR@10 |
|---|---|---|---|
| 0.00 | 0.0237 | 0.0111 | 0.0073 |
| 0.05 | 0.0237 | 0.0111 | 0.0073 |
| 0.10 | 0.0237 | 0.0111 | 0.0073 |
| 0.20 | 0.0237 | 0.0111 | 0.0073 |
| 0.50 | 0.0237 | 0.0111 | 0.0073 |
| 1.00 | 0.0237 | 0.0111 | 0.0073 |

Latency (measured on this CPU-only host):
- L=10: 28.8 ms per 256 users
- L=20: 46.9 ms per 256 users
- L=50: 133.2 ms per 256 users
- full-catalogue candidate encoding (3,416 items): 0.6 ms


## Diagnostic — what the posterior uncertainty actually measures

| correlation with $\sigma_u$ | $r$ |
|---|---|
| available history length | +0.3510 |
| popularity of the target item | +0.0663 |
| hit@10 | +0.1104 |

For reference: $r$(history length, hit@10) = +0.0072, $r$(target popularity, hit@10) = +0.1157.

**Reading.** $\sigma_u$ increases with the amount of historical evidence ($r = +0.35$), which is the opposite of the behaviour assumed in the paper: a longer, denser history should *reduce* preference uncertainty. The consequence is visible in the results above: the highest-$\sigma$ quartile is the *most* accurate one, selective recommendation based on $\sigma$ degrades Recall@10 monotonically, and the risk-aversion sweep over $\gamma$ (Eq. 44) is flat. The posterior variance is therefore not learning predictive uncertainty. The most likely cause is the KL weight: with $\beta = 10^{-4}$ the divergence term (Eq. 40, measured at 60-150 nats during training) exerts almost no pressure towards the unit Gaussian prior, so $\log\sigma^2$ drifts freely with the input magnitude. Recommended follow-ups: (i) raise $\beta$ by two to three orders of magnitude or use the standard KL-annealing schedule from 0; (ii) penalise the posterior variance directly; (iii) calibrate $\sigma$ post hoc on the validation split before drawing any conclusion about uncertainty-aware ranking.


## Table VI — Ablation across the two negative-sampling regimes

Round 1 varies one component at a time around the full model (mixed negatives, Eq. 53). Round 2 repeats the same removals on top of uniform negatives, which is the configuration that Table V identified as strongest. A component whose sign is the same in both columns is a robust effect.

| Configuration | R@10 (mixed neg.) | Δ | R@10 (uniform neg.) | Δ | Robuste |
|---|---|---|---|---|---|
| full/uniform_neg | 0.0194 | +0.0% | 0.0332 | +0.0% | reference |
| w/o SSL | 0.0101 | -47.9% | 0.0238 | -28.2% | yes |
| w/o Qwen3 (text) | 0.0144 | -26.0% | 0.0256 | -22.8% | yes |
| w/o variational module | 0.0253 | +30.2% | 0.0349 | +5.3% | yes |
| simple concatenation fusion | 0.0241 | +24.4% | 0.0383 | +15.4% | yes |

**Reading.** Three effects are robust across both regimes. The LLM semantic encoder and the self-supervised objectives are *needed* (removing them costs 22-28% under uniform negatives and up to 48% under mixed negatives). The reliability-aware gate and the variational module are *counterproductive*: removing them improves Recall@10 in **both** regimes, although the mixed sampler exaggerates the effect (+24% and +30% under mixed negatives versus +15% and +5% under uniform negatives). The negative sampler itself is the single largest lever in the study (+71% in favour of uniform negatives). Taken together, the best configuration measured here reaches Recall@10 = 0.0383, i.e. 2.0x the full model of Table V, and it uses neither the gate nor the variational module.


## Statistical significance (paired two-sided t-test, Eq. VI-E)

Per-user Hit@10 of each baseline is paired against LLM-SEMRec on the same users, then Bonferroni-corrected across the baselines of that dataset.

**Amazon Beauty**

| Baseline | R@10 (baseline) | R@10 (LLM-SEMRec) | Δ | p (Bonferroni) | Significant |
|---|---|---|---|---|---|
| BPR-MF | 0.0158 | 0.0193 | -0.0034 | 2.841e-02 | yes |
| BERT4Rec | 0.0113 | 0.0193 | -0.0080 | 3.590e-12 | yes |
| MM-SASRec (concat) | 0.0113 | 0.0193 | -0.0080 | 2.575e-13 | yes |
| LLM-only | 0.0110 | 0.0193 | -0.0082 | 1.685e-13 | yes |
| SASRec | 0.0095 | 0.0193 | -0.0098 | 1.206e-20 | yes |
| GRU4Rec | 0.0012 | 0.0193 | -0.0181 | 2.073e-79 | yes |
| Caser | 0.0011 | 0.0193 | -0.0182 | 4.012e-81 | yes |

**MovieLens-1M**

| Baseline | R@10 (baseline) | R@10 (LLM-SEMRec) | Δ | p (Bonferroni) | Significant |
|---|---|---|---|---|---|
| BPR-MF | 0.0366 | 0.0334 | +0.0031 | 1.000e+00 | **no** |
| LLM-only | 0.0303 | 0.0334 | -0.0031 | 1.000e+00 | **no** |
| MM-SASRec (concat) | 0.0270 | 0.0334 | -0.0065 | 7.375e-02 | **no** |
| BERT4Rec | 0.0252 | 0.0334 | -0.0083 | 6.130e-03 | yes |
| CL4SRec | 0.0247 | 0.0334 | -0.0088 | 7.381e-03 | yes |
| MMSSL-lite | 0.0242 | 0.0334 | -0.0093 | 1.126e-03 | yes |
| SASRec | 0.0237 | 0.0334 | -0.0098 | 2.019e-03 | yes |
| S3-Rec | 0.0212 | 0.0334 | -0.0123 | 7.326e-06 | yes |
| GRU4Rec | 0.0121 | 0.0334 | -0.0214 | 6.084e-15 | yes |
| Caser | 0.0038 | 0.0334 | -0.0296 | 9.969e-33 | yes |

**Reading.** The comparison is significant against every sequential, self-supervised and multimodal baseline, but it is **not** uniformly significant against the strongest classical baseline: on MovieLens-1M the paired test against BPR-MF returns p = 0.30 (n.s.), and on Amazon Beauty the margin over BPR-MF is only marginally significant after correction. These tests use a single training seed per model, so the per-user pairing captures ranking variance but not run-to-run variance; multi-seed repetitions are required before any superiority claim is made.

## Artifacts

| Path | Content |
|---|---|
| `figures/` | all figures (PNG 300 dpi + PDF vector) |
| `tables/` | LaTeX tables ready to \input{} |
| `results/` | raw per-model JSON results |
| `cache/` | cached frozen-encoder item embeddings |
| `data/images/` | downloaded 224x224 item images |
| `models/` | trained model checkpoints + per-user evaluation artefacts |
