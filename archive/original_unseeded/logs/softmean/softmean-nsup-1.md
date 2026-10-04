# softmean — nsup-1  (halt > 0, n_sup = 1)

Q-loss: soft-mean of token correctness (pad-masked). Role: supervision sweep s=1 (no recursion).
Common settings: c=0.1, T=3, 100 epochs; metrics logged every 10 epochs (0–90).
Source: trm_qloss_softmean.ipynb, cell 26. Every value below is parsed verbatim from the run's printed log — nothing is recomputed.

## Per-epoch metrics

| epoch | train CE | Q-loss | avg steps (train, batch-level) | halt-logit mean | std | max | %logit>0 | %logit>1.5 | %step1 (train) | %never (train) | val acc | val steps mean | %step1 (val) | %never (val) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.6129 | 0.6244 | 1.0 | -0.724 | 0.383 | 0.196 | 0.3% | 0.0% | 100.0% | 100.0% | 0.10% | 1.00 | 100.0% | 100.0% |
| 10 | 0.6493 | 0.5857 | 1.0 | 0.991 | 0.257 | 1.977 | 100.0% | 2.5% | 100.0% | 100.0% | 7.92% | 1.00 | 100.0% | 100.0% |
| 20 | 0.5414 | 0.5338 | 1.0 | 1.237 | 0.193 | 1.955 | 100.0% | 9.0% | 100.0% | 100.0% | 8.60% | 1.00 | 100.0% | 100.0% |
| 30 | 0.0982 | 0.1373 | 1.0 | 3.598 | 0.631 | 6.535 | 100.0% | 99.9% | 100.0% | 100.0% | 86.60% | 1.00 | 100.0% | 100.0% |
| 40 | 0.0402 | 0.0688 | 1.0 | 4.677 | 0.954 | 8.324 | 100.0% | 100.0% | 100.0% | 100.0% | 94.26% | 1.00 | 100.0% | 100.0% |
| 50 | 0.0356 | 0.0545 | 1.0 | 5.104 | 1.093 | 9.834 | 100.0% | 99.9% | 100.0% | 100.0% | 94.06% | 1.00 | 100.0% | 100.0% |
| 60 | 0.0264 | 0.0441 | 1.0 | 5.289 | 0.995 | 9.408 | 100.0% | 100.0% | 100.0% | 100.0% | 97.56% | 1.00 | 100.0% | 100.0% |
| 70 | 0.0185 | 0.0351 | 1.0 | 5.561 | 1.004 | 9.983 | 100.0% | 100.0% | 100.0% | 100.0% | 98.74% | 1.00 | 100.0% | 100.0% |
| 80 | 0.0233 | 0.0315 | 1.0 | 6.096 | 1.253 | 10.487 | 100.0% | 99.8% | 100.0% | 100.0% | 76.86% | 1.00 | 100.0% | 100.0% |
| 90 | 0.0142 | 0.0261 | 1.0 | 6.226 | 1.311 | 11.740 | 100.0% | 100.0% | 100.0% | 100.0% | 98.90% | 1.00 | 100.0% | 100.0% |


## Training halt-step distribution (per-sample counts, whole train set)
Counts of samples by recorded halting step. Column `step 1` mixes two cases the log cannot separate: halted exactly at step 1, and never halted (both are recorded as 1). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | total | val acc (epoch) |
|---|---|---|---|
| 0 | 45000 | 45000 | 0.10% |
| 10 | 45000 | 45000 | 7.92% |
| 20 | 45000 | 45000 | 8.60% |
| 30 | 45000 | 45000 | 86.60% |
| 40 | 45000 | 45000 | 94.26% |
| 50 | 45000 | 45000 | 94.06% |
| 60 | 45000 | 45000 | 97.56% |
| 70 | 45000 | 45000 | 98.74% |
| 80 | 45000 | 45000 | 76.86% |
| 90 | 45000 | 45000 | 98.90% |

## Validation halt-step distribution (per-sample counts, whole validation set)
Counts of samples by recorded halting step. Column `step 1` mixes two cases the log cannot separate: halted exactly at step 1, and never halted (both are recorded as 1). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | total | val acc (epoch) |
|---|---|---|---|
| 0 | 5000 | 5000 | 0.10% |
| 10 | 5000 | 5000 | 7.92% |
| 20 | 5000 | 5000 | 8.60% |
| 30 | 5000 | 5000 | 86.60% |
| 40 | 5000 | 5000 | 94.26% |
| 50 | 5000 | 5000 | 94.06% |
| 60 | 5000 | 5000 | 97.56% |
| 70 | 5000 | 5000 | 98.74% |
| 80 | 5000 | 5000 | 76.86% |
| 90 | 5000 | 5000 | 98.90% |

## Validation halt-logit mean by supervision step
(first validation batch only; blank = validation halted every sample before reaching that step, so it was not executed; val acc is the epoch-level validation accuracy)

| epoch | s1 | val acc (epoch) |
|---|---|---|
| 0 | -0.624 | 0.10% |
| 10 | 1.130 | 7.92% |
| 20 | 1.279 | 8.60% |
| 30 | 3.527 | 86.60% |
| 40 | 4.839 | 94.26% |
| 50 | 4.553 | 94.06% |
| 60 | 5.535 | 97.56% |
| 70 | 6.275 | 98.74% |
| 80 | 2.709 | 76.86% |
| 90 | 6.524 | 98.90% |
