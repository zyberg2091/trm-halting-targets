# geomean — nsup-1  (halt > 0, n_sup = 1)

Q-loss: geometric mean of correct-token probabilities = exp(-mean CE) (pad-masked). Role: supervision sweep s=1 (no recursion).
Common settings: c=0.1, T=3, 100 epochs; metrics logged every 10 epochs (0–90).
Source: trm_qloss_geomean.ipynb, cell 27. Every value below is parsed verbatim from the run's printed log — nothing is recomputed.

## Per-epoch metrics

| epoch | train CE | Q-loss | avg steps (train, batch-level) | halt-logit mean | std | max | %logit>0 | %logit>1.5 | %step1 (train) | %never (train) | val acc | val steps mean | %step1 (val) | %never (val) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.6230 | 0.4684 | 1.0 | -1.540 | 0.371 | -0.016 | 0.0% | 0.0% | 100.0% | 100.0% | 0.04% | 1.00 | 100.0% | 100.0% |
| 10 | 0.6366 | 0.6916 | 1.0 | 0.034 | 0.208 | 0.874 | 56.2% | 0.0% | 100.0% | 100.0% | 6.86% | 1.00 | 100.0% | 100.0% |
| 20 | 0.5398 | 0.6837 | 1.0 | 0.248 | 0.166 | 0.776 | 93.5% | 0.0% | 100.0% | 100.0% | 8.18% | 1.00 | 100.0% | 100.0% |
| 30 | 0.2700 | 0.5385 | 1.0 | 1.208 | 0.345 | 2.853 | 100.0% | 20.3% | 100.0% | 100.0% | 68.76% | 1.00 | 100.0% | 100.0% |
| 40 | 0.0365 | 0.1197 | 1.0 | 3.917 | 0.909 | 7.198 | 100.0% | 99.2% | 100.0% | 100.0% | 97.74% | 1.00 | 100.0% | 100.0% |
| 50 | 0.0283 | 0.0974 | 1.0 | 4.293 | 1.035 | 8.048 | 100.0% | 99.6% | 100.0% | 100.0% | 96.70% | 1.00 | 100.0% | 100.0% |
| 60 | 0.0167 | 0.0572 | 1.0 | 5.236 | 1.339 | 9.078 | 100.0% | 99.1% | 100.0% | 100.0% | 95.82% | 1.00 | 100.0% | 100.0% |
| 70 | 0.0200 | 0.0794 | 1.0 | 4.534 | 0.942 | 7.898 | 100.0% | 99.8% | 100.0% | 100.0% | 98.76% | 1.00 | 100.0% | 100.0% |
| 80 | 0.0251 | 0.0710 | 1.0 | 4.815 | 1.169 | 8.849 | 100.0% | 99.2% | 100.0% | 100.0% | 98.84% | 1.00 | 100.0% | 100.0% |
| 90 | 0.0165 | 0.0594 | 1.0 | 4.992 | 1.122 | 9.809 | 100.0% | 99.8% | 100.0% | 100.0% | 99.10% | 1.00 | 100.0% | 100.0% |


## Training halt-step distribution (per-sample counts, whole train set)
Counts of samples by recorded halting step. Column `step 1` mixes two cases the log cannot separate: halted exactly at step 1, and never halted (both are recorded as 1). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | total | val acc (epoch) |
|---|---|---|---|
| 0 | 45000 | 45000 | 0.04% |
| 10 | 45000 | 45000 | 6.86% |
| 20 | 45000 | 45000 | 8.18% |
| 30 | 45000 | 45000 | 68.76% |
| 40 | 45000 | 45000 | 97.74% |
| 50 | 45000 | 45000 | 96.70% |
| 60 | 45000 | 45000 | 95.82% |
| 70 | 45000 | 45000 | 98.76% |
| 80 | 45000 | 45000 | 98.84% |
| 90 | 45000 | 45000 | 99.10% |

## Validation halt-step distribution (per-sample counts, whole validation set)
Counts of samples by recorded halting step. Column `step 1` mixes two cases the log cannot separate: halted exactly at step 1, and never halted (both are recorded as 1). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | total | val acc (epoch) |
|---|---|---|---|
| 0 | 5000 | 5000 | 0.04% |
| 10 | 5000 | 5000 | 6.86% |
| 20 | 5000 | 5000 | 8.18% |
| 30 | 5000 | 5000 | 68.76% |
| 40 | 5000 | 5000 | 97.74% |
| 50 | 5000 | 5000 | 96.70% |
| 60 | 5000 | 5000 | 95.82% |
| 70 | 5000 | 5000 | 98.76% |
| 80 | 5000 | 5000 | 98.84% |
| 90 | 5000 | 5000 | 99.10% |

## Validation halt-logit mean by supervision step
(first validation batch only; blank = validation halted every sample before reaching that step, so it was not executed; val acc is the epoch-level validation accuracy)

| epoch | s1 | val acc (epoch) |
|---|---|---|
| 0 | -1.386 | 0.04% |
| 10 | 0.101 | 6.86% |
| 20 | 0.260 | 8.18% |
| 30 | 1.700 | 68.76% |
| 40 | 4.363 | 97.74% |
| 50 | 5.424 | 96.70% |
| 60 | 4.412 | 95.82% |
| 70 | 4.655 | 98.76% |
| 80 | 6.055 | 98.84% |
| 90 | 5.587 | 99.10% |
