# geomean — halt-1.5  (halt > 1.5, n_sup = 5)

Q-loss: geometric mean of correct-token probabilities = exp(-mean CE) (pad-masked). Role: threshold sweep t=1.5.
Common settings: c=0.1, T=3, 100 epochs; metrics logged every 10 epochs (0–90).
Source: trm_qloss_geomean.ipynb, cell 25. Every value below is parsed verbatim from the run's printed log — nothing is recomputed.

## Per-epoch metrics

| epoch | train CE | Q-loss | avg steps (train, batch-level) | halt-logit mean | std | max | %logit>0 | %logit>1.5 | %step1 (train) | %never (train) | val acc | val steps mean | %step1 (val) | %never (val) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.6364 | 0.4645 | 5.0 | -1.550 | 0.376 | -0.024 | 0.0% | 0.0% | 0.0% | 100.0% | 0.10% | 5.00 | 0.0% | 100.0% |
| 10 | 0.6615 | 0.6914 | 5.0 | -0.016 | 0.209 | 0.756 | 48.8% | 0.0% | 0.0% | 100.0% | 7.44% | 5.00 | 0.0% | 100.0% |
| 20 | 0.5535 | 0.6858 | 5.0 | 0.218 | 0.166 | 0.826 | 89.0% | 0.0% | 0.0% | 100.0% | 7.66% | 5.00 | 0.0% | 100.0% |
| 30 | 0.5117 | 0.6788 | 5.0 | 0.325 | 0.156 | 0.912 | 98.7% | 0.0% | 0.0% | 100.0% | 13.24% | 5.00 | 0.0% | 100.0% |
| 40 | 0.2383 | 0.4115 | 4.8450604122245915 | 1.813 | 0.410 | 3.523 | 100.0% | 79.0% | 74.8% | 17.4% | 78.28% | 1.57 | 81.4% | 12.7% |
| 50 | 0.2617 | 0.3831 | 4.505330490405117 | 1.937 | 0.499 | 3.843 | 99.9% | 82.0% | 82.5% | 14.5% | 78.04% | 1.35 | 89.9% | 8.2% |
| 60 | 0.0309 | 0.1329 | 1.0014214641080312 | 3.582 | 0.560 | 5.793 | 100.0% | 100.0% | 100.0% | 0.0% | 96.24% | 1.00 | 100.0% | 0.0% |
| 70 | 0.2135 | 0.3310 | 4.23454157782516 | 2.208 | 0.599 | 4.870 | 100.0% | 88.7% | 89.2% | 8.3% | 86.12% | 1.20 | 93.7% | 4.6% |
| 80 | 0.2547 | 0.3379 | 4.16133617626155 | 2.148 | 0.604 | 4.775 | 99.9% | 85.6% | 87.6% | 10.9% | 89.16% | 1.03 | 98.8% | 0.7% |
| 90 | 0.0086 | 0.0493 | 1.003553660270078 | 4.981 | 0.873 | 7.998 | 100.0% | 100.0% | 100.0% | 0.0% | 98.64% | 1.00 | 100.0% | 0.0% |


## Training halt-step distribution (per-sample counts, whole train set)
Counts of samples by recorded halting step. Column `step 5` mixes two cases the log cannot separate: halted exactly at step 5, and never halted (both are recorded as 5). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 45000 | 45000 | 0.10% |
| 10 | 0 | 0 | 0 | 0 | 45000 | 45000 | 7.44% |
| 20 | 0 | 0 | 0 | 0 | 45000 | 45000 | 7.66% |
| 30 | 0 | 0 | 0 | 0 | 45000 | 45000 | 13.24% |
| 40 | 33650 | 3240 | 243 | 41 | 7826 | 45000 | 78.28% |
| 50 | 37104 | 1233 | 126 | 27 | 6510 | 45000 | 78.04% |
| 60 | 44998 | 2 | 0 | 0 | 0 | 45000 | 96.24% |
| 70 | 40130 | 988 | 136 | 15 | 3731 | 45000 | 86.12% |
| 80 | 39405 | 620 | 67 | 10 | 4898 | 45000 | 89.16% |
| 90 | 44998 | 1 | 0 | 0 | 1 | 45000 | 98.64% |

## Validation halt-step distribution (per-sample counts, whole validation set)
Counts of samples by recorded halting step. Column `step 5` mixes two cases the log cannot separate: halted exactly at step 5, and never halted (both are recorded as 5). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 5000 | 5000 | 0.10% |
| 10 | 0 | 0 | 0 | 0 | 5000 | 5000 | 7.44% |
| 20 | 0 | 0 | 0 | 0 | 5000 | 5000 | 7.66% |
| 30 | 0 | 0 | 0 | 0 | 5000 | 5000 | 13.24% |
| 40 | 4070 | 274 | 22 | 1 | 633 | 5000 | 78.28% |
| 50 | 4497 | 89 | 3 | 1 | 410 | 5000 | 78.04% |
| 60 | 5000 | 0 | 0 | 0 | 0 | 5000 | 96.24% |
| 70 | 4685 | 73 | 10 | 2 | 230 | 5000 | 86.12% |
| 80 | 4939 | 25 | 0 | 2 | 34 | 5000 | 89.16% |
| 90 | 5000 | 0 | 0 | 0 | 0 | 5000 | 98.64% |

## Validation halt-logit mean by supervision step
(first validation batch only; blank = validation halted every sample before reaching that step, so it was not executed; val acc is the epoch-level validation accuracy)

| epoch | s1 | s2 | s3 | s4 | s5 | val acc (epoch) |
|---|---|---|---|---|---|---|
| 0 | -1.372 | -1.395 | -1.412 | -1.409 | -1.410 | 0.10% |
| 10 | 0.048 | 0.040 | 0.040 | 0.039 | 0.037 | 7.44% |
| 20 | 0.310 | 0.311 | 0.313 | 0.311 | 0.313 | 7.66% |
| 30 | 0.413 | 0.406 | 0.404 | 0.404 | 0.403 | 13.24% |
| 40 | 1.973 | 2.072 |  |  |  | 78.28% |
| 50 | 2.103 | 2.205 |  |  |  | 78.04% |
| 60 | 3.945 |  |  |  |  | 96.24% |
| 70 | 2.303 |  |  |  |  | 86.12% |
| 80 | 2.619 |  |  |  |  | 89.16% |
| 90 | 5.341 |  |  |  |  | 98.64% |
