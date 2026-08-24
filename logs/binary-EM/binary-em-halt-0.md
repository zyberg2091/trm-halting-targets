# binary-em — halt-0  (halt > 0, n_sup = 5)

Q-loss: binary exact-match: target 1 iff every real token correct (pad-masked). Role: threshold sweep t=0 (center cell).
Common settings: c=0.1, T=3, 100 epochs; metrics logged every 10 epochs (0–90).
Source: trm_qloss_binary_em.ipynb, cell 23. Every value below is parsed verbatim from the run's printed log — nothing is recomputed.

## Per-epoch metrics

| epoch | train CE | Q-loss | avg steps (train, batch-level) | halt-logit mean | std | max | %logit>0 | %logit>1.5 | %step1 (train) | %never (train) | val acc | val steps mean | %step1 (val) | %never (val) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.6411 | 0.0155 | 5.0 | -8.634 | 3.023 | -0.030 | 0.0% | 0.0% | 0.0% | 100.0% | 0.08% | 5.00 | 0.0% | 100.0% |
| 10 | 0.6647 | 0.2695 | 5.0 | -2.642 | 0.521 | -0.657 | 0.0% | 0.0% | 0.0% | 100.0% | 6.82% | 5.00 | 0.0% | 100.0% |
| 20 | 0.5662 | 0.3253 | 5.0 | -2.263 | 0.365 | -0.502 | 0.0% | 0.0% | 0.0% | 100.0% | 8.22% | 5.00 | 0.0% | 100.0% |
| 30 | 0.4710 | 0.5020 | 5.0 | -1.425 | 0.410 | 0.333 | 0.1% | 0.0% | 0.1% | 99.8% | 23.06% | 5.00 | 0.1% | 99.9% |
| 40 | 0.1232 | 0.3643 | 1.6226012793176972 | 1.380 | 1.378 | 5.234 | 82.3% | 49.2% | 93.7% | 5.2% | 66.80% | 1.63 | 81.0% | 14.3% |
| 50 | 0.0176 | 0.0915 | 1.0007107320540156 | 4.455 | 1.093 | 9.381 | 100.0% | 99.8% | 100.0% | 0.0% | 96.86% | 1.00 | 100.0% | 0.0% |
| 60 | 0.0108 | 0.0472 | 1.0 | 5.254 | 1.140 | 10.267 | 100.0% | 100.0% | 100.0% | 0.0% | 99.12% | 1.00 | 100.0% | 0.0% |
| 70 | 0.0060 | 0.0286 | 1.0007107320540156 | 6.188 | 1.317 | 11.589 | 100.0% | 99.9% | 100.0% | 0.0% | 98.88% | 1.00 | 100.0% | 0.0% |
| 80 | 0.0098 | 0.0717 | 1.0 | 5.183 | 1.555 | 11.000 | 100.0% | 99.8% | 100.0% | 0.0% | 98.38% | 1.00 | 100.0% | 0.0% |
| 90 | 0.0142 | 0.0704 | 1.0099502487562189 | 4.834 | 1.636 | 10.342 | 100.0% | 97.4% | 100.0% | 0.0% | 99.46% | 1.00 | 100.0% | 0.0% |


## Training halt-step distribution (per-sample counts, whole train set)
Counts of samples by recorded halting step. Column `step 5` mixes two cases the log cannot separate: halted exactly at step 5, and never halted (both are recorded as 5). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 45000 | 45000 | 0.08% |
| 10 | 0 | 0 | 0 | 0 | 45000 | 45000 | 6.82% |
| 20 | 0 | 0 | 0 | 0 | 45000 | 45000 | 8.22% |
| 30 | 45 | 24 | 2 | 0 | 44929 | 45000 | 23.06% |
| 40 | 42177 | 407 | 56 | 8 | 2352 | 45000 | 66.80% |
| 50 | 44999 | 1 | 0 | 0 | 0 | 45000 | 96.86% |
| 60 | 45000 | 0 | 0 | 0 | 0 | 45000 | 99.12% |
| 70 | 44999 | 1 | 0 | 0 | 0 | 45000 | 98.88% |
| 80 | 45000 | 0 | 0 | 0 | 0 | 45000 | 98.38% |
| 90 | 44993 | 4 | 1 | 0 | 2 | 45000 | 99.46% |

## Validation halt-step distribution (per-sample counts, whole validation set)
Counts of samples by recorded halting step. Column `step 5` mixes two cases the log cannot separate: halted exactly at step 5, and never halted (both are recorded as 5). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 5000 | 5000 | 0.08% |
| 10 | 0 | 0 | 0 | 0 | 5000 | 5000 | 6.82% |
| 20 | 0 | 0 | 0 | 0 | 5000 | 5000 | 8.22% |
| 30 | 3 | 3 | 0 | 0 | 4994 | 5000 | 23.06% |
| 40 | 4048 | 209 | 24 | 6 | 713 | 5000 | 66.80% |
| 50 | 5000 | 0 | 0 | 0 | 0 | 5000 | 96.86% |
| 60 | 5000 | 0 | 0 | 0 | 0 | 5000 | 99.12% |
| 70 | 5000 | 0 | 0 | 0 | 0 | 5000 | 98.88% |
| 80 | 5000 | 0 | 0 | 0 | 0 | 5000 | 98.38% |
| 90 | 5000 | 0 | 0 | 0 | 0 | 5000 | 99.46% |

## Validation halt-logit mean by supervision step
(first validation batch only; blank = validation halted every sample before reaching that step, so it was not executed; val acc is the epoch-level validation accuracy)

| epoch | s1 | s2 | s3 | s4 | s5 | val acc (epoch) |
|---|---|---|---|---|---|---|
| 0 | -7.166 | -6.851 | -6.982 | -6.974 | -6.979 | 0.08% |
| 10 | -2.712 | -2.608 | -2.579 | -2.574 | -2.572 | 6.82% |
| 20 | -2.334 | -2.320 | -2.331 | -2.326 | -2.331 | 8.22% |
| 30 | -1.466 | -1.458 | -1.461 | -1.462 | -1.463 | 23.06% |
| 40 | 0.360 | 0.436 | 0.441 | 0.441 | 0.441 | 66.80% |
| 50 | 3.877 |  |  |  |  | 96.86% |
| 60 | 5.392 |  |  |  |  | 99.12% |
| 70 | 5.618 |  |  |  |  | 98.88% |
| 80 | 4.385 |  |  |  |  | 98.38% |
| 90 | 6.331 |  |  |  |  | 99.46% |
