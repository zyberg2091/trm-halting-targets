# binary-em — halt-1.5  (halt > 1.5, n_sup = 5)

Q-loss: binary exact-match: target 1 iff every real token correct (pad-masked). Role: threshold sweep t=1.5.
Common settings: c=0.1, T=3, 100 epochs; metrics logged every 10 epochs (0–90).
Source: trm_qloss_binary_em.ipynb, cell 24. Every value below is parsed verbatim from the run's printed log — nothing is recomputed.

## Per-epoch metrics

| epoch | train CE | Q-loss | avg steps (train, batch-level) | halt-logit mean | std | max | %logit>0 | %logit>1.5 | %step1 (train) | %never (train) | val acc | val steps mean | %step1 (val) | %never (val) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.6255 | 0.0200 | 5.0 | -7.781 | 2.765 | 0.047 | 0.1% | 0.0% | 0.0% | 100.0% | 0.06% | 5.00 | 0.0% | 100.0% |
| 10 | 0.6819 | 0.2585 | 5.0 | -2.711 | 0.569 | -0.184 | 0.0% | 0.0% | 0.0% | 100.0% | 6.86% | 5.00 | 0.0% | 100.0% |
| 20 | 0.5514 | 0.3359 | 5.0 | -2.202 | 0.345 | -0.773 | 0.0% | 0.0% | 0.0% | 100.0% | 9.22% | 5.00 | 0.0% | 100.0% |
| 30 | 0.4984 | 0.4421 | 5.0 | -1.696 | 0.346 | -0.298 | 0.0% | 0.0% | 0.0% | 100.0% | 16.36% | 5.00 | 0.0% | 100.0% |
| 40 | 0.1610 | 0.4122 | 4.909026297085998 | 1.865 | 0.659 | 4.327 | 99.3% | 72.7% | 67.6% | 23.0% | 88.08% | 1.48 | 84.1% | 10.4% |
| 50 | 0.1873 | 0.3747 | 4.633262260127932 | 2.039 | 0.773 | 4.505 | 99.1% | 76.9% | 74.6% | 18.2% | 86.14% | 1.23 | 91.6% | 4.7% |
| 60 | 0.1206 | 0.3260 | 4.384506041222459 | 2.274 | 0.713 | 4.986 | 99.9% | 85.4% | 84.6% | 11.0% | 92.32% | 1.07 | 96.6% | 1.0% |
| 70 | 0.1728 | 0.3335 | 4.565031982942431 | 2.245 | 0.739 | 4.586 | 99.2% | 85.1% | 84.3% | 11.7% | 79.48% | 1.28 | 90.5% | 6.1% |
| 80 | 0.1327 | 0.3194 | 4.442786069651741 | 2.332 | 0.768 | 5.568 | 99.8% | 86.0% | 85.7% | 10.9% | 90.64% | 1.18 | 94.3% | 4.1% |
| 90 | 0.0833 | 0.2302 | 3.037668798862829 | 2.611 | 1.438 | 8.786 | 99.1% | 82.7% | 88.5% | 9.5% | 87.56% | 1.42 | 87.9% | 10.0% |


## Training halt-step distribution (per-sample counts, whole train set)
Counts of samples by recorded halting step. Column `step 5` mixes two cases the log cannot separate: halted exactly at step 5, and never halted (both are recorded as 5). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 45000 | 45000 | 0.06% |
| 10 | 0 | 0 | 0 | 0 | 45000 | 45000 | 6.86% |
| 20 | 0 | 0 | 0 | 0 | 45000 | 45000 | 9.22% |
| 30 | 0 | 0 | 0 | 0 | 45000 | 45000 | 16.36% |
| 40 | 30402 | 3725 | 448 | 64 | 10361 | 45000 | 88.08% |
| 50 | 33583 | 2881 | 300 | 35 | 8201 | 45000 | 86.14% |
| 60 | 38052 | 1845 | 162 | 12 | 4929 | 45000 | 92.32% |
| 70 | 37938 | 1645 | 140 | 26 | 5251 | 45000 | 79.48% |
| 80 | 38566 | 1397 | 103 | 30 | 4904 | 45000 | 90.64% |
| 90 | 39827 | 817 | 78 | 15 | 4263 | 45000 | 87.56% |

## Validation halt-step distribution (per-sample counts, whole validation set)
Counts of samples by recorded halting step. Column `step 5` mixes two cases the log cannot separate: halted exactly at step 5, and never halted (both are recorded as 5). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 5000 | 5000 | 0.06% |
| 10 | 0 | 0 | 0 | 0 | 5000 | 5000 | 6.86% |
| 20 | 0 | 0 | 0 | 0 | 5000 | 5000 | 9.22% |
| 30 | 0 | 0 | 0 | 0 | 5000 | 5000 | 16.36% |
| 40 | 4203 | 222 | 47 | 9 | 519 | 5000 | 88.08% |
| 50 | 4579 | 156 | 25 | 5 | 235 | 5000 | 86.14% |
| 60 | 4832 | 111 | 5 | 0 | 52 | 5000 | 92.32% |
| 70 | 4526 | 159 | 8 | 0 | 307 | 5000 | 79.48% |
| 80 | 4714 | 74 | 5 | 0 | 207 | 5000 | 90.64% |
| 90 | 4393 | 98 | 7 | 0 | 502 | 5000 | 87.56% |

## Validation halt-logit mean by supervision step
(first validation batch only; blank = validation halted every sample before reaching that step, so it was not executed; val acc is the epoch-level validation accuracy)

| epoch | s1 | s2 | s3 | s4 | s5 | val acc (epoch) |
|---|---|---|---|---|---|---|
| 0 | -7.339 | -7.017 | -7.118 | -7.158 | -7.129 | 0.06% |
| 10 | -2.596 | -2.633 | -2.572 | -2.585 | -2.607 | 6.86% |
| 20 | -2.306 | -2.345 | -2.338 | -2.336 | -2.334 | 9.22% |
| 30 | -1.609 | -1.636 | -1.632 | -1.631 | -1.630 | 16.36% |
| 40 | 2.017 | 2.143 | 2.143 | 2.135 | 2.139 | 88.08% |
| 50 | 2.299 | 2.420 | 2.412 | 2.409 | 2.413 | 86.14% |
| 60 | 2.587 | 2.714 |  |  |  | 92.32% |
| 70 | 2.387 | 2.499 | 2.448 | 2.458 | 2.466 | 79.48% |
| 80 | 2.465 | 2.580 | 2.550 | 2.546 | 2.542 | 90.64% |
| 90 | 2.128 | 2.227 | 2.205 | 2.202 | 2.206 | 87.56% |
