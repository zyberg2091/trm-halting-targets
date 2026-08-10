# geomean — halt-3.0  (halt > 3, n_sup = 5)

Q-loss: geometric mean of correct-token probabilities = exp(-mean CE) (pad-masked). Role: threshold sweep t=3.0.
Common settings: c=0.1, T=3, 100 epochs; metrics logged every 10 epochs (0–90).
Source: trm_qloss_geomean.ipynb, cell 26. Every value below is parsed verbatim from the run's printed log — nothing is recomputed.

## Per-epoch metrics

| epoch | train CE | Q-loss | avg steps (train, batch-level) | halt-logit mean | std | max | %logit>0 | %logit>1.5 | %step1 (train) | %never (train) | val acc | val steps mean | %step1 (val) | %never (val) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.6290 | 0.4681 | 5.0 | -1.543 | 0.400 | 0.008 | 0.0% | 0.0% | 0.0% | 100.0% | 0.06% | 5.00 | 0.0% | 100.0% |
| 10 | 0.6612 | 0.6912 | 5.0 | -0.014 | 0.211 | 0.679 | 49.4% | 0.0% | 0.0% | 100.0% | 7.34% | 5.00 | 0.0% | 100.0% |
| 20 | 0.5535 | 0.6854 | 5.0 | 0.220 | 0.171 | 0.819 | 89.3% | 0.0% | 0.0% | 100.0% | 8.54% | 5.00 | 0.0% | 100.0% |
| 30 | 0.1939 | 0.4282 | 5.0 | 1.736 | 0.459 | 3.581 | 100.0% | 69.8% | 0.0% | 99.6% | 75.18% | 5.00 | 0.0% | 99.9% |
| 40 | 0.1148 | 0.2440 | 5.0 | 2.793 | 0.744 | 5.071 | 99.5% | 94.6% | 28.0% | 49.2% | 87.70% | 2.68 | 42.0% | 36.1% |
| 50 | 0.0862 | 0.1980 | 4.994314143567875 | 3.112 | 0.686 | 5.423 | 100.0% | 98.3% | 48.5% | 34.5% | 92.72% | 2.49 | 46.3% | 31.1% |
| 60 | 0.0749 | 0.1803 | 4.953802416488983 | 3.258 | 0.685 | 5.562 | 99.9% | 98.7% | 57.8% | 26.3% | 91.72% | 2.05 | 62.7% | 22.2% |
| 70 | 0.0778 | 0.1651 | 4.912579957356077 | 3.422 | 0.742 | 6.050 | 100.0% | 98.8% | 65.9% | 21.3% | 88.92% | 3.38 | 29.6% | 55.2% |
| 80 | 0.0746 | 0.1552 | 4.810234541577826 | 3.462 | 0.699 | 6.143 | 100.0% | 99.2% | 70.7% | 18.9% | 92.54% | 1.71 | 76.3% | 15.5% |
| 90 | 0.0673 | 0.1325 | 4.667377398720682 | 3.715 | 0.793 | 6.280 | 100.0% | 99.3% | 80.0% | 14.0% | 95.54% | 1.73 | 75.0% | 15.7% |

### Column guide — what each column is and why it is tracked

| column | what it is (how it is computed) | why it is tracked |
|---|---|---|
| epoch | Training epoch at logging time (every 10th epoch, 0–90). | The time axis every trajectory is read against. |
| train CE | Per-token cross-entropy over not-yet-halted samples' tokens, averaged over the supervision steps the batch executed, then over batches. Pad positions add 0 to the numerator but are counted in the denominator. | Backbone learning progress — the competence signal that drives every halt target upward. For this run's geo-mean target, the halt target equals exp(−mean CE), so CE = 0.693 is exactly where the target crosses 0.5. |
| Q-loss | Binary cross-entropy between the halt logit h and this run's target t, batch-mean (halted samples included), accumulated like CE. | Shows whether the halt head is fitting its target; its level and turning points mark when the target starts saturating. |
| avg steps (train, batch-level) | How many supervision steps the batch executed before every sample in it had halted (or n_sup was reached), averaged over batches. | Recursion compute actually spent in training. Collapse pulls it toward 1. Batch-level: one stubborn sample holds the whole batch, so it upper-bounds the per-sample mean — the distribution tables below give the per-sample truth. |
| halt-logit mean | Mean of h over every (sample, executed supervision step) pair in the epoch; halted samples still contribute from frozen states. | The measurable form of the mechanism: the gradient sigma(h) − t pulls mean h toward logit(t), so this column tracks the equilibrium the analysis predicts. |
| std | Standard deviation of that same pool of h values. | Whether halting is sample-selective or one global decision — near-zero spread means the head treats all samples alike. |
| max | Largest h seen in the epoch. | The leading edge: the first samples to approach a cutoff show here before any percentage moves. |
| %logit>0, %logit>1.5 | Fraction of that pool above the fixed reference cutoffs 0 and 1.5 (printed for every run, independent of this run's own threshold). | Lets any run be read against both decision thresholds at once — e.g. a threshold-0 run's %>1.5 previews what a 1.5-threshold run would do. |
| %step1 (train) | Fraction of training samples whose first h > 3 happened at supervision step 1. | The headline collapse indicator: halting at step 1 means recursion is skipped from the start. |
| %never (train) | Fraction of training samples recorded at step 5 — halted-at-the-last-step and never-halted combined, so an upper bound on true never-halting. | The opposite failure mode: no halting at all, and the gradient-sparsity question. |
| val acc | Sequence exact-match on the 5,000 validation samples, each scored at the output of its own halting step (never-halted samples use the last executed step). | The outcome metric — what every halting behavior is ultimately traded against. |
| val steps mean | Mean recorded halting step over the validation set. | Recursion compute at evaluation; read next to val acc to price the halting decisions. |
| %step1 (val), %never (val) | The same two indicators as the train columns, measured with evaluation-time halting on the validation set. | Confirms the training-side halting behavior transfers to eval, where halting decides which step's answer is actually used. |


## Training halt-step distribution (per-sample counts, whole train set)
Counts of samples by recorded halting step. Column `step 5` mixes two cases the log cannot separate: halted exactly at step 5, and never halted (both are recorded as 5). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 45000 | 45000 | 0.06% |
| 10 | 0 | 0 | 0 | 0 | 45000 | 45000 | 7.34% |
| 20 | 0 | 0 | 0 | 0 | 45000 | 45000 | 8.54% |
| 30 | 14 | 167 | 14 | 4 | 44801 | 45000 | 75.18% |
| 40 | 12605 | 9474 | 656 | 106 | 22159 | 45000 | 87.70% |
| 50 | 21834 | 7044 | 486 | 100 | 15536 | 45000 | 92.72% |
| 60 | 26030 | 6598 | 436 | 96 | 11840 | 45000 | 91.72% |
| 70 | 29668 | 5360 | 340 | 60 | 9572 | 45000 | 88.92% |
| 80 | 31798 | 4344 | 311 | 60 | 8487 | 45000 | 92.54% |
| 90 | 36008 | 2479 | 180 | 23 | 6310 | 45000 | 95.54% |

## Validation halt-step distribution (per-sample counts, whole validation set)
Counts of samples by recorded halting step. Column `step 5` mixes two cases the log cannot separate: halted exactly at step 5, and never halted (both are recorded as 5). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 5000 | 5000 | 0.06% |
| 10 | 0 | 0 | 0 | 0 | 5000 | 5000 | 7.34% |
| 20 | 0 | 0 | 0 | 0 | 5000 | 5000 | 8.54% |
| 30 | 0 | 6 | 0 | 1 | 4993 | 5000 | 75.18% |
| 40 | 2099 | 1013 | 67 | 15 | 1806 | 5000 | 87.70% |
| 50 | 2317 | 1052 | 64 | 12 | 1555 | 5000 | 92.72% |
| 60 | 3135 | 690 | 57 | 10 | 1108 | 5000 | 91.72% |
| 70 | 1480 | 685 | 67 | 8 | 2760 | 5000 | 88.92% |
| 80 | 3816 | 381 | 24 | 3 | 776 | 5000 | 92.54% |
| 90 | 3748 | 426 | 35 | 5 | 786 | 5000 | 95.54% |

## Validation halt-logit mean by supervision step
(first validation batch only; blank = validation halted every sample before reaching that step, so it was not executed; val acc is the epoch-level validation accuracy)

| epoch | s1 | s2 | s3 | s4 | s5 | val acc (epoch) |
|---|---|---|---|---|---|---|
| 0 | -1.366 | -1.269 | -1.297 | -1.288 | -1.292 | 0.06% |
| 10 | 0.037 | 0.053 | 0.040 | 0.036 | 0.040 | 7.34% |
| 20 | 0.222 | 0.225 | 0.229 | 0.228 | 0.225 | 8.54% |
| 30 | 1.765 | 1.974 | 1.950 | 1.948 | 1.950 | 75.18% |
| 40 | 2.959 | 3.222 | 3.190 | 3.188 | 3.188 | 87.70% |
| 50 | 3.097 | 3.261 | 3.213 | 3.228 | 3.222 | 92.72% |
| 60 | 3.209 | 3.349 | 3.334 | 3.333 | 3.330 | 91.72% |
| 70 | 2.597 | 2.806 | 2.774 | 2.775 | 2.772 | 88.92% |
| 80 | 3.430 | 3.603 | 3.596 | 3.581 | 3.579 | 92.54% |
| 90 | 3.581 | 3.639 | 3.632 | 3.621 | 3.626 | 95.54% |
