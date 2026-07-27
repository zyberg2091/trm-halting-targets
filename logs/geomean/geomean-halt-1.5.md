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
| %step1 (train) | Fraction of training samples whose first h > 1.5 happened at supervision step 1. | The headline collapse indicator: halting at step 1 means recursion is skipped from the start. |
| %never (train) | Fraction of training samples recorded at step 5 — halted-at-the-last-step and never-halted combined, so an upper bound on true never-halting. | The opposite failure mode: no halting at all, and the gradient-sparsity question. |
| val acc | Sequence exact-match on the 5,000 validation samples, each scored at the output of its own halting step (never-halted samples use the last executed step). | The outcome metric — what every halting behavior is ultimately traded against. |
| val steps mean | Mean recorded halting step over the validation set. | Recursion compute at evaluation; read next to val acc to price the halting decisions. |
| %step1 (val), %never (val) | The same two indicators as the train columns, measured with evaluation-time halting on the validation set. | Confirms the training-side halting behavior transfers to eval, where halting decides which step's answer is actually used. |

Fine print for every definition (populations, pad handling, the step-5 conflation): METRICS.md.

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
