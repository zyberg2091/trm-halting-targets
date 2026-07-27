# softmean — halt-1.5  (halt > 1.5, n_sup = 5)

Q-loss: soft-mean of token correctness (pad-masked). Role: threshold sweep t=1.5.
Common settings: c=0.1, T=3, 100 epochs; metrics logged every 10 epochs (0–90).
Source: trm_qloss_softmean.ipynb, cell 25. Every value below is parsed verbatim from the run's printed log — nothing is recomputed.

## Per-epoch metrics

| epoch | train CE | Q-loss | avg steps (train, batch-level) | halt-logit mean | std | max | %logit>0 | %logit>1.5 | %step1 (train) | %never (train) | val acc | val steps mean | %step1 (val) | %never (val) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.6268 | 0.6183 | 5.0 | -0.752 | 0.418 | 0.083 | 0.1% | 0.0% | 0.0% | 100.0% | 0.08% | 5.00 | 0.0% | 100.0% |
| 10 | 0.7100 | 0.6067 | 5.0 | 0.881 | 0.251 | 1.792 | 99.8% | 0.3% | 0.4% | 99.3% | 6.44% | 4.97 | 0.5% | 99.1% |
| 20 | 0.5510 | 0.5381 | 5.0 | 1.218 | 0.203 | 2.009 | 100.0% | 7.9% | 7.3% | 89.4% | 8.32% | 4.55 | 7.2% | 87.4% |
| 30 | 0.5084 | 0.5127 | 5.0 | 1.335 | 0.179 | 2.062 | 100.0% | 18.9% | 16.9% | 76.9% | 12.44% | 3.77 | 25.8% | 67.5% |
| 40 | 0.2061 | 0.2070 | 1.9836531627576404 | 2.557 | 1.009 | 6.048 | 100.0% | 86.4% | 94.0% | 4.8% | 53.28% | 1.19 | 94.2% | 4.3% |
| 50 | 0.0435 | 0.0550 | 1.091684434968017 | 5.029 | 1.470 | 10.176 | 100.0% | 99.6% | 99.9% | 0.1% | 98.14% | 1.00 | 100.0% | 0.0% |
| 60 | 0.0119 | 0.0136 | 1.0 | 6.797 | 1.210 | 11.364 | 100.0% | 100.0% | 100.0% | 0.0% | 98.72% | 1.00 | 100.0% | 0.0% |
| 70 | 0.1709 | 0.1366 | 2.031272210376688 | 3.466 | 2.140 | 11.125 | 99.7% | 88.8% | 95.2% | 4.2% | 96.82% | 1.00 | 100.0% | 0.0% |
| 80 | 0.0540 | 0.0371 | 1.1620469083155651 | 6.224 | 2.277 | 12.019 | 99.6% | 94.8% | 98.7% | 1.1% | 42.82% | 3.16 | 43.9% | 53.2% |
| 90 | 0.0080 | 0.0171 | 1.0 | 7.213 | 1.807 | 14.270 | 100.0% | 100.0% | 100.0% | 0.0% | 98.12% | 1.00 | 100.0% | 0.0% |

### Column guide — what each column is and why it is tracked

| column | what it is (how it is computed) | why it is tracked |
|---|---|---|
| epoch | Training epoch at logging time (every 10th epoch, 0–90). | The time axis every trajectory is read against. |
| train CE | Per-token cross-entropy over not-yet-halted samples' tokens, averaged over the supervision steps the batch executed, then over batches. Pad positions add 0 to the numerator but are counted in the denominator. | Backbone learning progress — the competence signal that drives every halt target upward. |
| Q-loss | Binary cross-entropy between the halt logit h and this run's target t, batch-mean (halted samples included), accumulated like CE. | Shows whether the halt head is fitting its target; its level and turning points mark when the target starts saturating. |
| avg steps (train, batch-level) | How many supervision steps the batch executed before every sample in it had halted (or n_sup was reached), averaged over batches. | Recursion compute actually spent in training. Collapse pulls it toward 1. Batch-level: one stubborn sample holds the whole batch, so it upper-bounds the per-sample mean — the distribution tables below give the per-sample truth. |
| halt-logit mean | Mean of h over every (sample, executed supervision step) pair in the epoch; halted samples still contribute from frozen states. | The measurable form of the mechanism: the gradient sigma(h) − t pulls mean h toward logit(t), so this column tracks the equilibrium the analysis predicts. |
| std | Standard deviation of that same pool of h values. | Whether halting is sample-selective or one global decision — near-zero spread means the head treats all samples alike. |
| max | Largest h seen in the epoch. | The leading edge: the first samples to approach a cutoff show here before any percentage moves. |
| %logit>0, %logit>1.5 | Fraction of that pool above the fixed reference cutoffs 0 and 1.5 (printed for every run, independent of this run's own threshold). | Lets any run be read against both decision thresholds at once — e.g. a threshold-0 run's %>1.5 previews what a 1.5-threshold run would do. |
| %step1 (train) | Fraction of training samples whose first h > 1.5 happened at supervision step 1. | The headline collapse indicator: halting at step 1 means recursion is skipped from the start. For this run's soft-mean target, this jumps when mean token accuracy crosses the sigmoid of the threshold. |
| %never (train) | Fraction of training samples recorded at step 5 — halted-at-the-last-step and never-halted combined, so an upper bound on true never-halting. | The opposite failure mode: no halting at all, and the gradient-sparsity question. |
| val acc | Sequence exact-match on the 5,000 validation samples, each scored at the output of its own halting step (never-halted samples use the last executed step). | The outcome metric — what every halting behavior is ultimately traded against. |
| val steps mean | Mean recorded halting step over the validation set. | Recursion compute at evaluation; read next to val acc to price the halting decisions. |
| %step1 (val), %never (val) | The same two indicators as the train columns, measured with evaluation-time halting on the validation set. | Confirms the training-side halting behavior transfers to eval, where halting decides which step's answer is actually used. |

Fine print for every definition (populations, pad handling, the step-5 conflation): METRICS.md.

## Training halt-step distribution (per-sample counts, whole train set)
Counts of samples by recorded halting step. Column `step 5` mixes two cases the log cannot separate: halted exactly at step 5, and never halted (both are recorded as 5). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 45000 | 45000 | 0.08% |
| 10 | 181 | 127 | 21 | 0 | 44671 | 45000 | 6.44% |
| 20 | 3284 | 1375 | 109 | 16 | 40216 | 45000 | 8.32% |
| 30 | 7590 | 2599 | 163 | 26 | 34622 | 45000 | 12.44% |
| 40 | 42308 | 483 | 58 | 9 | 2142 | 45000 | 53.28% |
| 50 | 44951 | 13 | 0 | 1 | 35 | 45000 | 98.14% |
| 60 | 45000 | 0 | 0 | 0 | 0 | 45000 | 98.72% |
| 70 | 42851 | 245 | 25 | 2 | 1877 | 45000 | 96.82% |
| 80 | 44431 | 50 | 6 | 3 | 510 | 45000 | 42.82% |
| 90 | 45000 | 0 | 0 | 0 | 0 | 45000 | 98.12% |

## Validation halt-step distribution (per-sample counts, whole validation set)
Counts of samples by recorded halting step. Column `step 5` mixes two cases the log cannot separate: halted exactly at step 5, and never halted (both are recorded as 5). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 5000 | 5000 | 0.08% |
| 10 | 24 | 14 | 5 | 0 | 4957 | 5000 | 6.44% |
| 20 | 358 | 253 | 20 | 1 | 4368 | 5000 | 8.32% |
| 30 | 1291 | 304 | 27 | 4 | 3374 | 5000 | 12.44% |
| 40 | 4709 | 71 | 5 | 1 | 214 | 5000 | 53.28% |
| 50 | 5000 | 0 | 0 | 0 | 0 | 5000 | 98.14% |
| 60 | 5000 | 0 | 0 | 0 | 0 | 5000 | 98.72% |
| 70 | 5000 | 0 | 0 | 0 | 0 | 5000 | 96.82% |
| 80 | 2193 | 134 | 10 | 1 | 2662 | 5000 | 42.82% |
| 90 | 5000 | 0 | 0 | 0 | 0 | 5000 | 98.12% |

## Validation halt-logit mean by supervision step
(first validation batch only; blank = validation halted every sample before reaching that step, so it was not executed; val acc is the epoch-level validation accuracy)

| epoch | s1 | s2 | s3 | s4 | s5 | val acc (epoch) |
|---|---|---|---|---|---|---|
| 0 | -0.601 | -0.575 | -0.579 | -0.581 | -0.581 | 0.08% |
| 10 | 0.944 | 0.978 | 0.969 | 0.971 | 0.973 | 6.44% |
| 20 | 1.210 | 1.237 | 1.237 | 1.237 | 1.237 | 8.32% |
| 30 | 1.384 | 1.396 | 1.398 | 1.393 | 1.395 | 12.44% |
| 40 | 2.189 | 2.207 | 2.210 | 2.208 | 2.207 | 53.28% |
| 50 | 6.183 |  |  |  |  | 98.14% |
| 60 | 7.232 |  |  |  |  | 98.72% |
| 70 | 5.171 |  |  |  |  | 96.82% |
| 80 | 1.457 | 1.463 | 1.457 | 1.458 | 1.458 | 42.82% |
| 90 | 5.924 |  |  |  |  | 98.12% |
