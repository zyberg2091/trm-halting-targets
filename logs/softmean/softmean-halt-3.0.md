# softmean — halt-3.0  (halt > 3, n_sup = 5)

Q-loss: soft-mean of token correctness (pad-masked). Role: threshold sweep t=3.0.
Common settings: c=0.1, T=3, 100 epochs; metrics logged every 10 epochs (0–90).
Source: trm_qloss_softmean.ipynb, cell 27. Every value below is parsed verbatim from the run's printed log — nothing is recomputed.

## Per-epoch metrics

| epoch | train CE | Q-loss | avg steps (train, batch-level) | halt-logit mean | std | max | %logit>0 | %logit>1.5 | %step1 (train) | %never (train) | val acc | val steps mean | %step1 (val) | %never (val) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.6237 | 0.6191 | 5.0 | -0.751 | 0.416 | 0.109 | 0.3% | 0.0% | 0.0% | 100.0% | 0.08% | 5.00 | 0.0% | 100.0% |
| 10 | 0.6891 | 0.6004 | 5.0 | 0.913 | 0.256 | 1.837 | 99.9% | 0.7% | 0.0% | 100.0% | 6.62% | 5.00 | 0.0% | 100.0% |
| 20 | 0.5531 | 0.5397 | 5.0 | 1.210 | 0.201 | 2.054 | 100.0% | 6.9% | 0.0% | 100.0% | 9.00% | 5.00 | 0.0% | 100.0% |
| 30 | 0.5112 | 0.5138 | 5.0 | 1.328 | 0.184 | 2.128 | 100.0% | 18.2% | 0.0% | 100.0% | 13.52% | 5.00 | 0.0% | 100.0% |
| 40 | 0.1689 | 0.1936 | 4.9815209665955935 | 3.094 | 0.558 | 5.465 | 100.0% | 99.3% | 52.6% | 33.8% | 75.54% | 2.04 | 65.2% | 22.2% |
| 50 | 0.1362 | 0.1518 | 4.844349680170576 | 3.464 | 0.604 | 5.870 | 100.0% | 99.7% | 74.4% | 16.2% | 77.50% | 2.59 | 51.4% | 36.5% |
| 60 | 0.1672 | 0.1510 | 4.7690120824449185 | 3.489 | 0.693 | 5.912 | 100.0% | 99.2% | 73.1% | 17.0% | 85.78% | 2.61 | 53.4% | 37.7% |
| 70 | 0.1166 | 0.1158 | 4.550817341862118 | 3.853 | 0.734 | 6.773 | 100.0% | 99.9% | 85.9% | 9.2% | 90.04% | 1.25 | 90.3% | 4.8% |
| 80 | 0.0963 | 0.0996 | 3.616915422885572 | 4.012 | 0.802 | 7.344 | 100.0% | 99.9% | 90.9% | 6.0% | 87.28% | 1.72 | 77.6% | 16.3% |
| 90 | 0.0923 | 0.0924 | 3.8272921108742004 | 4.162 | 0.853 | 7.324 | 100.0% | 99.9% | 91.8% | 5.1% | 86.60% | 1.35 | 88.3% | 7.6% |

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
| %step1 (train) | Fraction of training samples whose first h > 3 happened at supervision step 1. | The headline collapse indicator: halting at step 1 means recursion is skipped from the start. For this run's soft-mean target, this jumps when mean token accuracy crosses the sigmoid of the threshold. |
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
| 10 | 0 | 0 | 0 | 0 | 45000 | 45000 | 6.62% |
| 20 | 0 | 0 | 0 | 0 | 45000 | 45000 | 9.00% |
| 30 | 0 | 0 | 0 | 0 | 45000 | 45000 | 13.52% |
| 40 | 23686 | 5147 | 755 | 202 | 15210 | 45000 | 75.54% |
| 50 | 33484 | 3850 | 307 | 64 | 7295 | 45000 | 77.50% |
| 60 | 32878 | 4064 | 332 | 63 | 7663 | 45000 | 85.78% |
| 70 | 38647 | 2004 | 188 | 34 | 4127 | 45000 | 90.04% |
| 80 | 40885 | 1279 | 116 | 16 | 2704 | 45000 | 87.28% |
| 90 | 41291 | 1281 | 112 | 26 | 2290 | 45000 | 86.60% |

## Validation halt-step distribution (per-sample counts, whole validation set)
Counts of samples by recorded halting step. Column `step 5` mixes two cases the log cannot separate: halted exactly at step 5, and never halted (both are recorded as 5). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 5000 | 5000 | 0.08% |
| 10 | 0 | 0 | 0 | 0 | 5000 | 5000 | 6.62% |
| 20 | 0 | 0 | 0 | 0 | 5000 | 5000 | 9.00% |
| 30 | 0 | 0 | 0 | 0 | 5000 | 5000 | 13.52% |
| 40 | 3259 | 540 | 71 | 18 | 1112 | 5000 | 75.54% |
| 50 | 2572 | 553 | 39 | 10 | 1826 | 5000 | 77.50% |
| 60 | 2668 | 394 | 43 | 9 | 1886 | 5000 | 85.78% |
| 70 | 4514 | 225 | 19 | 1 | 241 | 5000 | 90.04% |
| 80 | 3878 | 290 | 14 | 2 | 816 | 5000 | 87.28% |
| 90 | 4413 | 190 | 14 | 2 | 381 | 5000 | 86.60% |

## Validation halt-logit mean by supervision step
(first validation batch only; blank = validation halted every sample before reaching that step, so it was not executed; val acc is the epoch-level validation accuracy)

| epoch | s1 | s2 | s3 | s4 | s5 | val acc (epoch) |
|---|---|---|---|---|---|---|
| 0 | -0.699 | -0.665 | -0.654 | -0.662 | -0.658 | 0.08% |
| 10 | 0.854 | 0.954 | 0.930 | 0.947 | 0.931 | 6.62% |
| 20 | 1.285 | 1.302 | 1.302 | 1.303 | 1.300 | 9.00% |
| 30 | 1.401 | 1.396 | 1.407 | 1.405 | 1.404 | 13.52% |
| 40 | 3.346 | 3.441 | 3.469 | 3.482 | 3.480 | 75.54% |
| 50 | 3.189 | 3.253 | 3.222 | 3.245 | 3.241 | 77.50% |
| 60 | 3.217 | 3.334 | 3.335 | 3.320 | 3.316 | 85.78% |
| 70 | 3.955 | 3.955 | 3.973 | 3.971 | 3.976 | 90.04% |
| 80 | 3.618 | 3.681 | 3.699 | 3.690 | 3.684 | 87.28% |
| 90 | 3.956 | 4.026 |  |  |  | 86.60% |
