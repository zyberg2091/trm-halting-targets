# softmean — halt-0  (halt > 0, n_sup = 5)

Q-loss: soft-mean of token correctness (pad-masked). Role: threshold sweep t=0 (center cell).
Common settings: c=0.1, T=3, 100 epochs; metrics logged every 10 epochs (0–90).
Source: trm_qloss_softmean.ipynb, cell 24. Every value below is parsed verbatim from the run's printed log — nothing is recomputed.

## Per-epoch metrics

| epoch | train CE | Q-loss | avg steps (train, batch-level) | halt-logit mean | std | max | %logit>0 | %logit>1.5 | %step1 (train) | %never (train) | val acc | val steps mean | %step1 (val) | %never (val) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.6185 | 0.6229 | 5.0 | -0.743 | 0.402 | 0.267 | 0.3% | 0.0% | 1.4% | 98.6% | 0.08% | 4.97 | 0.0% | 99.1% |
| 10 | 0.9364 | 0.6574 | 1.953091684434968 | 0.381 | 0.286 | 1.861 | 90.6% | 0.0% | 96.4% | 3.2% | 2.52% | 1.06 | 98.2% | 1.5% |
| 20 | 0.5481 | 0.5377 | 1.0 | 1.220 | 0.196 | 1.926 | 100.0% | 7.8% | 100.0% | 0.0% | 9.18% | 1.00 | 100.0% | 0.0% |
| 30 | 0.3628 | 0.4190 | 1.0 | 1.767 | 0.322 | 3.216 | 100.0% | 79.9% | 100.0% | 0.0% | 44.58% | 1.00 | 100.0% | 0.0% |
| 40 | 0.0341 | 0.0604 | 1.0 | 4.832 | 0.907 | 8.342 | 100.0% | 100.0% | 100.0% | 0.0% | 95.44% | 1.00 | 100.0% | 0.0% |
| 50 | 0.0253 | 0.0458 | 1.0 | 5.457 | 1.238 | 9.940 | 100.0% | 100.0% | 100.0% | 0.0% | 97.34% | 1.00 | 100.0% | 0.0% |
| 60 | 0.0223 | 0.0364 | 1.0 | 5.836 | 1.358 | 10.234 | 100.0% | 100.0% | 100.0% | 0.0% | 99.66% | 1.00 | 100.0% | 0.0% |
| 70 | 0.0387 | 0.0446 | 1.0021321961620469 | 5.909 | 1.637 | 10.760 | 100.0% | 99.6% | 100.0% | 0.0% | 99.68% | 1.00 | 100.0% | 0.0% |
| 80 | 0.0118 | 0.0236 | 1.0 | 6.425 | 1.305 | 11.255 | 100.0% | 100.0% | 100.0% | 0.0% | 96.86% | 1.00 | 100.0% | 0.0% |
| 90 | 0.0157 | 0.0272 | 1.0 | 6.152 | 1.300 | 10.542 | 100.0% | 100.0% | 100.0% | 0.0% | 97.26% | 1.00 | 100.0% | 0.0% |

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
| %step1 (train) | Fraction of training samples whose first h > 0 happened at supervision step 1. | The headline collapse indicator: halting at step 1 means recursion is skipped from the start. For this run's soft-mean target, this jumps when mean token accuracy crosses the sigmoid of the threshold. |
| %never (train) | Fraction of training samples recorded at step 5 — halted-at-the-last-step and never-halted combined, so an upper bound on true never-halting. | The opposite failure mode: no halting at all, and the gradient-sparsity question. |
| val acc | Sequence exact-match on the 5,000 validation samples, each scored at the output of its own halting step (never-halted samples use the last executed step). | The outcome metric — what every halting behavior is ultimately traded against. |
| val steps mean | Mean recorded halting step over the validation set. | Recursion compute at evaluation; read next to val acc to price the halting decisions. |
| %step1 (val), %never (val) | The same two indicators as the train columns, measured with evaluation-time halting on the validation set. | Confirms the training-side halting behavior transfers to eval, where halting decides which step's answer is actually used. |


## Training halt-step distribution (per-sample counts, whole train set)
Counts of samples by recorded halting step. Column `step 5` mixes two cases the log cannot separate: halted exactly at step 5, and never halted (both are recorded as 5). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|
| 0 | 613 | 0 | 0 | 1 | 44386 | 45000 | 0.08% |
| 10 | 43358 | 181 | 16 | 0 | 1445 | 45000 | 2.52% |
| 20 | 45000 | 0 | 0 | 0 | 0 | 45000 | 9.18% |
| 30 | 45000 | 0 | 0 | 0 | 0 | 45000 | 44.58% |
| 40 | 45000 | 0 | 0 | 0 | 0 | 45000 | 95.44% |
| 50 | 45000 | 0 | 0 | 0 | 0 | 45000 | 97.34% |
| 60 | 45000 | 0 | 0 | 0 | 0 | 45000 | 99.66% |
| 70 | 44998 | 1 | 1 | 0 | 0 | 45000 | 99.68% |
| 80 | 45000 | 0 | 0 | 0 | 0 | 45000 | 96.86% |
| 90 | 45000 | 0 | 0 | 0 | 0 | 45000 | 97.26% |

## Validation halt-step distribution (per-sample counts, whole validation set)
Counts of samples by recorded halting step. Column `step 5` mixes two cases the log cannot separate: halted exactly at step 5, and never halted (both are recorded as 5). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 46 | 0 | 0 | 4954 | 5000 | 0.08% |
| 10 | 4908 | 19 | 0 | 0 | 73 | 5000 | 2.52% |
| 20 | 5000 | 0 | 0 | 0 | 0 | 5000 | 9.18% |
| 30 | 5000 | 0 | 0 | 0 | 0 | 5000 | 44.58% |
| 40 | 5000 | 0 | 0 | 0 | 0 | 5000 | 95.44% |
| 50 | 5000 | 0 | 0 | 0 | 0 | 5000 | 97.34% |
| 60 | 5000 | 0 | 0 | 0 | 0 | 5000 | 99.66% |
| 70 | 5000 | 0 | 0 | 0 | 0 | 5000 | 99.68% |
| 80 | 5000 | 0 | 0 | 0 | 0 | 5000 | 96.86% |
| 90 | 5000 | 0 | 0 | 0 | 0 | 5000 | 97.26% |

## Validation halt-logit mean by supervision step
(first validation batch only; blank = validation halted every sample before reaching that step, so it was not executed; val acc is the epoch-level validation accuracy)

| epoch | s1 | s2 | s3 | s4 | s5 | val acc (epoch) |
|---|---|---|---|---|---|---|
| 0 | -0.473 | -0.455 | -0.465 | -0.461 | -0.463 | 0.08% |
| 10 | 0.354 | 0.345 |  |  |  | 2.52% |
| 20 | 1.207 |  |  |  |  | 9.18% |
| 30 | 2.083 |  |  |  |  | 44.58% |
| 40 | 4.816 |  |  |  |  | 95.44% |
| 50 | 5.092 |  |  |  |  | 97.34% |
| 60 | 7.288 |  |  |  |  | 99.66% |
| 70 | 6.995 |  |  |  |  | 99.68% |
| 80 | 5.286 |  |  |  |  | 96.86% |
| 90 | 5.511 |  |  |  |  | 97.26% |
