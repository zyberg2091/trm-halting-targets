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
| %never (train) | Fraction of training samples recorded at step 1 — halted-at-the-last-step and never-halted combined, so an upper bound on true never-halting. | The opposite failure mode: no halting at all, and the gradient-sparsity question. |
| val acc | Sequence exact-match on the 5,000 validation samples, each scored at the output of its own halting step (never-halted samples use the last executed step). | The outcome metric — what every halting behavior is ultimately traded against. |
| val steps mean | Mean recorded halting step over the validation set. | Recursion compute at evaluation; read next to val acc to price the halting decisions. |
| %step1 (val), %never (val) | The same two indicators as the train columns, measured with evaluation-time halting on the validation set. | Confirms the training-side halting behavior transfers to eval, where halting decides which step's answer is actually used. |


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
