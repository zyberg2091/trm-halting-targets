# binary-em — halt-3.0  (halt > 3, n_sup = 5)

Q-loss: binary exact-match: target 1 iff every real token correct (pad-masked). Role: threshold sweep t=3.0.
Common settings: c=0.1, T=3, 100 epochs; metrics logged every 10 epochs (0–90).
Source: trm_qloss_binary_em.ipynb, cell 25. Every value below is parsed verbatim from the run's printed log — nothing is recomputed.

## Per-epoch metrics

| epoch | train CE | Q-loss | avg steps (train, batch-level) | halt-logit mean | std | max | %logit>0 | %logit>1.5 | %step1 (train) | %never (train) | val acc | val steps mean | %step1 (val) | %never (val) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.6359 | 0.0190 | 5.0 | -7.495 | 2.398 | 0.008 | 0.0% | 0.0% | 0.0% | 100.0% | 0.10% | 5.00 | 0.0% | 100.0% |
| 10 | 0.6654 | 0.2644 | 5.0 | -2.674 | 0.536 | -0.668 | 0.0% | 0.0% | 0.0% | 100.0% | 7.12% | 5.00 | 0.0% | 100.0% |
| 20 | 0.5525 | 0.3331 | 5.0 | -2.220 | 0.357 | -0.578 | 0.0% | 0.0% | 0.0% | 100.0% | 8.36% | 5.00 | 0.0% | 100.0% |
| 30 | 0.5253 | 0.3705 | 5.0 | -2.025 | 0.321 | -0.401 | 0.0% | 0.0% | 0.0% | 100.0% | 9.36% | 5.00 | 0.0% | 100.0% |
| 40 | 0.1183 | 0.3974 | 5.0 | 1.934 | 0.757 | 4.437 | 98.4% | 75.0% | 3.6% | 91.8% | 89.68% | 4.71 | 5.1% | 91.7% |
| 50 | 0.0859 | 0.2695 | 5.0 | 2.775 | 0.959 | 5.753 | 99.3% | 89.9% | 36.4% | 50.1% | 94.10% | 2.81 | 43.9% | 41.0% |
| 60 | 0.0580 | 0.2031 | 4.994314143567875 | 3.182 | 0.885 | 6.170 | 99.8% | 96.0% | 53.8% | 33.9% | 93.80% | 1.87 | 71.6% | 19.1% |
| 70 | 0.0708 | 0.2000 | 4.979388770433546 | 3.224 | 0.916 | 6.205 | 99.8% | 95.4% | 56.6% | 31.6% | 92.50% | 1.81 | 70.3% | 16.7% |
| 80 | 0.0762 | 0.1886 | 4.982942430703624 | 3.343 | 1.000 | 6.578 | 99.7% | 95.2% | 60.1% | 28.7% | 87.94% | 2.29 | 57.4% | 28.2% |
| 90 | 0.0652 | 0.1687 | 4.9502487562189055 | 3.526 | 0.897 | 6.942 | 99.9% | 98.1% | 67.9% | 21.8% | 81.12% | 4.89 | 1.1% | 96.7% |

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
| %step1 (train) | Fraction of training samples whose first h > 3 happened at supervision step 1. | The headline collapse indicator: halting at step 1 means recursion is skipped from the start. |
| %never (train) | Fraction of training samples recorded at step 5 — halted-at-the-last-step and never-halted combined, so an upper bound on true never-halting. | The opposite failure mode: no halting at all, and the gradient-sparsity question. For this run's binary-EM target, 100% here early is the expected all-zero-target phase: no sample is fully correct yet, so nothing should halt. |
| val acc | Sequence exact-match on the 5,000 validation samples, each scored at the output of its own halting step (never-halted samples use the last executed step). | The outcome metric — what every halting behavior is ultimately traded against. |
| val steps mean | Mean recorded halting step over the validation set. | Recursion compute at evaluation; read next to val acc to price the halting decisions. |
| %step1 (val), %never (val) | The same two indicators as the train columns, measured with evaluation-time halting on the validation set. | Confirms the training-side halting behavior transfers to eval, where halting decides which step's answer is actually used. |


## Training halt-step distribution (per-sample counts, whole train set)
Counts of samples by recorded halting step. Column `step 5` mixes two cases the log cannot separate: halted exactly at step 5, and never halted (both are recorded as 5). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 45000 | 45000 | 0.10% |
| 10 | 0 | 0 | 0 | 0 | 45000 | 45000 | 7.12% |
| 20 | 0 | 0 | 0 | 0 | 45000 | 45000 | 8.36% |
| 30 | 0 | 0 | 0 | 0 | 45000 | 45000 | 9.36% |
| 40 | 1636 | 1875 | 171 | 28 | 41290 | 45000 | 89.68% |
| 50 | 16387 | 5495 | 510 | 66 | 22542 | 45000 | 94.10% |
| 60 | 24224 | 4804 | 605 | 91 | 15276 | 45000 | 93.80% |
| 70 | 25478 | 4720 | 493 | 85 | 14224 | 45000 | 92.50% |
| 80 | 27049 | 4518 | 449 | 91 | 12893 | 45000 | 87.94% |
| 90 | 30544 | 4165 | 373 | 97 | 9821 | 45000 | 81.12% |

## Validation halt-step distribution (per-sample counts, whole validation set)
Counts of samples by recorded halting step. Column `step 5` mixes two cases the log cannot separate: halted exactly at step 5, and never halted (both are recorded as 5). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 5000 | 5000 | 0.10% |
| 10 | 0 | 0 | 0 | 0 | 5000 | 5000 | 7.12% |
| 20 | 0 | 0 | 0 | 0 | 5000 | 5000 | 8.36% |
| 30 | 0 | 0 | 0 | 0 | 5000 | 5000 | 9.36% |
| 40 | 253 | 136 | 23 | 5 | 4583 | 5000 | 89.68% |
| 50 | 2197 | 659 | 84 | 9 | 2051 | 5000 | 94.10% |
| 60 | 3578 | 420 | 44 | 3 | 955 | 5000 | 93.80% |
| 70 | 3515 | 579 | 62 | 9 | 835 | 5000 | 92.50% |
| 80 | 2871 | 654 | 50 | 16 | 1409 | 5000 | 87.94% |
| 90 | 56 | 88 | 18 | 5 | 4833 | 5000 | 81.12% |

## Validation halt-logit mean by supervision step
(first validation batch only; blank = validation halted every sample before reaching that step, so it was not executed; val acc is the epoch-level validation accuracy)

| epoch | s1 | s2 | s3 | s4 | s5 | val acc (epoch) |
|---|---|---|---|---|---|---|
| 0 | -7.797 | -7.443 | -7.565 | -7.518 | -7.530 | 0.10% |
| 10 | -2.792 | -2.645 | -2.644 | -2.656 | -2.659 | 7.12% |
| 20 | -2.340 | -2.362 | -2.354 | -2.354 | -2.352 | 8.36% |
| 30 | -1.982 | -1.983 | -1.975 | -1.975 | -1.975 | 9.36% |
| 40 | 2.026 | 2.172 | 2.171 | 2.156 | 2.152 | 89.68% |
| 50 | 2.949 | 3.158 | 3.162 | 3.154 | 3.146 | 94.10% |
| 60 | 3.333 | 3.441 | 3.415 | 3.421 | 3.416 | 93.80% |
| 70 | 3.220 | 3.471 | 3.433 | 3.431 | 3.434 | 92.50% |
| 80 | 3.174 | 3.346 | 3.329 | 3.325 | 3.328 | 87.94% |
| 90 | 1.204 | 1.395 | 1.400 | 1.374 | 1.378 | 81.12% |
