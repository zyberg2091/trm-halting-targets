# binary-em — nsup-16  (halt > 0, n_sup = 16)

Q-loss: binary exact-match: target 1 iff every real token correct (pad-masked). Role: supervision sweep s=16 (paper's N_sup).
Common settings: c=0.1, T=3, 100 epochs; metrics logged every 10 epochs (0–90).
Source: trm_qloss_binary_em.ipynb, cell 27. Every value below is parsed verbatim from the run's printed log — nothing is recomputed.

## Per-epoch metrics

| epoch | train CE | Q-loss | avg steps (train, batch-level) | halt-logit mean | std | max | %logit>0 | %logit>1.5 | %step1 (train) | %never (train) | val acc | val steps mean | %step1 (val) | %never (val) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.6314 | 0.0194 | 15.957356076759062 | -7.889 | 2.714 | 0.093 | 0.2% | 0.0% | 0.4% | 99.6% | 0.08% | 16.00 | 0.0% | 100.0% |
| 10 | 0.6857 | 0.2629 | 16.0 | -2.700 | 0.540 | -0.138 | 0.0% | 0.0% | 0.0% | 100.0% | 6.96% | 16.00 | 0.0% | 100.0% |
| 20 | 0.5552 | 0.3342 | 16.0 | -2.225 | 0.361 | -0.659 | 0.0% | 0.0% | 0.0% | 100.0% | 8.32% | 16.00 | 0.0% | 100.0% |
| 30 | 0.5107 | 0.4223 | 16.0 | -1.791 | 0.381 | -0.270 | 0.0% | 0.0% | 0.0% | 100.0% | 14.24% | 16.00 | 0.0% | 100.0% |
| 40 | 0.2863 | 0.6511 | 12.077469793887705 | 0.485 | 0.434 | 2.561 | 87.1% | 1.1% | 89.4% | 8.5% | 52.06% | 1.61 | 93.5% | 3.8% |
| 50 | 0.3082 | 0.5934 | 10.985785358919687 | 0.720 | 0.699 | 3.676 | 85.7% | 13.0% | 89.3% | 9.1% | 85.80% | 1.06 | 99.5% | 0.4% |
| 60 | 0.2086 | 0.3989 | 6.357498223169865 | 0.710 | 0.923 | 6.222 | 83.9% | 10.8% | 93.6% | 5.8% | 96.54% | 1.00 | 100.0% | 0.0% |
| 70 | 0.0147 | 0.0655 | 1.0127931769722816 | 4.808 | 1.170 | 8.674 | 100.0% | 99.6% | 100.0% | 0.0% | 98.72% | 1.00 | 100.0% | 0.0% |
| 80 | 0.0104 | 0.0500 | 1.0 | 5.048 | 0.963 | 8.194 | 100.0% | 100.0% | 100.0% | 0.0% | 98.24% | 1.00 | 100.0% | 0.0% |
| 90 | 0.0121 | 0.0612 | 1.0 | 4.800 | 1.043 | 9.294 | 100.0% | 99.9% | 100.0% | 0.0% | 99.06% | 1.00 | 100.0% | 0.0% |

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
| %step1 (train) | Fraction of training samples whose first h > 0 happened at supervision step 1. | The headline collapse indicator: halting at step 1 means recursion is skipped from the start. |
| %never (train) | Fraction of training samples recorded at step 16 — halted-at-the-last-step and never-halted combined, so an upper bound on true never-halting. | The opposite failure mode: no halting at all, and the gradient-sparsity question. For this run's binary-EM target, 100% here early is the expected all-zero-target phase: no sample is fully correct yet, so nothing should halt. |
| val acc | Sequence exact-match on the 5,000 validation samples, each scored at the output of its own halting step (never-halted samples use the last executed step). | The outcome metric — what every halting behavior is ultimately traded against. |
| val steps mean | Mean recorded halting step over the validation set. | Recursion compute at evaluation; read next to val acc to price the halting decisions. |
| %step1 (val), %never (val) | The same two indicators as the train columns, measured with evaluation-time halting on the validation set. | Confirms the training-side halting behavior transfers to eval, where halting decides which step's answer is actually used. |


## Training halt-step distribution (per-sample counts, whole train set)
Counts of samples by recorded halting step. Column `step 16` mixes two cases the log cannot separate: halted exactly at step 16, and never halted (both are recorded as 16). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | step 9 | step 10 | step 11 | step 12 | step 13 | step 14 | step 15 | step 16 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 189 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 44811 | 45000 | 0.08% |
| 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45000 | 45000 | 6.96% |
| 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45000 | 45000 | 8.32% |
| 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45000 | 45000 | 14.24% |
| 40 | 40222 | 825 | 101 | 19 | 5 | 3 | 2 | 1 | 1 | 1 | 2 | 1 | 1 | 1 | 1 | 3814 | 45000 | 52.06% |
| 50 | 40199 | 632 | 79 | 9 | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4077 | 45000 | 85.80% |
| 60 | 42130 | 235 | 30 | 7 | 3 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2593 | 45000 | 96.54% |
| 70 | 44997 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 45000 | 98.72% |
| 80 | 45000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45000 | 98.24% |
| 90 | 45000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45000 | 99.06% |

## Validation halt-step distribution (per-sample counts, whole validation set)
Counts of samples by recorded halting step. Column `step 16` mixes two cases the log cannot separate: halted exactly at step 16, and never halted (both are recorded as 16). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | step 9 | step 10 | step 11 | step 12 | step 13 | step 14 | step 15 | step 16 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 5000 | 0.08% |
| 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 5000 | 6.96% |
| 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 5000 | 8.32% |
| 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 5000 | 14.24% |
| 40 | 4677 | 106 | 17 | 2 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 191 | 5000 | 52.06% |
| 50 | 4973 | 6 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 19 | 5000 | 85.80% |
| 60 | 5000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 96.54% |
| 70 | 5000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 98.72% |
| 80 | 5000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 98.24% |
| 90 | 5000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 99.06% |

## Validation halt-logit mean by supervision step
(first validation batch only; blank = validation halted every sample before reaching that step, so it was not executed; val acc is the epoch-level validation accuracy)

| epoch | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 | s11 | s12 | s13 | s14 | s15 | s16 | val acc (epoch) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -7.358 | -7.266 | -7.292 | -7.301 | -7.289 | -7.290 | -7.290 | -7.290 | -7.289 | -7.290 | -7.290 | -7.289 | -7.290 | -7.290 | -7.290 | -7.290 | 0.08% |
| 10 | -2.305 | -2.396 | -2.373 | -2.369 | -2.369 | -2.367 | -2.365 | -2.371 | -2.372 | -2.371 | -2.371 | -2.370 | -2.367 | -2.366 | -2.368 | -2.369 | 6.96% |
| 20 | -2.407 | -2.441 | -2.430 | -2.419 | -2.419 | -2.420 | -2.421 | -2.420 | -2.419 | -2.420 | -2.420 | -2.420 | -2.420 | -2.420 | -2.420 | -2.420 | 8.32% |
| 30 | -1.635 | -1.595 | -1.585 | -1.589 | -1.590 | -1.589 | -1.588 | -1.590 | -1.589 | -1.590 | -1.588 | -1.589 | -1.590 | -1.588 | -1.590 | -1.588 | 14.24% |
| 40 | 0.548 | 0.597 | 0.608 | 0.608 | 0.605 | 0.603 | 0.603 | 0.604 | 0.604 | 0.604 | 0.603 | 0.602 | 0.602 | 0.604 | 0.605 | 0.604 | 52.06% |
| 50 | 1.860 | 1.922 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 85.80% |
| 60 | 3.803 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 96.54% |
| 70 | 5.494 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 98.72% |
| 80 | 5.182 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 98.24% |
| 90 | 5.217 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 99.06% |
