# geomean — nsup-16  (halt > 0, n_sup = 16)

Q-loss: geometric mean of correct-token probabilities = exp(-mean CE) (pad-masked). Role: supervision sweep s=16 (paper's N_sup).
Common settings: c=0.1, T=3, 100 epochs; metrics logged every 10 epochs (0–90).
Source: trm_qloss_geomean.ipynb, cell 28. Every value below is parsed verbatim from the run's printed log — nothing is recomputed.

## Per-epoch metrics

| epoch | train CE | Q-loss | avg steps (train, batch-level) | halt-logit mean | std | max | %logit>0 | %logit>1.5 | %step1 (train) | %never (train) | val acc | val steps mean | %step1 (val) | %never (val) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.6240 | 0.4683 | 16.0 | -1.540 | 0.386 | -0.031 | 0.0% | 0.0% | 0.0% | 100.0% | 0.04% | 16.00 | 0.0% | 100.0% |
| 10 | 0.8412 | 0.6760 | 16.0 | -0.367 | 0.199 | 0.554 | 2.7% | 0.0% | 3.9% | 94.2% | 5.62% | 11.22 | 24.9% | 67.3% |
| 20 | 0.6181 | 0.6929 | 15.990760483297796 | 0.048 | 0.124 | 0.566 | 65.3% | 0.0% | 65.5% | 28.8% | 6.66% | 4.11 | 73.4% | 20.3% |
| 30 | 0.6025 | 0.6921 | 15.762615493958778 | 0.085 | 0.124 | 0.601 | 76.1% | 0.0% | 75.8% | 20.9% | 7.90% | 5.20 | 68.5% | 27.7% |
| 40 | 0.6203 | 0.6914 | 14.673773987206824 | 0.103 | 0.109 | 0.552 | 82.5% | 0.0% | 83.3% | 14.2% | 9.94% | 2.30 | 89.9% | 8.5% |
| 50 | 0.1324 | 0.3570 | 1.0 | 2.078 | 0.448 | 4.347 | 100.0% | 90.5% | 100.0% | 0.0% | 83.90% | 1.00 | 100.0% | 0.0% |
| 60 | 0.0402 | 0.1290 | 1.0 | 3.818 | 0.914 | 7.138 | 100.0% | 99.3% | 100.0% | 0.0% | 96.74% | 1.00 | 100.0% | 0.0% |
| 70 | 0.1722 | 0.2579 | 2.565031982942431 | 1.466 | 1.521 | 6.627 | 89.8% | 34.4% | 98.3% | 1.5% | 98.04% | 1.00 | 100.0% | 0.0% |
| 80 | 0.0184 | 0.0777 | 1.0 | 4.514 | 0.924 | 8.291 | 100.0% | 99.9% | 100.0% | 0.0% | 97.78% | 1.00 | 100.0% | 0.0% |
| 90 | 0.0363 | 0.1369 | 1.0 | 3.723 | 1.124 | 7.331 | 100.0% | 98.2% | 100.0% | 0.0% | 98.76% | 1.00 | 100.0% | 0.0% |

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
| %step1 (train) | Fraction of training samples whose first h > 0 happened at supervision step 1. | The headline collapse indicator: halting at step 1 means recursion is skipped from the start. |
| %never (train) | Fraction of training samples recorded at step 16 — halted-at-the-last-step and never-halted combined, so an upper bound on true never-halting. | The opposite failure mode: no halting at all, and the gradient-sparsity question. |
| val acc | Sequence exact-match on the 5,000 validation samples, each scored at the output of its own halting step (never-halted samples use the last executed step). | The outcome metric — what every halting behavior is ultimately traded against. |
| val steps mean | Mean recorded halting step over the validation set. | Recursion compute at evaluation; read next to val acc to price the halting decisions. |
| %step1 (val), %never (val) | The same two indicators as the train columns, measured with evaluation-time halting on the validation set. | Confirms the training-side halting behavior transfers to eval, where halting decides which step's answer is actually used. |

Fine print for every definition (populations, pad handling, the step-16 conflation): METRICS.md.

## Training halt-step distribution (per-sample counts, whole train set)
Counts of samples by recorded halting step. Column `step 16` mixes two cases the log cannot separate: halted exactly at step 16, and never halted (both are recorded as 16). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | step 9 | step 10 | step 11 | step 12 | step 13 | step 14 | step 15 | step 16 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45000 | 45000 | 0.04% |
| 10 | 1758 | 607 | 170 | 58 | 8 | 9 | 4 | 3 | 5 | 1 | 0 | 1 | 1 | 1 | 2 | 42372 | 45000 | 5.62% |
| 20 | 29494 | 2225 | 260 | 50 | 13 | 4 | 7 | 3 | 1 | 1 | 0 | 1 | 2 | 0 | 1 | 12938 | 45000 | 6.66% |
| 30 | 34089 | 1286 | 160 | 26 | 10 | 3 | 6 | 3 | 1 | 2 | 1 | 1 | 0 | 1 | 0 | 9411 | 45000 | 7.90% |
| 40 | 37507 | 933 | 110 | 20 | 9 | 5 | 5 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 6407 | 45000 | 9.94% |
| 50 | 45000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45000 | 83.90% |
| 60 | 45000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45000 | 96.74% |
| 70 | 44223 | 108 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 669 | 45000 | 98.04% |
| 80 | 45000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45000 | 97.78% |
| 90 | 45000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45000 | 98.76% |

## Validation halt-step distribution (per-sample counts, whole validation set)
Counts of samples by recorded halting step. Column `step 16` mixes two cases the log cannot separate: halted exactly at step 16, and never halted (both are recorded as 16). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | step 9 | step 10 | step 11 | step 12 | step 13 | step 14 | step 15 | step 16 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 5000 | 0.04% |
| 10 | 1247 | 258 | 88 | 26 | 8 | 3 | 2 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 3365 | 5000 | 5.62% |
| 20 | 3671 | 274 | 32 | 7 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1013 | 5000 | 6.66% |
| 30 | 3424 | 171 | 14 | 2 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1386 | 5000 | 7.90% |
| 40 | 4495 | 74 | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 427 | 5000 | 9.94% |
| 50 | 5000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 83.90% |
| 60 | 5000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 96.74% |
| 70 | 5000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 98.04% |
| 80 | 5000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 97.78% |
| 90 | 5000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 98.76% |

## Validation halt-logit mean by supervision step
(first validation batch only; blank = validation halted every sample before reaching that step, so it was not executed; val acc is the epoch-level validation accuracy)

| epoch | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 | s11 | s12 | s13 | s14 | s15 | s16 | val acc (epoch) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -1.454 | -1.426 | -1.450 | -1.445 | -1.446 | -1.445 | -1.446 | -1.446 | -1.446 | -1.446 | -1.446 | -1.446 | -1.446 | -1.446 | -1.446 | -1.446 | 0.04% |
| 10 | -0.097 | -0.098 | -0.086 | -0.085 | -0.081 | -0.088 | -0.083 | -0.081 | -0.085 | -0.083 | -0.084 | -0.082 | -0.083 | -0.087 | -0.081 | -0.082 | 5.62% |
| 20 | 0.118 | 0.109 | 0.102 | 0.106 | 0.106 | 0.105 | 0.105 | 0.105 | 0.106 | 0.105 | 0.105 | 0.106 | 0.105 | 0.105 | 0.105 | 0.106 | 6.66% |
| 30 | 0.060 | 0.056 | 0.053 | 0.054 | 0.054 | 0.054 | 0.054 | 0.054 | 0.054 | 0.054 | 0.054 | 0.054 | 0.054 | 0.054 | 0.054 | 0.054 | 7.90% |
| 40 | 0.183 | 0.182 | 0.179 | 0.179 | 0.179 | 0.179 | 0.179 | 0.179 | 0.179 | 0.179 | 0.179 | 0.179 | 0.179 | 0.179 | 0.179 | 0.179 | 9.94% |
| 50 | 2.553 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 83.90% |
| 60 | 4.182 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 96.74% |
| 70 | 4.434 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 98.04% |
| 80 | 5.171 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 97.78% |
| 90 | 4.935 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 98.76% |
