# softmean — nsup-16  (halt > 0, n_sup = 16)

Q-loss: soft-mean of token correctness (pad-masked). Role: supervision sweep s=16 (paper's N_sup).
Common settings: c=0.1, T=3, 100 epochs; metrics logged every 10 epochs (0–90).
Source: trm_qloss_softmean.ipynb, cell 28. Every value below is parsed verbatim from the run's printed log — nothing is recomputed.

## Per-epoch metrics

| epoch | train CE | Q-loss | avg steps (train, batch-level) | halt-logit mean | std | max | %logit>0 | %logit>1.5 | %step1 (train) | %never (train) | val acc | val steps mean | %step1 (val) | %never (val) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.6392 | 0.6163 | 16.0 | -0.779 | 0.436 | 0.233 | 0.5% | 0.0% | 1.3% | 98.6% | 0.06% | 16.00 | 0.0% | 100.0% |
| 10 | 1.0739 | 0.6898 | 12.73134328358209 | 0.153 | 0.153 | 0.875 | 83.7% | 0.0% | 87.3% | 10.3% | 2.00% | 1.50 | 95.5% | 3.2% |
| 20 | 0.5612 | 0.5443 | 1.0 | 1.190 | 0.203 | 2.077 | 100.0% | 6.7% | 100.0% | 0.0% | 8.46% | 1.00 | 100.0% | 0.0% |
| 30 | 0.5203 | 0.5216 | 1.0 | 1.292 | 0.186 | 2.006 | 100.0% | 14.3% | 100.0% | 0.0% | 9.72% | 1.00 | 100.0% | 0.0% |
| 40 | 0.0663 | 0.1007 | 1.0 | 4.097 | 0.786 | 7.246 | 100.0% | 99.9% | 100.0% | 0.0% | 89.48% | 1.00 | 100.0% | 0.0% |
| 50 | 0.0459 | 0.0641 | 1.0 | 4.862 | 1.056 | 8.540 | 100.0% | 99.8% | 100.0% | 0.0% | 96.94% | 1.00 | 100.0% | 0.0% |
| 60 | 0.0196 | 0.0388 | 1.0 | 5.384 | 0.914 | 8.632 | 100.0% | 100.0% | 100.0% | 0.0% | 98.66% | 1.00 | 100.0% | 0.0% |
| 70 | 0.0200 | 0.0395 | 1.0 | 5.426 | 1.017 | 8.865 | 100.0% | 100.0% | 100.0% | 0.0% | 98.58% | 1.00 | 100.0% | 0.0% |
| 80 | 0.0217 | 0.0369 | 1.0 | 5.720 | 1.186 | 10.384 | 100.0% | 100.0% | 100.0% | 0.0% | 94.26% | 1.00 | 100.0% | 0.0% |
| 90 | 0.0132 | 0.0241 | 1.0 | 6.272 | 1.210 | 9.874 | 100.0% | 100.0% | 100.0% | 0.0% | 94.72% | 1.00 | 100.0% | 0.0% |

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
| %never (train) | Fraction of training samples recorded at step 16 — halted-at-the-last-step and never-halted combined, so an upper bound on true never-halting. | The opposite failure mode: no halting at all, and the gradient-sparsity question. |
| val acc | Sequence exact-match on the 5,000 validation samples, each scored at the output of its own halting step (never-halted samples use the last executed step). | The outcome metric — what every halting behavior is ultimately traded against. |
| val steps mean | Mean recorded halting step over the validation set. | Recursion compute at evaluation; read next to val acc to price the halting decisions. |
| %step1 (val), %never (val) | The same two indicators as the train columns, measured with evaluation-time halting on the validation set. | Confirms the training-side halting behavior transfers to eval, where halting decides which step's answer is actually used. |


## Training halt-step distribution (per-sample counts, whole train set)
Counts of samples by recorded halting step. Column `step 16` mixes two cases the log cannot separate: halted exactly at step 16, and never halted (both are recorded as 16). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | step 9 | step 10 | step 11 | step 12 | step 13 | step 14 | step 15 | step 16 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 597 | 22 | 6 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 44374 | 45000 | 0.06% |
| 10 | 39277 | 956 | 114 | 10 | 1 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4639 | 45000 | 2.00% |
| 20 | 45000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45000 | 8.46% |
| 30 | 45000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45000 | 9.72% |
| 40 | 45000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45000 | 89.48% |
| 50 | 45000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45000 | 96.94% |
| 60 | 45000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45000 | 98.66% |
| 70 | 45000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45000 | 98.58% |
| 80 | 45000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45000 | 94.26% |
| 90 | 45000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 45000 | 94.72% |

## Validation halt-step distribution (per-sample counts, whole validation set)
Counts of samples by recorded halting step. Column `step 16` mixes two cases the log cannot separate: halted exactly at step 16, and never halted (both are recorded as 16). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | step 9 | step 10 | step 11 | step 12 | step 13 | step 14 | step 15 | step 16 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 5000 | 0.06% |
| 10 | 4773 | 50 | 11 | 2 | 1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 159 | 5000 | 2.00% |
| 20 | 5000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 8.46% |
| 30 | 5000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 9.72% |
| 40 | 5000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 89.48% |
| 50 | 5000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 96.94% |
| 60 | 5000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 98.66% |
| 70 | 5000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 98.58% |
| 80 | 5000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 94.26% |
| 90 | 5000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5000 | 94.72% |

## Validation halt-logit mean by supervision step
(first validation batch only; blank = validation halted every sample before reaching that step, so it was not executed; val acc is the epoch-level validation accuracy)

| epoch | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 | s11 | s12 | s13 | s14 | s15 | s16 | val acc (epoch) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | -0.666 | -0.695 | -0.703 | -0.704 | -0.706 | -0.705 | -0.706 | -0.705 | -0.705 | -0.705 | -0.705 | -0.705 | -0.705 | -0.705 | -0.705 | -0.705 | 0.06% |
| 10 | 0.247 | 0.247 | 0.257 | 0.249 | 0.254 | 0.251 | 0.252 | 0.252 | 0.251 | 0.252 | 0.251 | 0.252 | 0.250 | 0.253 | 0.250 | 0.255 | 2.00% |
| 20 | 1.143 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 8.46% |
| 30 | 1.321 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 9.72% |
| 40 | 4.415 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 89.48% |
| 50 | 5.094 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 96.94% |
| 60 | 5.847 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 98.66% |
| 70 | 6.711 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 98.58% |
| 80 | 4.523 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 94.26% |
| 90 | 4.507 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 94.72% |
