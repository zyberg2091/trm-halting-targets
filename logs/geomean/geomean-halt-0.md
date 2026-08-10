# geomean — halt-0  (halt > 0, n_sup = 5)

Q-loss: geometric mean of correct-token probabilities = exp(-mean CE) (pad-masked). Role: threshold sweep t=0 (center cell).
Common settings: c=0.1, T=3, 100 epochs; metrics logged every 10 epochs (0–90).
Source: trm_qloss_geomean.ipynb, cell 24. Every value below is parsed verbatim from the run's printed log — nothing is recomputed.

## Per-epoch metrics

| epoch | train CE | Q-loss | avg steps (train, batch-level) | halt-logit mean | std | max | %logit>0 | %logit>1.5 | %step1 (train) | %never (train) | val acc | val steps mean | %step1 (val) | %never (val) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.6453 | 0.4628 | 5.0 | -1.558 | 0.379 | -0.019 | 0.0% | 0.0% | 0.0% | 100.0% | 0.10% | 5.00 | 0.0% | 100.0% |
| 10 | 0.6846 | 0.6929 | 5.0 | -0.067 | 0.183 | 0.656 | 37.3% | 0.0% | 38.9% | 53.9% | 6.84% | 3.16 | 41.0% | 51.8% |
| 20 | 0.6096 | 0.6924 | 4.9893390191897655 | 0.071 | 0.127 | 0.522 | 70.5% | 0.0% | 70.1% | 25.4% | 7.82% | 1.99 | 72.4% | 23.7% |
| 30 | 0.5815 | 0.6899 | 4.930348258706467 | 0.135 | 0.134 | 0.598 | 82.3% | 0.0% | 81.5% | 15.5% | 7.68% | 1.24 | 92.2% | 5.4% |
| 40 | 0.5686 | 0.6837 | 2.4378109452736316 | 0.189 | 0.136 | 0.753 | 92.9% | 0.0% | 96.4% | 3.0% | 14.14% | 1.00 | 100.0% | 0.0% |
| 50 | 0.0475 | 0.1567 | 1.0 | 3.448 | 0.693 | 6.053 | 100.0% | 99.4% | 100.0% | 0.0% | 94.48% | 1.00 | 100.0% | 0.0% |
| 60 | 0.0217 | 0.0877 | 1.0 | 4.361 | 0.920 | 7.590 | 100.0% | 99.9% | 100.0% | 0.0% | 95.00% | 1.00 | 100.0% | 0.0% |
| 70 | 0.0172 | 0.0675 | 1.0 | 4.814 | 1.097 | 9.050 | 100.0% | 99.9% | 100.0% | 0.0% | 98.54% | 1.00 | 100.0% | 0.0% |
| 80 | 0.1376 | 0.1870 | 1.3795309168443497 | 3.068 | 2.114 | 9.144 | 96.0% | 67.8% | 98.7% | 1.0% | 95.90% | 1.00 | 100.0% | 0.0% |
| 90 | 0.0108 | 0.0543 | 1.0 | 4.959 | 1.077 | 9.096 | 100.0% | 99.9% | 100.0% | 0.0% | 99.64% | 1.00 | 100.0% | 0.0% |

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
| %never (train) | Fraction of training samples recorded at step 5 — halted-at-the-last-step and never-halted combined, so an upper bound on true never-halting. | The opposite failure mode: no halting at all, and the gradient-sparsity question. |
| val acc | Sequence exact-match on the 5,000 validation samples, each scored at the output of its own halting step (never-halted samples use the last executed step). | The outcome metric — what every halting behavior is ultimately traded against. |
| val steps mean | Mean recorded halting step over the validation set. | Recursion compute at evaluation; read next to val acc to price the halting decisions. |
| %step1 (val), %never (val) | The same two indicators as the train columns, measured with evaluation-time halting on the validation set. | Confirms the training-side halting behavior transfers to eval, where halting decides which step's answer is actually used. |


## Training halt-step distribution (per-sample counts, whole train set)
Counts of samples by recorded halting step. Column `step 5` mixes two cases the log cannot separate: halted exactly at step 5, and never halted (both are recorded as 5). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 45000 | 45000 | 0.10% |
| 10 | 17485 | 2614 | 495 | 133 | 24273 | 45000 | 6.84% |
| 20 | 31565 | 1737 | 208 | 54 | 11436 | 45000 | 7.82% |
| 30 | 36663 | 1171 | 162 | 26 | 6978 | 45000 | 7.68% |
| 40 | 43372 | 238 | 27 | 9 | 1354 | 45000 | 14.14% |
| 50 | 45000 | 0 | 0 | 0 | 0 | 45000 | 94.48% |
| 60 | 45000 | 0 | 0 | 0 | 0 | 45000 | 95.00% |
| 70 | 45000 | 0 | 0 | 0 | 0 | 45000 | 98.54% |
| 80 | 44435 | 92 | 11 | 2 | 460 | 45000 | 95.90% |
| 90 | 45000 | 0 | 0 | 0 | 0 | 45000 | 99.64% |

## Validation halt-step distribution (per-sample counts, whole validation set)
Counts of samples by recorded halting step. Column `step 5` mixes two cases the log cannot separate: halted exactly at step 5, and never halted (both are recorded as 5). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | step 2 | step 3 | step 4 | step 5 | total | val acc (epoch) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 5000 | 5000 | 0.10% |
| 10 | 2048 | 314 | 36 | 13 | 2589 | 5000 | 6.84% |
| 20 | 3622 | 169 | 21 | 1 | 1187 | 5000 | 7.82% |
| 30 | 4609 | 106 | 12 | 1 | 272 | 5000 | 7.68% |
| 40 | 4999 | 1 | 0 | 0 | 0 | 5000 | 14.14% |
| 50 | 5000 | 0 | 0 | 0 | 0 | 5000 | 94.48% |
| 60 | 5000 | 0 | 0 | 0 | 0 | 5000 | 95.00% |
| 70 | 5000 | 0 | 0 | 0 | 0 | 5000 | 98.54% |
| 80 | 5000 | 0 | 0 | 0 | 0 | 5000 | 95.90% |
| 90 | 5000 | 0 | 0 | 0 | 0 | 5000 | 99.64% |

## Validation halt-logit mean by supervision step
(first validation batch only; blank = validation halted every sample before reaching that step, so it was not executed; val acc is the epoch-level validation accuracy)

| epoch | s1 | s2 | s3 | s4 | s5 | val acc (epoch) |
|---|---|---|---|---|---|---|
| 0 | -1.451 | -1.479 | -1.445 | -1.462 | -1.456 | 0.10% |
| 10 | -0.056 | -0.076 | -0.065 | -0.063 | -0.065 | 6.84% |
| 20 | 0.102 | 0.100 | 0.096 | 0.096 | 0.095 | 7.82% |
| 30 | 0.227 | 0.219 | 0.223 | 0.219 | 0.223 | 7.68% |
| 40 | 0.429 |  |  |  |  | 14.14% |
| 50 | 3.967 |  |  |  |  | 94.48% |
| 60 | 3.714 |  |  |  |  | 95.00% |
| 70 | 5.969 |  |  |  |  | 98.54% |
| 80 | 3.749 |  |  |  |  | 95.90% |
| 90 | 6.248 |  |  |  |  | 99.64% |
