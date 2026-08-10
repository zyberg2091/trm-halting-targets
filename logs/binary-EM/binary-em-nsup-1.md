# binary-em — nsup-1  (halt > 0, n_sup = 1)

Q-loss: binary exact-match: target 1 iff every real token correct (pad-masked). Role: supervision sweep s=1 (no recursion).
Common settings: c=0.1, T=3, 100 epochs; metrics logged every 10 epochs (0–90).
Source: trm_qloss_binary_em.ipynb, cell 26. Every value below is parsed verbatim from the run's printed log — nothing is recomputed.

## Per-epoch metrics

| epoch | train CE | Q-loss | avg steps (train, batch-level) | halt-logit mean | std | max | %logit>0 | %logit>1.5 | %step1 (train) | %never (train) | val acc | val steps mean | %step1 (val) | %never (val) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.6313 | 0.0182 | 1.0 | -7.587 | 2.406 | 0.019 | 0.0% | 0.0% | 100.0% | 100.0% | 0.06% | 1.00 | 100.0% | 100.0% |
| 10 | 0.8161 | 0.1780 | 1.0 | -3.362 | 0.713 | -0.911 | 0.0% | 0.0% | 100.0% | 100.0% | 5.94% | 1.00 | 100.0% | 100.0% |
| 20 | 0.5408 | 0.3399 | 1.0 | -2.189 | 0.352 | -0.644 | 0.0% | 0.0% | 100.0% | 100.0% | 8.90% | 1.00 | 100.0% | 100.0% |
| 30 | 0.2601 | 0.6539 | 1.0 | 0.304 | 0.731 | 3.389 | 63.9% | 5.7% | 100.0% | 100.0% | 72.50% | 1.00 | 100.0% | 100.0% |
| 40 | 0.0558 | 0.2249 | 1.0 | 3.171 | 1.219 | 7.991 | 99.5% | 90.7% | 100.0% | 100.0% | 95.46% | 1.00 | 100.0% | 100.0% |
| 50 | 0.0244 | 0.1460 | 1.0 | 3.755 | 0.986 | 7.345 | 99.9% | 98.2% | 100.0% | 100.0% | 95.22% | 1.00 | 100.0% | 100.0% |
| 60 | 0.0207 | 0.1164 | 1.0 | 4.211 | 1.146 | 8.554 | 99.9% | 98.6% | 100.0% | 100.0% | 87.20% | 1.00 | 100.0% | 100.0% |
| 70 | 0.0261 | 0.1265 | 1.0 | 4.100 | 1.251 | 8.390 | 99.9% | 97.5% | 100.0% | 100.0% | 98.24% | 1.00 | 100.0% | 100.0% |
| 80 | 0.0144 | 0.0998 | 1.0 | 4.324 | 1.055 | 8.690 | 100.0% | 99.3% | 100.0% | 100.0% | 98.32% | 1.00 | 100.0% | 100.0% |
| 90 | 0.0196 | 0.0953 | 1.0 | 4.662 | 1.441 | 10.028 | 99.9% | 98.6% | 100.0% | 100.0% | 99.06% | 1.00 | 100.0% | 100.0% |

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
| %never (train) | Fraction of training samples recorded at step 1 — halted-at-the-last-step and never-halted combined, so an upper bound on true never-halting. | The opposite failure mode: no halting at all, and the gradient-sparsity question. For this run's binary-EM target, 100% here early is the expected all-zero-target phase: no sample is fully correct yet, so nothing should halt. |
| val acc | Sequence exact-match on the 5,000 validation samples, each scored at the output of its own halting step (never-halted samples use the last executed step). | The outcome metric — what every halting behavior is ultimately traded against. |
| val steps mean | Mean recorded halting step over the validation set. | Recursion compute at evaluation; read next to val acc to price the halting decisions. |
| %step1 (val), %never (val) | The same two indicators as the train columns, measured with evaluation-time halting on the validation set. | Confirms the training-side halting behavior transfers to eval, where halting decides which step's answer is actually used. |


## Training halt-step distribution (per-sample counts, whole train set)
Counts of samples by recorded halting step. Column `step 1` mixes two cases the log cannot separate: halted exactly at step 1, and never halted (both are recorded as 1). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | total | val acc (epoch) |
|---|---|---|---|
| 0 | 45000 | 45000 | 0.06% |
| 10 | 45000 | 45000 | 5.94% |
| 20 | 45000 | 45000 | 8.90% |
| 30 | 45000 | 45000 | 72.50% |
| 40 | 45000 | 45000 | 95.46% |
| 50 | 45000 | 45000 | 95.22% |
| 60 | 45000 | 45000 | 87.20% |
| 70 | 45000 | 45000 | 98.24% |
| 80 | 45000 | 45000 | 98.32% |
| 90 | 45000 | 45000 | 99.06% |

## Validation halt-step distribution (per-sample counts, whole validation set)
Counts of samples by recorded halting step. Column `step 1` mixes two cases the log cannot separate: halted exactly at step 1, and never halted (both are recorded as 1). `val acc (epoch)` is that epoch's validation exact-match accuracy — the only accuracy the logs record — repeated here for alignment; it is not a per-split or per-step accuracy.

| epoch | step 1 | total | val acc (epoch) |
|---|---|---|---|
| 0 | 5000 | 5000 | 0.06% |
| 10 | 5000 | 5000 | 5.94% |
| 20 | 5000 | 5000 | 8.90% |
| 30 | 5000 | 5000 | 72.50% |
| 40 | 5000 | 5000 | 95.46% |
| 50 | 5000 | 5000 | 95.22% |
| 60 | 5000 | 5000 | 87.20% |
| 70 | 5000 | 5000 | 98.24% |
| 80 | 5000 | 5000 | 98.32% |
| 90 | 5000 | 5000 | 99.06% |

## Validation halt-logit mean by supervision step
(first validation batch only; blank = validation halted every sample before reaching that step, so it was not executed; val acc is the epoch-level validation accuracy)

| epoch | s1 | val acc (epoch) |
|---|---|---|
| 0 | -7.235 | 0.06% |
| 10 | -2.586 | 5.94% |
| 20 | -2.054 | 8.90% |
| 30 | 0.917 | 72.50% |
| 40 | 3.976 | 95.46% |
| 50 | 3.367 | 95.22% |
| 60 | 2.056 | 87.20% |
| 70 | 4.531 | 98.24% |
| 80 | 4.450 | 98.32% |
| 90 | 5.401 | 99.06% |
