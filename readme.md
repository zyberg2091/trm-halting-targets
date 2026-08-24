# Halting targets in a Tiny Recursive Model

I built a simplified version of the [Tiny Recursive Model](https://arxiv.org/abs/2510.04871) to study one question: what should the halt head be trained to predict?

The paper uses binary exact match, so the target is 1 only when the complete answer is correct. I compared it with two graded targets that seemed like reasonable alternatives. Both gave the halt head a denser signal, but they also made the model stop after one supervision step while its answers were still mostly wrong.

This is a small TRM-style experiment, not a reproduction of the paper's results. My model has 740,413 parameters and uses a two-layer MLP over one vector per sample. The paper uses a Transformer over per-position representations.

## What I tested

The task was 4-digit addition. Each target had five runs: halt thresholds of `0`, `1.5`, and `3.0` with `n_sup = 5`, followed by supervision budgets of `1`, `5`, and `16` at threshold `0`. The threshold-0, `n_sup = 5` run is shared between the two comparisons, giving 15 runs in total.

Each run used 45,000 training examples, 5,000 validation examples, and 100 epochs. I compared:

* **Binary exact match:** 1 only if the full answer is correct.
* **Soft mean:** the fraction of answer positions predicted correctly.
* **Geometric mean:** the geometric mean of the probabilities assigned to the correct tokens.

The motivation for the graded targets was simple. Exact match is almost always zero early in training, while partial credit might give the halt head something useful to learn. The experiments showed a problem with this change: a partially correct answer can receive a high target even when the sequence is still wrong.

## Results

At threshold `0` with `n_sup = 5`, the two graded targets moved almost every validation sample to one supervision step much earlier than binary exact match:

| Halt target        | Near-total first-step halting | Validation exact match at that point |
| ------------------ | ----------------------------: | -----------------------------------: |
| Soft mean          |         epoch 20: 5,000/5,000 |                                9.18% |
| Geometric mean     |         epoch 40: 4,999/5,000 |                               14.14% |
| Binary exact match |         epoch 50: 5,000/5,000 |                               96.86% |

Soft mean therefore committed every sample to one step while the model solved about 9% of the validation set. Geometric mean did the same at about 14%. With binary exact match, this happened only after the model was already mostly correct.

Higher thresholds delayed halting, but threshold `3` did not produce a stable regime within 100 epochs. The fraction of samples halting after one step moved sharply between checkpoints as their halt logits crossed back and forth over the cutoff.

Changing the supervision budget also gave no clear accuracy improvement in these runs:

| Halt target        | `n_sup = 1` | `n_sup = 5` | `n_sup = 16` |
| ------------------ | ----------: | ----------: | -----------: |
| Soft mean          |      98.90% |      99.68% |       98.66% |
| Geometric mean     |      99.10% |      99.64% |       98.76% |
| Binary exact match |      99.06% |      99.46% |       99.06% |

These are the best validation accuracies observed in each run. The largest within-target difference was 1.02 percentage points. This does not show that recursion is useless: `n_sup = 1` still contains the inner recursive computation. It only shows that extra supervision steps did not provide a clear benefit on this task.

The main conclusion is narrower. The halt target strongly changed when the model stopped, but 4-digit addition was too easy to reveal the accuracy cost of stopping too early. A better test needs a task where additional computation actually improves the answer.

## Limits of this experiment

These results are preliminary. There is one run per configuration and no fixed random seed. Each target notebook generated its own dataset, split, and initialization, so the cross-target numbers are descriptive rather than a controlled paired comparison. Exact reruns will not reproduce the recorded values.

Training was also unstable, and this implementation omits several stabilizers used in the paper, including EMA, weight decay, stable-max loss, and its optimizer settings. The model architecture and validation procedure also differ from the paper, so the accuracy numbers should not be compared directly with its results.

The logs do not record correctness after every supervision step or the mean halt target during training. The intended per-sample mask on the halt loss was also ineffective because the loss had already been reduced to a batch mean. This applies to all 15 runs, but it should be corrected in a follow-up experiment.

## Code and logged results

The three notebooks contain the model, data generation, training loop, and five configurations for each target. Their outputs are retained, so the reported numbers can be inspected without rerunning the experiments.

* [`notebooks/trm_qloss_softmean.ipynb`](notebooks/trm_qloss_softmean.ipynb)
* [`notebooks/trm_qloss_geomean.ipynb`](notebooks/trm_qloss_geomean.ipynb)
* [`notebooks/trm_qloss_binary_em.ipynb`](notebooks/trm_qloss_binary_em.ipynb)
* [`logs/per-epoch.csv`](logs/per-epoch.csv)
* [`logs/step-distributions.csv`](logs/step-distributions.csv)
* [`logs/supstep-halt-logits.csv`](logs/supstep-halt-logits.csv)

No checkpoints are included. The next useful experiment would use identical datasets and initialization seeds across targets, multiple runs per configuration, and a task where extra supervision steps measurably improve accuracy.
