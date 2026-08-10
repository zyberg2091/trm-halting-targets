# Halting-target choice in the Tiny Recursive Model

Reimplementation of the Tiny Recursive Model ([arXiv:2510.04871](https://arxiv.org/abs/2510.04871)) from scratch, comparing three halting objectives across 15 runs on 4-digit addition.

## 1. Recursive models with deep supervision

The Tiny Recursive Model solves a task by repeatedly rewriting its own answer rather than producing it in a single pass.

It carries a current answer `y` and a latent state `z`. One **supervision step** runs the inner recursion: the network updates `z` n times from the input, the current answer and the current latent, then updates `y` once from `y` and `z` alone, without the input. A single shared network does both. This implementation uses n = 6 and T = 3 cycles, following the paper, and within a supervision step the first T-1 cycles run without gradient while only the final cycle is differentiated.

That recursion structure follows the paper. The network inside it does not: it is a 2-layer MLP rather than the paper's 2-layer Transformer, and the representation is one vector per sample rather than one per position.

Deep supervision repeats that. The revised answer becomes the starting point for another supervision step, up to `n_sup` times, and every step is trained against the ground truth. The loss accumulates across supervision steps. The parameter update happens once per batch, after the supervision steps have finished and after every sample in the batch has had its halting decision made.

## 2. The halting mechanism

At each supervision step, alongside the revised answer, a small **halt head** outputs one scalar, the **halt logit** `h`. If `h` exceeds a fixed **halt threshold**, that sample stops. It takes no further supervision steps and its current answer is final.

The halt head is trained with binary cross-entropy against a target `t` answering "is this answer good enough to stop?" So the design question is: what should `t` measure?

In the TRM paper, `t` is binary exact-match, 1 if every token of the answer is correct and 0 otherwise.

## 3. The modification under test

This study replaces the paper's halting target with two alternatives and changes nothing else in the implementation. Model architecture, data-generation process, optimizer, schedule and the recursion structure of section 1 are the same for all three targets. Only the definition of `t` differs between corresponding configurations.

An answer has `L` non-pad tokens. `y_i` is the ground-truth token at position i, `ŷ_i` the model's argmax there, and `p_i` the probability the model assigns to `y_i`. Pad positions are excluded from all three targets.

**Binary exact-match**, the paper's target: `t = 1` if `ŷ_i = y_i` at every one of the `L` positions, else `t = 0`.

**Soft-mean**, substitution 1: `t` = the fraction of positions where `ŷ_i = y_i`. Three correct digits out of four gives `t = 0.75`.

**Geometric-mean**, substitution 2: `t` = the geometric mean of the correct-token probabilities, `(∏ p_i)^(1/L)`, computed in log space as `exp(-mean cross-entropy)`. Confidence-aware rather than argmax-based. One badly-predicted token drags the whole value down.

Why substitute at all? Binary exact-match is 0 for almost every sample early in training, when no answer is fully correct. That looks like a sparse target giving the halt head nothing to learn from, and a graded target looks like the obvious repair: soft-mean as the direct partial-credit version, geometric-mean as the confidence-weighted one. Both are changes a reimplementer might make without a second thought.

## 4. Experimental setup

### Task

4-digit addition. Each example is a pair of uniformly sampled integers in [1000, 9999]. The input is the two operands with a `+` token between them, giving a fixed length of 9 tokens. The target is the sum, 4 or 5 digits, right-padded to length 5. Vocabulary is 12 tokens: digits 0 to 9, a `+` token (id 10), and a pad token (id 11). Pad positions are excluded from the loss and from all three halting targets.

Within each target notebook, 50,000 examples were generated once and split into 45,000 training and 5,000 validation examples. The five configurations within that notebook shared the same dataset and split. The three target notebooks used independently generated datasets and splits.

### Model

| component | definition | parameters |
|---|---|---|
| embedding | `nn.Embedding(12, 256)` | 3,072 |
| input projection | `nn.Linear(9 * 256, 256)` | 590,080 |
| recursion network | `Linear(256, 256)`, ReLU, `Linear(256, 256)` | 131,584 |
| output head | `nn.Linear(256, 5 * 12)` | 15,420 |
| halt head | `nn.Linear(256, 1)` | 257 |
| **total** | | **740,413** |

*Columns: `component` is the module name in `TinyRModel`; `definition` is its constructor call with the values used here; `parameters` is the trainable parameter count including biases, computed by instantiating the model. Hidden size is 256 throughout.*

The embedded input is flattened across the 9 positions and projected to a single 256-dimensional vector, so `x`, `y` and `z` are each one vector per sample. Both `y` and `z` are initialised to zeros at the start of every supervision sequence. The output head expands `y` to the full 5-token answer in one projection, and the halt head reads the same `y`.

### Training

| setting | value |
|---|---|
| optimizer | Adam, learning rate 1e-4, default betas (0.9, 0.999) |
| weight decay | none |
| learning-rate schedule | none, no warmup |
| batch size | 32, giving 1,407 training batches per epoch |
| epochs | 100 per run |
| output loss | cross-entropy, `ignore_index` set to the pad token |
| halting loss | binary cross-entropy with logits, weight `c = 0.1` |
| EMA of weights | not used |
| device | single GPU |

*Columns: `setting` is the training-configuration item; `value` is what every run uses. All values are read from the notebooks and are identical across all 15 runs.*

Hyperparameters were not tuned. Because every value above is held constant across all 15 runs, none of them can account for a difference between the three halting targets.

The halting loss weight is the one departure that could matter, since it scales the loss whose target is being manipulated. The paper adds the halting term with no coefficient, that is at weight 1.0. Because the gradient on the halt logit is `c · (σ(h) - t)`, the weight `c` scales the rate at which the halt logit approaches `ln(t / (1 - t))` but not the value it approaches, so it changes when halting engages and not whether. A smaller `c` therefore delays halting relative to the paper's setting, which makes the premature-halting result in section 5 conservative rather than exaggerated.

Four stability mechanisms used by the paper are absent here: an exponential moving average of the weights at 0.999, weight decay, the paper's lower beta2 of 0.95, and stable-max loss. Section 10.1 quantifies the resulting noise.

### Halting

The halt head is learned during training and used at validation, where it decides which step's output is scored. The paper instead runs the full supervision budget at test time, so accuracy here is not directly comparable to the paper's.

### Configurations

Each run is one (halt threshold, `n_sup`) pair. Five per target:

| run | halt threshold | n_sup | analysis |
|---|---|---|---|
| 1 | 0 | 5 | shared reference |
| 2 | 1.5 | 5 | threshold |
| 3 | 3.0 | 5 | threshold |
| 4 | 0 | 1 | supervision steps |
| 5 | 0 | 16 | supervision steps |

*Columns: `halt threshold` is the cutoff the halt logit must exceed for a sample to stop; `n_sup` is the maximum supervision steps available; `analysis` names which comparison the run belongs to.*

The threshold analysis holds `n_sup` at 5 and varies the cutoff. The supervision-step analysis holds the cutoff at 0 and varies the budget. Run 1 is shared by both. `n_sup = 1` is the no-halting control, because with one step every sample is a step-1 sample and there is no halting decision.

Five configurations times three targets gives 15 runs, one run each.

No random seed is set. Model initialisation therefore differs across all 15 runs, and because the dataset is generated inside each notebook, the three targets were trained on independently drawn datasets and splits of the same size and distribution: 50,000 uniform 4-digit pairs, split 45,000 and 5,000. Within a notebook the five configurations share one dataset and split. The task distribution is homogeneous and the sample is large, so different draws should be near-equivalent, but the comparison across targets is not on identical data.

### Logging

Logged every 10 epochs (0 to 90): training cross-entropy and halting loss, halt-logit statistics, the per-sample halting-step distribution for training and validation, validation accuracy, and mean validation halt logit per supervision step. Field definitions are in `logs/METRICS.md`.

Accuracy throughout means validation sequence exact-match, every non-pad token correct, out of 5,000 samples.

One run per configuration is the study's main limitation, and it is bounded in section 10.1 using the `n_sup = 1` runs.

## 5. What the halting target does

Each target was run separately, and each produces a self-contained result: under both substituted targets first-step halting saturates while that model is still wrong on the large majority of its validation problems, while under the paper's exact-match target it saturates only after that model is largely correct. The three runs use independently generated datasets, so what follows is a qualitative difference in kind rather than a numerical comparison between them.

Samples using only one supervision step, at halt threshold 0 with `n_sup = 5`:

| epoch | soft-mean | geometric-mean | binary exact-match |
|---|---|---|---|
| 0 | 0.0% (0) | 0.0% (0) | 0.0% (0) |
| 10 | 98.2% (4908) | 41.0% (2048) | 0.0% (0) |
| 20 | 100.0% (5000) | 72.4% (3622) | 0.0% (0) |
| 30 | 100.0% (5000) | 92.2% (4609) | 0.1% (3) |
| 40 | 100.0% (5000) | 100.0% (4999) | 81.0% (4048) |
| 50 | 100.0% (5000) | 100.0% (5000) | 100.0% (5000) |
| 60 to 90 | 100.0% (5000) | 100.0% (5000) | 100.0% (5000) |

*Percentage of the 5,000 validation samples that used only one supervision step, raw count in brackets. Both values come straight from the logs. A sample counts here if its halt logit exceeded the threshold at step 1, so it stopped after a single pass and did no recursion. All three targets sit at 100% from epoch 50 through 90, so those five epochs are collapsed into one row. The three columns come from separate notebook runs on independently generated datasets and initialisations; see section 4.*

Validation accuracy over the same three runs:

| epoch | soft-mean | geometric-mean | binary exact-match |
|---|---|---|---|
| 0 | 0.08% | 0.10% | 0.08% |
| 10 | 2.52% | 6.84% | 6.82% |
| 20 | 9.18% | 7.82% | 8.22% |
| 30 | 44.58% | 7.68% | 23.06% |
| 40 | 95.44% | 14.14% | 66.80% |
| 50 | 97.34% | 94.48% | 96.86% |
| 60 | 99.66% | 95.00% | 99.12% |
| 70 | 99.68% | 98.54% | 98.88% |
| 80 | 96.86% | 95.90% | 98.38% |
| 90 | 97.26% | 99.64% | 99.46% |

*Validation sequence exact-match out of 5,000 samples, same epochs and same runs as above. All ten logged epochs. The three columns come from separate notebook runs on independently generated datasets and initialisations; see section 4.*

Soft-mean has committed every sample to a single step by epoch 20, when the model solves 9 problems in 100. Geometric-mean does the same by epoch 40, at 14 in 100. Exact-match reaches that point at epoch 50, at 97 in 100.

The gap does not depend on where "saturated" is drawn:

| level of first-step halting | soft-mean | geometric-mean | binary exact-match |
|---|---|---|---|
| epoch it first reaches 50% | 10 | 20 | 40 |
| accuracy at that epoch | 2.52% | 7.82% | 66.80% |
| epoch it first reaches 90% | 10 | 30 | 50 |
| accuracy at that epoch | 2.52% | 7.68% | 96.86% |
| epoch it first reaches 100% | 20 | 40 | 50 |
| accuracy at that epoch | 9.18% | 14.14% | 96.86% |

*Rows alternate between two quantities: an epoch number, and validation exact-match at that epoch. Both come from the two tables above. Each column is read from its own notebook's runs. The same qualitative split appears at every level: both substituted targets saturate below 15% accuracy, exact-match at 66.80% or above.*

Soft-mean has committed every sample to a single/first step by epoch 20, when the model solves 9 problems in 100. Geometric-mean does the same by epoch 40, at 14 in 100. Exact-match reaches that point at epoch 50, at 97 in 100. Those three readings use 100% as the definition of saturated. The table below repeats the same reading at 50% and at 90%.

This comparison is at halt threshold 0. Section 6 covers what happens at higher thresholds.

## 6. Raising the halt threshold

The halt threshold is the cutoff the halt logit must exceed. Raising it should make halting harder to trigger. It does, though not equally across targets.

| target | halt threshold | epoch halting reaches 90% | accuracy at that epoch | best accuracy in run |
|---|---|---|---|---|
| soft-mean | 0 | 10 | 2.52% | 99.68% |
| soft-mean | 1.5 | 40 | 53.28% | 98.72% |
| soft-mean | 3.0 | 70 | 90.04% | 90.04% |
| geometric-mean | 0 | 30 | 7.68% | 99.64% |
| geometric-mean | 1.5 | 60 | 96.24% | 98.64% |
| geometric-mean | 3.0 | never reached | | 95.54% |
| binary exact-match | 0 | 50 | 96.86% | 99.46% |
| binary exact-match | 1.5 | 50 | 86.14% | 92.32% |
| binary exact-match | 3.0 | never reached | | 94.10% |

*`target` and `halt threshold` identify the run; `n_sup = 5` throughout, one run each. `epoch halting reaches 90%` is the first logged epoch at which at least 90% of the 5,000 validation samples use only one supervision step; "never reached" means that level was not hit within 100 epochs. `accuracy at that epoch` is validation exact-match at that same epoch, blank where the level was never reached. `best accuracy in run` is the highest validation exact-match at any of the 10 logged epochs. All read from the logs. For soft-mean at threshold 3.0 the two accuracy columns hold the same number because epoch 70 is also that run's best epoch. Rows within one target are on identical data. Rows in different targets are not.*

Reading each target's three rows in turn. Soft-mean reaches 90% first-step halting at epoch 10, then 40, then 70 as the threshold goes 0, 1.5, 3.0, with accuracy at those epochs of 2.52%, 53.28% and 90.04%. Geometric-mean reaches it at epoch 30 then epoch 60 for thresholds 0 and 1.5, at 7.68% and 96.24% accuracy, and at threshold 3.0 never reaches it within 100 epochs.

Binary exact-match reaches the 90% level at epoch 50 for both threshold 0 and threshold 1.5, so the higher cutoff changed the timing not at all. Accuracy at that epoch is 96.86% and 86.14%, lower at the higher cutoff rather than higher.

A higher threshold does not settle halting. It makes it oscillate:

| epoch | soft-mean | geometric-mean | binary exact-match |
|---|---|---|---|
| 0 | 0.0% (0) | 0.0% (0) | 0.0% (0) |
| 10 | 0.0% (0) | 0.0% (0) | 0.0% (0) |
| 20 | 0.0% (0) | 0.0% (0) | 0.0% (0) |
| 30 | 0.0% (0) | 0.0% (0) | 0.0% (0) |
| 40 | 65.2% (3259) | 42.0% (2099) | 5.1% (253) |
| 50 | 51.4% (2572) | 46.3% (2317) | 43.9% (2197) |
| 60 | 53.4% (2668) | 62.7% (3135) | 71.6% (3578) |
| 70 | 90.3% (4514) | 29.6% (1480) | 70.3% (3515) |
| 80 | 77.6% (3878) | 76.3% (3816) | 57.4% (2871) |
| 90 | 88.3% (4413) | 75.0% (3748) | 1.1% (56) |

*Same quantity as the first table in section 5: percentage of the 5,000 validation samples that used only one supervision step, raw count in brackets. Halt threshold 3.0, `n_sup = 5`, one run. All ten logged epochs, nothing omitted. At halt threshold 0 the same quantity is pinned at 100% from epoch 50 through 90 in all three targets. The three columns come from separate notebook runs on independently generated datasets and initialisations; see section 4.*

Through epoch 30 not a single sample halts early in any target. From epoch 40 the number moves up and down without settling. Binary exact-match ends at 1.1%, which is 56 samples out of 5,000, so almost everything is running the full budget again by epoch 90. Its logits fell from an average of 3.174 to 1.204 (the values are from logs/supstep-halt-logits.csv), dropping almost the whole set below the cutoff, alongside validation accuracy falling from 87.94% to 81.12%.

Each validation sample has its own halt logit, so the share in the table is just how many of the 5,000 sit above the cutoff. Training moves those logits up and down, and at cutoff 3.0 they sit right around it, so ordinary movement pushes large numbers of samples across it; at cutoff 0 they sit far above, so nothing crosses and the share stays at 100%.

Halting at threshold 3 is not a stable regime in any target within 100 epochs.

Best accuracy in the run is highest at threshold 0 for all three: 99.68%, 99.64% and 99.46%. At the other two thresholds it ranges from 90.04% to 98.72%. Best-checkpoint is used here rather than last-epoch accuracy, and the two differ for soft-mean at threshold 0: 99.68% at epoch 70 against 97.26% at epoch 90, from the section 5 accuracy table. Over a 10-point curve the maximum is the more stable of the two.


## 7. Changing the recursion budget

The supervision budget `n_sup` is the maximum number of steps a sample may take. These runs hold the halt threshold at 0 and vary the budget: 1, 5, 16.

`n_sup = 1` is the no-halting control. With a single supervision step there is no halting decision to make, because every sample answers in one pass by construction. Any behaviour these runs show cannot be caused by halting.

Best accuracy barely moves with the budget:

| target | n_sup = 1 | n_sup = 5 | n_sup = 16 | spread |
|---|---|---|---|---|
| soft-mean | 98.90% | 99.68% | 98.66% | 1.02 |
| geometric-mean | 99.10% | 99.64% | 98.76% | 0.88 |
| binary exact-match | 99.06% | 99.46% | 99.06% | 0.40 |

*Highest validation exact-match reached at any of the ten logged epochs of that run, out of 5,000 validation samples. Halt threshold is 0 in all nine runs and only the supervision budget differs. `spread` is the largest value minus the smallest for that target, in percentage points. One run per configuration. The three columns come from separate notebook runs on independently generated datasets and initialisations; see section 4.*

Sixteen supervision steps buy nothing over one on this task. The largest difference anywhere is 1.02 points, and in every target the middle budget happens to be highest, so the ordering is noise rather than a trend.

A larger budget is never faster:

| target | n_sup = 1 | n_sup = 5 | n_sup = 16 |
|---|---|---|---|
| soft-mean | 30 | 40 | 40 |
| geometric-mean | 30 | 50 | 50 |
| binary exact-match | 30 | 40 | 50 |

*First logged epoch at which validation accuracy reaches 50%, same nine runs. Lower is faster. The three columns come from separate notebook runs on independently generated datasets and initialisations; see section 4.*

The single-step runs reach 50% accuracy at epoch 30 in all three targets. Every larger budget takes 10 to 20 epochs longer. The epoch axis understates the gap in compute: before halting engages, an `n_sup = 16` run executes sixteen supervision steps per batch where the single-step run executes one.

At `n_sup = 16` with halt threshold 0, the share of validation samples using only one supervision step reaches 100% by epoch 50 in all three targets, the same as at `n_sup = 5`. Eleven extra steps are available and go unused for the rest of training.

The single-step runs have no halting decision at all, yet they show the same steep accuracy rise in the same epoch 20 to 30 interval in all three targets. Whatever the halting target does, it is not what produces the rise.

Those runs also bound the study's noise:

| epoch | soft-mean | geometric-mean | binary exact-match |
|---|---|---|---|
| 0 | 0.10% | 0.04% | 0.06% |
| 10 | 7.92% | 6.86% | 5.94% |
| 20 | 8.60% | 8.18% | 8.90% |
| 30 | 86.60% | 68.76% | 72.50% |
| 40 | 94.26% | 97.74% | 95.46% |
| 50 | 94.06% | 96.70% | 95.22% |
| 60 | 97.56% | 95.82% | 87.20% |
| 70 | 98.74% | 98.76% | 98.24% |
| 80 | 76.86% | 98.84% | 98.32% |
| 90 | 98.90% | 99.10% | 99.06% |

*Validation sequence exact-match out of 5,000 samples, in runs with a single supervision step (n_sup=1) and therefore no halting decision. All ten logged epochs. The three columns come from separate notebook runs on independently generated datasets and initialisations; see section 4.*

Over epochs 50 to 90 these runs span 22.04 points (soft-mean), 3.28 (geometric-mean) and 11.86 (binary exact-match). Soft-mean drops from 98.74% to 76.86% and back to 98.90% in twenty epochs with no halting involved. Any effect smaller than that is not separable from run-to-run variation in this study.

## 8. Step count does not indicate correctness

The halting distribution invites a reading in which samples that ran the full budget are the ones the model got wrong. That reading fails.

Binary exact-match, halt threshold 3, epoch 40: 4,583 of the 5,000 validation samples ran the full budget, while the whole validation set contained only 516 wrong answers. At least 4,067 of those 4,583 must have been correct.

At halt threshold 1.5 the relation runs the other way. The last-step group is too small to account for the errors, so most wrong answers come from samples that halted early. Both regimes appear across the runs, and per-epoch counts for all of them are in `logs/step-distributions.csv`.

Halting behaviour and accuracy are best reported as separate quantities.

## 9. Why halting does not help on this task

### 9.1 The halting mechanism is not broken

The halt head is trained with binary cross-entropy on its logit. For one sample, write **h** for the halt logit (the scalar the halt head outputs), **t** for the halting target (the value that sample is trained toward), and **σ** for the sigmoid, σ(h) = 1 / (1 + e^(-h)). The loss is

```
L = -[ t · log σ(h) + (1 - t) · log(1 - σ(h)) ]
```

Differentiate with respect to h. Using σ′(h) = σ(h)·(1 - σ(h)):

```
d/dh [ log σ(h) ]     = (1/σ(h)) · σ(h)(1-σ(h))        = 1 - σ(h)
d/dh [ log(1-σ(h)) ]  = (1/(1-σ(h))) · (-σ(h)(1-σ(h))) = -σ(h)
```

Substituting both into L:

```
∂L/∂h = -t·(1 - σ(h)) - (1 - t)·(-σ(h))
      = -t + t·σ(h) + σ(h) - t·σ(h)
      = σ(h) - t
```

Gradient descent then updates h ← h - lr·(σ(h) - t), where lr is the learning rate. When σ(h) < t the update is negative times negative, so h rises; when σ(h) > t, h falls. The gradient reaches zero when σ(h) = t.

Solving that condition for h inverts the sigmoid:

```
σ(h) = t
1 / (1 + e^(-h)) = t
1 = t · (1 + e^(-h))
1/t = 1 + e^(-h)
(1 - t)/t = e^(-h)
ln((1 - t)/t) = -h
h = ln(t / (1 - t))
```

This inverse of the sigmoid is the logit function, which is what the halt head's output is named after.

Three cases matter. `t = 0` gives `h = -∞`, so a sample no answer can satisfy is pushed arbitrarily negative and never halts. That is binary exact-match early in training, and the logged halt logits show it. `t = 1` gives `h = +∞`. And `t = 0.5` gives `h = 0` exactly, which is why σ(0) = 0.5 is the crossing point at halt threshold 0.

So the halt head's sigmoid output converges on its target. That is what binary cross-entropy is supposed to do, and the logged halt logits move this way in every run.

Because σ and its inverse are both increasing, the halting rule `h > threshold` can be rewritten entirely in terms of the target:

```
h > threshold   ⟺   t > σ(threshold)
```

**Halting begins when the target `t` passes σ(threshold).** For the three thresholds used here:

```
σ(0)   = 0.5000
σ(1.5) = 1 / (1 + e^-1.5) = 1 / 1.2231 = 0.8176
σ(3.0) = 1 / (1 + e^-3.0) = 1 / 1.0498 = 0.9526
```

The threshold sets a bar. The target definition sets what has to happen for a sample to clear it. Neither involves the model being right unless the target says so.

One qualification. This is where the gradient would reach zero if `t` were fixed. In training `t` rises as the model improves, so `h` trails a moving value rather than settling on it.

### 9.2 One wrong answer, three targets

Take a sample whose correct answer is `1234` and whose model output is `1239`, so three of four digits are right and the answer is wrong. Suppose the model assigns probability 0.95 to each of the three correct digits it got right, and 0.10 to the correct digit at the position it got wrong. These probabilities are illustrative, chosen to make the arithmetic concrete.

Soft-mean: `t` = fraction of positions correct = 3/4 = 0.75. Since 0.75 > σ(0) = 0.5, this sample clears the threshold-0 bar. The logit it converges to is `h = ln(0.75 / 0.25) = ln 3 = 1.099`. It halts, and the answer is wrong.

Geometric-mean: `t = (0.95 × 0.95 × 0.95 × 0.10)^(1/4)`. Step by step: 0.95³ = 0.857375; times 0.10 = 0.085738; ln(0.085738) = -2.4565; divided by 4 = -0.6141; e^(-0.6141) = 0.541. Also above 0.5, so it also clears the threshold-0 bar, but only just: `h = ln(0.541 / 0.459) = 0.165`. It halts too, and the answer is still wrong.

Binary exact-match: `t = 0`, because not every position is correct. The logit is driven negative and the sample does not halt.

Now read the same three numbers against the higher thresholds. Soft-mean's 1.099 is below 1.5, and geometric-mean's 0.165 is far below it, so at threshold 1.5 neither sample halts on this evidence alone. That is the mechanism behind section 6: raising the bar does not repair the target, it just requires the target to climb further before halting engages, which pushes halting past the point where the model has actually learned the task.

It also accounts for the ordering in section 5. Soft-mean's target crosses 0.5 as soon as half the digits are right. Geometric-mean's crosses when the confidence-weighted score does, which takes longer. Binary exact-match's crosses only when half the answers are fully right.

### 9.3 The accuracy rise is the model learning the task

The `n_sup = 1` runs have a single supervision step and therefore no halting decision to make, yet they show the same steep rise at epoch 20 to 30 in all three targets, and reach 50% accuracy earlier than any larger budget (section 7). No halting configuration produces the rise, and none delays it beyond where the model would have learned anyway.

### 9.4 Recursion is not needed here, so halting has nothing to allocate

One supervision step reaches the same final accuracy as sixteen, within about one point (section 7). The architecture's premise, that hard inputs deserve more passes, has nothing to act on when a single pass already solves the task.

This tests deep supervision, the outer loop. The inner recursion of T = 3 cycles with n = 6 latent updates runs in every configuration, including `n_sup = 1`, so it is never ablated and its contribution is untested here. A matched non-recursive baseline would settle it.

A correct halting target and a badly-chosen one therefore end at the same accuracy on 4-digit addition, because the compute they allocate is not compute the task needs. That is why the failure in section 5 is silent: the quantity that would expose it, recursion mattering to the answer, is inactive.

### 9.5 What this task can and cannot measure

It measures halting behaviour: when halting engages, how it responds to the threshold, how it tracks the target. It cannot measure the accuracy cost of getting halting wrong. That needs a task where recursion pays for itself.

The derivation in 9.1 says the halt head converges toward its target, and the logged halt logits move consistently with it, but these runs do not record the mean target itself, because it is computed inside the loss and discarded. The threshold sweep in section 6 is indirect evidence for the relation, not a direct verification of it. Logging the batch-mean target is listed under planned additions for that reason.

## 10. Limitations

### 10.1 One run per configuration, no seed control

Every result here comes from a single run with no seed set, so initialisation variance is included in the bounds below rather than controlled for. The `n_sup = 1` runs have no halting decision to make, so their epoch-to-epoch movement is ordinary training variation and gives a bound on the study's own noise.

| target | lowest accuracy, epochs 50 to 90 | highest accuracy, epochs 50 to 90 | spread |
|---|---|---|---|
| soft-mean | 76.86% | 98.90% | 22.04 |
| geometric-mean | 95.82% | 99.10% | 3.28 |
| binary exact-match | 87.20% | 99.06% | 11.86 |

*Validation sequence exact-match out of 5,000 samples, taken from the five logged epochs 50 through 90 of the `n_sup = 1`, halt-threshold-0 run for that target. `spread` is the highest minus the lowest, in percentage points. These runs use a single supervision step, so no halting decision exists in them. The three columns come from separate notebook runs on independently generated datasets and initialisations; see section 4.*

Four stability mechanisms used by the paper are absent here: an exponential moving average of the weights, weight decay, the paper's lower beta2 of 0.95, and stable-max loss. The paper includes EMA to prevent sharp collapse on small data and reports a 7.5-point cost on Sudoku-Extreme when it is removed. The swings above have that signature, a sharp drop with full recovery inside twenty epochs. All four are constant across all 15 runs, so they raise the noise floor uniformly rather than producing any difference between targets, but they are the most likely reason the floor is as high as it is.

A single supervision step removes the halting decision, not the halting loss. The halt head is still trained in these runs, so each bounds training variation for its own target's auxiliary objective rather than for a model with no halting machinery at all.

Any effect smaller than a target's own spread cannot be separated from run-to-run variation here. That matters unevenly across the results:

- The section 5 finding is a difference in kind rather than in degree: in two targets halting saturates below 15% accuracy, in one it saturates above 96%. No redraw of 50,000 uniform 4-digit pairs produces that.
- The section 6 threshold costs, at 1 to 10 points, are not outside them. Read the direction, not the magnitude.
- The section 7 result that the recursion budget changes accuracy by at most 1.02 points is a null result stated against noise of the same size or larger, so it is a bound rather than a measurement.

### 10.2 One task, one network, and a network unlike the paper's

4-digit addition with a 740K-parameter model whose recursion network is a 2-layer MLP over a single vector per sample. The paper uses a 2-layer Transformer over per-position representations. The comparison between halting targets is unaffected, since the network is identical across all 15 runs and the mechanism in section 9.1 refers only to the halt logit, but nothing here establishes that the ordering transfers to the paper's architecture or to other tasks.

### 10.3 Recursion is unused at this task's accuracy ceiling

One supervision step matches sixteen (section 7). This is what makes the failure silent and is therefore part of the finding, but it also means these runs cannot show an accuracy cost of premature halting. Demonstrating that cost requires a task where recursion pays for itself.

### 10.4 What the logs do not record

Per-supervision-step correctness. The runs log halt logits at each step but not accuracy at each step, so no claim is made about whether an answer improves or degrades when a sample is forced to continue.

The halting target itself. It is computed inside the loss and discarded. A direct check of the relation in section 9.1 needs the batch-mean target logged alongside the halt-logit statistics.

### 10.5 Known quirks in the logged quantities

These affect how the logs read, not the comparisons, because each applies identically to all 15 runs.

The last-step group conflates two cases. A sample that never halts is recorded at the final step, the same value as one that halted exactly there. So `%never` and the final column of every halting-step distribution are upper bounds on true never-halting. Steps 1 through `n_sup - 1` are unambiguous.

`Avg Steps` is batch-level. It counts how many supervision steps the batch executed before every sample in it had halted, averaged over batches, not the mean per-sample halting step. One sample that never halts holds the whole batch at `n_sup`. Per-sample behaviour is in the halting-step distributions.

Halt-logit statistics include already-halted samples, because the forward pass runs on the full batch at every step and halted samples contribute logits from frozen hidden states.

Training cross-entropy is deflated by the pad fraction. Pad positions contribute zero to the numerator but are counted in the denominator, so the logged value is lower than the per-real-token cross-entropy. The halting targets exclude pads correctly and only the logged cross-entropy is affected.

The per-sample mask on the halting loss is inert. The loss function returns a batch-mean scalar, so multiplying by a sample mask and renormalising returns the same scalar. The halting loss averages over all samples in the batch, including halted ones.

## Departures from the TRM paper

| setting | this study | the paper |
|---|---|---|
| inner recursion | T = 3 cycles, n = 6 latent updates | same |
| recursion network | 2-layer MLP, linear, ReLU, linear | 2-layer Transformer with RMSNorm, no bias, rotary embeddings, SwiGLU |
| representation | one vector per sample, size 256 | one vector per position |
| normalisation | none | RMSNorm |
| halting loss weight `c` | 0.1 | 1.0, added without a coefficient |
| supervision budget `n_sup` | swept: 1, 5, 16 | 16 |
| halting at evaluation | used, decides which step's output is scored | not used, full budget runs at test time |
| optimizer | Adam, lr 1e-4, default betas | AdamW, beta2 = 0.95, lr 1e-4, 2K warmup |
| weight decay | none | 1.0 on Sudoku-Extreme and Maze-Hard |
| EMA of weights | not used | 0.999 |
| output loss | cross-entropy | stable-max |

*Columns: `setting` is the configuration item, and the other two give its value here and in the paper. Values for this study are read from the notebooks; values for the paper are from its hyperparameter section, architecture description and pseudocode. The evaluation difference means accuracy values here are not directly comparable to the paper's.*

## Repository layout

```
├── README.md
├── notebooks/
│   ├── trm_qloss_softmean.ipynb    5 configurations, outputs retained
│   ├── trm_qloss_binary_em.ipynb   5 configurations, outputs retained
│   └── trm_qloss_geomean.ipynb     5 configurations, outputs retained
└── logs/
    ├── softmean/  geomean/  binary-em/     5 files each, one per configuration
    ├── per-epoch.csv           one row per (target, configuration, epoch)
    ├── step-distributions.csv  one row per (target, configuration, split, epoch, step)
    ├── supstep-halt-logits.csv mean validation halt logit per supervision step
    └── METRICS.md              every logged field and how the code computes it
```

`logs/METRICS.md` documents the two places the logs are ambiguous: the last-step group conflates "halted at the final step" with "never halted", and `Avg Steps` is batch-level rather than per-sample.

## Reproducing

Each notebook is self-contained: it builds the dataset, defines the model and its halting target, and runs the five configurations in order. Cell outputs are retained, so the logged numbers can be read without running anything. The files in `logs/` are those outputs parsed into per-epoch tables.

No model checkpoints are released. The runs are short enough to regenerate from the notebooks directly. Note that without seeds a re-run draws a new dataset and a new initialisation, so numbers will differ from those recorded here.

## Planned additions

- Set seeds for dataset generation, split and initialisation, then run three seeds on the nine threshold runs, to put the section 6 costs outside the bound in section 10.1 and to put all three targets on identical data.
- Add EMA at 0.999 and re-run the threshold sweep, to test whether the noise floor in section 10.1 falls far enough to make the section 6 threshold costs separable from variation.
- A sensitivity check at `c = 1.0`, the paper's halting-loss weight, on the three threshold-0 runs.
- Log the batch-mean halting target each epoch, turning section 9.1 into a directly verified relation.
- Forced-rollout per-step accuracy at each checkpoint.
- A task where recursion is required, to measure the accuracy cost of premature halting.

## Reference

Tiny Recursive Model: [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)
