# Halting-target choice: original 4-digit study

This study compares three halting objectives in a simplified MLP implementation inspired by the Tiny Recursive Model ([arXiv:2510.04871](https://arxiv.org/abs/2510.04871)).

Sections 1–10 cover the original 15 unseeded 4-digit runs. The measured results are unchanged; some interpretations have been corrected. The main seeded 4-digit and 8-digit experiments have a separate [three-seed results report](seeded_results.md), summarised in section 11 below. The [original notebooks, logs and report](../archive/original_unseeded/readme.md) are preserved locally without changes, as well as in [Git history](https://github.com/zyberg2091/trm-halting-targets/tree/0294880997046b3a8352796b7ce58a2d27717b63).

## 1. Recursive models with deep supervision

The Tiny Recursive Model repeatedly updates a candidate answer to a task.

The model keeps a current answer `y` and a latent state `z`. In each inner cycle, a shared network updates `z` n times using the input, `y` and `z`, then updates `y` once using `y` and `z` without the input. One **supervision step** contains T such cycles. This implementation uses n = 6 and T = 3, following the paper. The first T-1 cycles run without gradients; only the final cycle is differentiated.

The recursion follows the paper, but the network is a 2-layer MLP in place of the paper's 2-layer Transformer. Each sample is represented by one vector; the paper keeps one vector per position.

Deep supervision repeats this process up to `n_sup` times, using each revised answer as the starting point for the next step. Each step contributes a loss against the ground truth. The model makes halting decisions during this loop, then updates its parameters once at the end of the batch.

## 2. The halting mechanism

At each supervision step, a small **halt head** produces a scalar, the **halt logit** `h`. A sample stops when `h` exceeds a fixed **halt threshold**. Its current answer is then final.

The halt head is trained with binary cross-entropy against a target `t`. The choice of `t` determines what the head learns to treat as a reason to stop.

In the TRM paper, `t` is binary exact-match, 1 if every token of the answer is correct and 0 otherwise.

## 3. The modification under test

This study compares the paper's halting target with two alternatives. The model architecture, data-generation process, optimizer, schedule and recursion structure in section 1 are the same for all three. Corresponding configurations differ only in the definition of `t`.

An answer has `L` non-pad tokens. `y_i` is the ground-truth token at position i, `ŷ_i` the model's argmax there, and `p_i` the probability the model assigns to `y_i`. Pad positions are excluded from all three targets.

**Binary exact-match**, the paper's target: `t = 1` if `ŷ_i = y_i` at every one of the `L` positions, else `t = 0`.

**Soft-mean**, substitution 1: `t` = the fraction of positions where `ŷ_i = y_i`. Three correct digits out of four gives `t = 0.75`.

**Geometric-mean**, substitution 2: `t` = the geometric mean of the correct-token probabilities, `(∏ p_i)^(1/L)`, computed in log space as `exp(-mean cross-entropy)`. This uses the model's confidence in each correct token. A low probability at one position lowers the whole score.

The motivation was to give the halt head more feedback early in training. Binary exact-match is 0 for almost every sample while the model is still learning to produce a complete answer. Soft-mean gives partial credit for correct positions, and geometric-mean uses the probabilities of the correct tokens. Binary zeros still train the head to keep going, though, so adding partial credit changes the meaning of the stopping target; it is not simply more information about the same objective.

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

Hyperparameters were fixed across the original 15 runs and were not tuned. They may still interact with the halt target.

The paper adds the halting loss at weight 1.0; these runs use `c = 0.1`. With a fixed target and an independently optimised logit, a positive `c` scales the gradient `c · (σ(h) - t)` without moving its stationary point. Here, however, the model shares weights and the targets change during training. A separate ablation is needed to tell how the smaller weight affects the results; it cannot be assumed to change only the timing of halting or to make the findings conservative.

Four stability mechanisms used by the paper are absent here: an exponential moving average of the weights at 0.999, weight decay, the paper's lower beta2 of 0.95, and stable-max loss. Section 10.1 describes the variation in these runs; their individual effects were not tested.

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

There are 15 runs in total: five configurations for each of the three targets.

No random seed was set. All 15 runs use separate model initialisations. Each target notebook also generates its own dataset: 50,000 uniform 4-digit pairs, split 45,000 and 5,000. The five configurations within a notebook share that dataset and split. The large samples come from the same task distribution, but the comparisons across targets do not use identical examples.

## 5. What the halting target does

Both graded targets lead to nearly all validation samples stopping at the first step while most answers are still wrong. With binary exact-match, nearly all samples stop only after most answers are correct. These are observations within each run. Since the three runs use independently generated datasets, the comparison shows a qualitative pattern, not a controlled estimate of the difference between targets.

Samples using only one supervision step, at halt threshold 0 with `n_sup = 5`:

| epoch | soft-mean | geometric-mean | binary exact-match |
|---|---|---|---|
| 0 | 0.0% (0) | 0.0% (0) | 0.0% (0) |
| 10 | 98.2% (4908) | 41.0% (2048) | 0.0% (0) |
| 20 | 100.0% (5000) | 72.4% (3622) | 0.0% (0) |
| 30 | 100.0% (5000) | 92.2% (4609) | 0.1% (3) |
| 40 | 100.0% (5000) | 99.98% (4999) | 81.0% (4048) |
| 50 | 100.0% (5000) | 100.0% (5000) | 100.0% (5000) |
| 60 to 90 | 100.0% (5000) | 100.0% (5000) | 100.0% (5000) |

*Percentage of the 5,000 validation samples that used only one supervision step, raw count in brackets. Percentages are calculated from the logged counts; 4,999 is shown as 99.98% to distinguish it from 5,000. A sample counts here if its halt logit exceeded the threshold at step 1, so it stopped after one supervision step and took no further supervision steps. All three targets sit at 100% from epoch 50 through 90; the four checkpoints from 60 through 90 share one row. The three columns come from separate notebook runs on independently generated datasets and initialisations; see section 4.*

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


With soft-mean, every validation sample stops at the first step by epoch 20, when exact match is only 9.18%. Geometric-mean stops 4,999 of 5,000 samples at epoch 40, with 14.14% exact match. It first stops all 5,000 at epoch 50, when exact match has reached 94.48%. Binary exact-match also stops all 5,000 at epoch 50, at 96.86%.

The same ordering appears when 50% and 90% of samples stop at the first step. Reaching exactly 100% gives a different comparison, so rounded percentages can be misleading:

| level of first-step halting | soft-mean | geometric-mean | binary exact-match |
|---|---|---|---|
| epoch it first reaches 50% | 10 | 20 | 40 |
| accuracy at that epoch | 2.52% | 7.82% | 66.80% |
| epoch it first reaches 90% | 10 | 30 | 50 |
| accuracy at that epoch | 2.52% | 7.68% | 96.86% |
| epoch it first reaches 100% | 20 | 50 | 50 |
| accuracy at that epoch | 9.18% | 94.48% | 96.86% |

*Rows alternate between two quantities: an epoch number, and validation exact-match at that epoch. Both come from the two tables above. Each column is read from its own notebook's runs. At the 50% and 90% levels, both graded targets stop early while exact match is below 15%; binary exact-match reaches those levels at 66.80% or above. At literal 100%, geometric-mean is already at 94.48%.*

This comparison is at halt threshold 0. Section 6 covers what happens at higher thresholds.

## 6. Raising the halt threshold

A higher threshold requires a larger halt logit before a sample can stop. In these runs, it delays widespread first-step halting for the graded targets, but the effect differs by target.

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

For soft-mean, 90% first-step halting moves from epoch 10 to 40 to 70 as the threshold rises from 0 to 1.5 to 3.0. Accuracy at those points is 2.52%, 53.28% and 90.04%. Geometric-mean reaches the same level at epoch 30 and epoch 60 for thresholds 0 and 1.5, with 7.68% and 96.24% accuracy. At threshold 3.0 it never reaches that level within 100 epochs.

Binary exact-match reaches the 90% level at epoch 50 at both threshold 0 and threshold 1.5. There is no delay visible at the logged checkpoints. Accuracy at that epoch is 96.86% and 86.14%, respectively, so the higher threshold does not improve accuracy in this comparison.

In these original runs, first-step halting at threshold 3.0 fluctuates between checkpoints:

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

At the logged checkpoints through epoch 30, no sample halts at step 1 in any target. From epoch 40 the number moves up and down without settling. Binary exact-match ends with 56 of 5,000 samples stopping at step 1 and 4,833 using the full budget at epoch 90. Its logits fell from an average of 3.174 to 1.204 (see the [original halt-logit table](../archive/original_unseeded/logs/supstep-halt-logits.csv)), dropping almost the whole set below the cutoff, alongside validation accuracy falling from 87.94% to 81.12%.

Each validation sample has its own halt logit. The table counts how many of the 5,000 exceed the cutoff. At threshold 3.0, changes in the logits move many samples across that cutoff between checkpoints. At threshold 0, all logits remain above the cutoff at the later logged checkpoints, keeping the share at 100%.

First-step halting at threshold 3.0 does not settle in these original 100-epoch runs. The [seeded results](seeded_results.md#2-graded-targets-lead-to-stopping-while-most-answers-are-still-wrong) check the effect with fixed seeds and a longer budget.

Best accuracy in the run is highest at threshold 0 for all three: 99.68%, 99.64% and 99.46%. At the other two thresholds it ranges from 90.04% to 98.72%. Best-checkpoint is used here rather than last-epoch accuracy, and the two differ for soft-mean at threshold 0: 99.68% at epoch 70 against 97.26% at epoch 90, from the section 5 accuracy table. These are selected best checkpoints, not estimates of average or final-checkpoint performance.


## 7. Changing the recursion budget

The supervision budget `n_sup` is the maximum number of steps a sample may take. These runs hold the halt threshold at 0 and vary the budget: 1, 5, 16.

`n_sup = 1` removes adaptive stopping: every sample receives exactly one outer supervision step, which still contains inner recursion. The halt loss remains active, so this is not a control with all halting machinery removed.

The best logged accuracies are close across the three budgets:

| target | n_sup = 1 | n_sup = 5 | n_sup = 16 | spread |
|---|---|---|---|---|
| soft-mean | 98.90% | 99.68% | 98.66% | 1.02 |
| geometric-mean | 99.10% | 99.64% | 98.76% | 0.88 |
| binary exact-match | 99.06% | 99.46% | 99.06% | 0.40 |

*Highest validation exact-match reached at any of the ten logged epochs of that run, out of 5,000 validation samples. Halt threshold is 0 in all nine runs and only the supervision budget differs. `spread` is the largest value minus the smallest for that target, in percentage points. One run per configuration. Within each target, budgets share the dataset and split but use separate model initialisations; see section 4.*

In these original runs, sixteen supervision steps do not beat one, and the largest within-target spread is 1.02 percentage points. With one unseeded run per setting, this does not establish that the ordering is noise or that extra supervision can never help. The [seeded comparison](seeded_results.md#4-more-supervision-steps-do-not-consistently-beat-one-step-training) revisits that conclusion.

The larger budgets reach 50% validation exact match later in these original runs:

| target | n_sup = 1 | n_sup = 5 | n_sup = 16 |
|---|---|---|---|
| soft-mean | 30 | 40 | 40 |
| geometric-mean | 30 | 50 | 50 |
| binary exact-match | 30 | 40 | 50 |

*First logged epoch at which validation accuracy reaches 50%, same nine runs. Lower is faster. Within each target, budgets share the dataset and split but use separate model initialisations; see section 4.*

The single-step runs reach 50% accuracy at epoch 30 in all three targets. Every larger budget takes 10 to 20 epochs longer. The epoch axis understates the gap in compute: before halting engages, an `n_sup = 16` run executes sixteen supervision steps per batch where the single-step run executes one.

At `n_sup = 16` with halt threshold 0, all three targets reach 100% validation step-1 halting by epoch 60, ten epochs later than at `n_sup = 5`. The extra steps remain unused at the subsequent logged validation checkpoints.

The single-step runs also show a steep accuracy rise between epoch 20 and 30 for all three targets. These runs have no adaptive stopping decision, so that rise does not require a change in stopping behaviour. The halting loss is still part of training.

Those runs also show substantial within-run training instability:

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

Over epochs 50 to 90 these runs span 22.04 points (soft-mean), 3.28 (geometric-mean) and 11.86 (binary exact-match). Soft-mean drops from 98.74% to 76.86% and back to 98.90% in twenty epochs with no halting involved. These within-run swings are not a bound on run-to-run uncertainty; repeated seeded runs are needed to assess reproducible differences.

## 8. Step count does not indicate correctness

Using the full supervision budget does not imply that an answer is wrong.

Binary exact-match, halt threshold 3, epoch 40: 4,583 of the 5,000 validation samples ran the full budget, while the whole validation set contained only 516 wrong answers. At least 4,067 of those 4,583 must have been correct.

At halt threshold 1.5, the last-step group is too small to contain all the errors; most wrong answers come from samples that stopped earlier. Both patterns occur in these runs. The [original step-distribution table](../archive/original_unseeded/logs/step-distributions.csv) gives the counts at each epoch.

Halting behaviour and accuracy are best reported as separate quantities.

## 9. Interpreting the original halting results

### 9.1 What the halt loss encourages

For one sample, let **h** be the halt logit, **t** the halting target, and **σ** the sigmoid, σ(h) = 1 / (1 + e^(-h)). The binary cross-entropy loss is

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

If h were optimised on its own, this derivative would push it up when σ(h) < t and down when σ(h) > t. For fixed t, the derivative is zero at σ(h) = t. The model actually updates shared weights, so the derivative describes the loss's local direction; it does not give the change in each sample's logit after an update.

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

For this fixed-target idealisation, `t = 0` favours increasingly negative logits, `t = 1` favours increasingly positive logits, and `t = 0.5` has its stationary point at `h = 0`. The infinite limits are not claims about logits reached during finite training.

Binary cross-entropy encourages the halt prediction to fit its target. Shared parameters and changing targets mean that convergence or calibration is not guaranteed by this derivative alone.

Because sigmoid is increasing, the exact halting rule is:

```
h > threshold   ⟺   σ(h) > σ(threshold)
```

Only an ideal fit to a fixed target lets us replace `σ(h)` with `t`. Comparing fitted halt scores with this cutoff helps explain the behaviour, but does not predict the exact onset of halting. The three thresholds used here correspond to:

```
σ(0)   = 0.5000
σ(1.5) = 1 / (1 + e^-1.5) = 1 / 1.2231 = 0.8176
σ(3.0) = 1 / (1 + e^-3.0) = 1 / 1.0498 = 0.9526
```

The threshold sets the required halt score; the target defines what that score is trained to represent. Even with a binary exact-match target, the trained head can stop on a wrong answer.

During training, targets and shared representations change, and target values need not rise monotonically. Whole-dataset exact-match accuracy is not a ceiling on individual halt logits.

### 9.2 One wrong answer, three targets

Suppose the correct answer is `1234` and the model outputs `1239`. Three of four digits are correct. For illustration, give each correctly predicted digit probability 0.95, and the correct digit at the wrong position probability 0.10. The decisions below assume a halt head that fits each fixed target exactly; a trained head need not do so.

Soft-mean: `t` = fraction of positions correct = 3/4 = 0.75. Since 0.75 > σ(0) = 0.5, this sample clears the threshold-0 bar. The logit it converges to is `h = ln(0.75 / 0.25) = ln 3 = 1.099`. It halts, and the answer is wrong.

Geometric-mean: `t = (0.95 × 0.95 × 0.95 × 0.10)^(1/4)`. Step by step: 0.95³ = 0.857375; times 0.10 = 0.085738; ln(0.085738) = -2.4565; divided by 4 = -0.6141; e^(-0.6141) = 0.541. Also above 0.5, so it also clears the threshold-0 bar, but only just: `h = ln(0.541 / 0.459) = 0.165`. It halts too, and the answer is still wrong.

Binary exact-match: `t = 0`, because not every position is correct. The logit is driven negative and the sample does not halt.

Now read the same three numbers against the higher thresholds. Soft-mean's 1.099 is below 1.5, and geometric-mean's 0.165 is far below it, so at threshold 1.5 neither sample halts on this evidence alone. This illustrates why a higher cutoff can delay stopping for the same fitted score. It does not guarantee that stopping waits until the task is learned.

This example helps explain the ordering in section 5. A partially correct answer can have a positive graded target while its binary exact-match target is zero. The actual timing still has to be measured: the dataset's mean target cannot be substituted for every sample's predicted score.

### 9.3 The accuracy rise is the model learning the task

The `n_sup = 1` runs have a single supervision step and therefore no halting decision to make, yet they show the same steep rise at epoch 20 to 30 in all three targets, and reach 50% accuracy earlier than any larger budget (section 7). The accuracy rise therefore does not require a transition in adaptive stopping. These observations do not isolate the cause of its timing.

### 9.4 Extra supervision shows little benefit in the original runs

One supervision step reaches similar best logged accuracy to sixteen in the original runs (section 7). This limits what those runs establish about the benefit of assigning additional outer steps.

This comparison tests the outer supervision loop. Every configuration, including `n_sup = 1`, retains the inner recursion of T = 3 cycles with n = 6 latent updates. Its contribution remains untested; that needs a matched non-recursive baseline.

Different targets can have similar best logged accuracy despite very different stopping behaviour. This does not prove that the task never benefits from additional computation; the [seeded runs](seeded_results.md#4-more-supervision-steps-do-not-consistently-beat-one-step-training) show target-dependent exceptions.

### 9.5 What this task can and cannot measure

These runs show when samples begin stopping early and how that changes with the target and threshold. They do not isolate the accuracy cost of early stopping, because correctness was not measured with halting disabled at each depth. A stronger test of adaptive computation also needs a task where additional steps reliably improve the answer.

Section 9.1 describes the loss's fixed-target preference. The original runs discard the target after computing the loss, so the threshold sweep is not a direct verification of target fitting. Logging the target alongside the predicted halt probability remains useful.

## 10. Limitations

These limitations describe the original exploratory study. The [separate seeded study](seeded_results.md) adds three seeds, 8-digit addition and forced-depth measurements; its remaining limitations are stated there.

### 10.1 One run per configuration, no seed control

Each configuration in the original analysis has one unseeded run. The table below describes within-run variation with `n_sup = 1`; it does not estimate initialisation variance, run-to-run uncertainty, or confidence intervals. The [follow-up report](seeded_results.md) adds seeded runs.

| target | lowest accuracy, epochs 50 to 90 | highest accuracy, epochs 50 to 90 | spread |
|---|---|---|---|
| soft-mean | 76.86% | 98.90% | 22.04 |
| geometric-mean | 95.82% | 99.10% | 3.28 |
| binary exact-match | 87.20% | 99.06% | 11.86 |

*Validation sequence exact-match out of 5,000 samples, taken from the five logged epochs 50 through 90 of the `n_sup = 1`, halt-threshold-0 run for that target. `spread` is the highest minus the lowest, in percentage points. These runs use a single supervision step, so no halting decision exists in them. The three columns come from separate notebook runs on independently generated datasets and initialisations; see section 4.*

These runs omit four stability mechanisms from the paper: an exponential moving average of the weights, weight decay, the lower beta2 of 0.95, and stable-max loss. The paper uses EMA to prevent sharp collapse on small datasets and reports a 7.5-point cost on Sudoku-Extreme when it is removed. The sharp drops and recoveries here look similar, sometimes reversing within twenty epochs. The missing mechanisms may contribute, but these runs do not measure their individual effects or show that they affect all targets equally.

A single supervision step removes the halting decision, not the halting loss. The halt head is still trained in these runs, so each shows training variation with its own auxiliary objective, rather than for a model with no halting machinery at all.

The near-total early stopping of the graded targets is a clear observation in the original traces. The [follow-up report](seeded_results.md) tests its repeatability with seeded comparisons; smaller differences still need to be interpreted with the limited seed count in mind. Within-run swings alone neither prove nor disprove those differences.

### 10.2 One task, one network, and a network unlike the paper's

The original study uses 4-digit addition and a 740K-parameter model, with a 2-layer MLP operating on one vector per sample. The paper uses a 2-layer Transformer with a representation at each position. Keeping the architecture fixed across the original 15 runs does not rule out an interaction with the halt target. These results do not establish the same ordering on the paper's architecture or on other tasks.

### 10.3 Inner recursion was not ablated

The original runs show little best-checkpoint benefit from larger outer supervision budgets (section 7). Every configuration retains inner recursion, so its necessity is untested. The [new forced-depth measurements](seeded_results.md#3-an-extra-step-can-make-a-good-answer-worse) address whether continuing the trained model improves its answer; they do not replace an inner-recursion ablation.

### 10.4 What the logs do not record

The original logs record halt logits at each supervision step, but not correctness. They cannot show whether forcing the model to continue improves its answer. The [seeded study](seeded_results.md#3-an-extra-step-can-make-a-good-answer-worse) includes these measurements.

The halting target is also missing: it is computed for the loss and then discarded. A direct check of the fitting account in section 9.1 needs per-sample targets alongside predicted halt probabilities. Batch means would show only aggregate agreement.

### 10.5 Known quirks in the logged quantities

These quirks apply to the original implementation. Some affect training as well as the logged values; sharing a quirk across targets does not rule out interactions with those targets. The [seeded report](seeded_results.md#1-experimental-setup) identifies the halt-loss mask fix and the padding issue that remains.

The last-step group includes both samples that never halt and samples that halt exactly at the final step. This makes `%never` and the final column of each halting-step distribution upper bounds on the number that never halt. Steps 1 through `n_sup - 1` are unambiguous.

`Avg Steps` is batch-level. It counts how many supervision steps the batch executed before every sample in it had halted, averaged over batches, not the mean per-sample halting step. One sample that never halts holds the whole batch at `n_sup`. Per-sample behaviour is in the halting-step distributions.

Halt-logit statistics include already-halted samples, because the forward pass runs on the full batch at every step and halted samples contribute logits from frozen hidden states.

Training cross-entropy is deflated by the pad fraction. Pad positions contribute zero to the numerator but are counted in the denominator, so the logged value is lower than the per-real-token cross-entropy. This scales both the training prediction loss and the logged cross-entropy. The halting targets exclude pads correctly.

The per-sample mask on the halting loss has no effect. The loss function has already averaged across the batch, so applying a sample mask and renormalising leaves the same scalar. Already-halted samples still contribute to that average.

## 11. Follow-up: seeded 4-digit and 8-digit results

[**Halting on 4-digit and 8-digit addition: three-seed results**](seeded_results.md) contains the main seeded analysis, accuracy table and limitations. It uses seeds 0, 40 and 100 and extends the original 4-digit experiments to 8 digits with a 200-epoch budget.

The graded targets still lead to early stopping. Forcing step 2 lowers token accuracy at **155 of 161** fully collapsed checkpoints, a comparison the original logs could not provide. The accuracy finding is more mixed: at 8 digits, one step wins all six graded-target comparisons, while five steps win for binary exact-match at two of three seeds. The new report describes the implementation changes and the limits of these comparisons.

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

## Repository layout and extraction

The current [notebooks](../notebooks/) and [text logs](../logs/) are grouped by digit count, epoch budget, seed and target. Each of the 21 notebooks retains all saved outputs; its matching text log includes five main runs plus diagnostics and dataset previews. The original unseeded evidence is linked at the top of this report.

The **AI-generated extraction script** copies saved output text without executing notebook code. From the repository root:

```bash
python scripts/extract_logs.py
python scripts/extract_logs.py --check
```

All 105 saved main-run outputs cover their configured epoch budgets. Detailed metrics are printed every ten epochs. No checkpoints or pinned training environment are included; fixed seeds alone do not guarantee bit-for-bit training reproduction.

## AI assistance

AI helped with finding literature and with organising and editing parts of this report. I built the experiments independently, checked the reported results, and read the referenced papers and technical claims.

## Planned additions

- **Completed:** three-seed reruns and forced-depth per-step evaluation; see the [seeded results report](seeded_results.md).
- Add EMA at 0.999 and repeat the threshold sweep to test its effect on stability and variation across seeds.
- A sensitivity check at `c = 1.0`, the paper's halting-loss weight, on the three threshold-0 runs.
- Log targets alongside predicted halt probabilities to test the fitting account in section 9.1 directly.
- A task where recursion is required, to measure the accuracy cost of premature halting.

## Reference

Tiny Recursive Model: [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)
