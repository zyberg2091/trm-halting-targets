# Halting on 4-digit and 8-digit addition: three-seed results

Across both task sizes and all three seeds, the graded targets lead to earlier stopping at threshold 0. Once every training sample stops at the first supervision step, forcing later steps usually makes predictions worse. Extra supervision gives no consistent accuracy gain.

These runs use a simplified TRM-inspired MLP with seeds **0, 40 and 100**: **100 epochs for 4 digits** and **200 for 8 digits**. Soft-mean measures the fraction of correct answer tokens, geometric-mean uses correct-token probabilities, and binary exact-match requires the whole answer to be correct. Each target has five configurations: five supervision steps at thresholds 0, 1.5 and 3.0, and one or sixteen steps at threshold 0. Here, a supervision step means one outer refinement step. All configurations still use inner recursion.

Saved [notebook](../notebooks/) outputs are also available as [text logs](../logs/). The 8-digit seed-0 main-run outputs saved at 100 epochs match the first 100 epochs of the corresponding 200-epoch runs, so they are not extra repetitions. The [original report](technical_report.md) covers the model, loss derivations and earlier unseeded experiments.

## 1. The target changes when early stopping takes over

To compare when stopping takes over, consider the first logged checkpoint where at least half the **training samples** stop at step 1. In the five-step runs at threshold 0, soft-mean, geometric-mean and binary exact-match reach this point at epochs **10, 20 and 40** on 4 digits, with the same result at every seed. On 8 digits, the corresponding epochs are **30, 80–90 and 100–140**.

The halt head learns to predict its target. If that target rewards partly correct answers, the model can learn to stop before it can produce a correct whole answer. **Soft-mean reaches this onset point at 0.00% validation exact match with halting enabled in all three 8-digit seeds.** Thresholds 0 and 3.0 require sigmoid halt scores above 50% and about 95.26%, respectively. Those are cutoffs for individual predictions, not required dataset accuracies. No 8-digit binary-EM run at threshold 3.0 has a majority of training samples stopping at step 1 at any logged checkpoint. Both the target and threshold affect when collapse begins and whether it finishes; whole-dataset accuracy cannot bound each sample's halt logit.

The higher-threshold runs show more selective stopping. At checkpoints where some validation samples stop at step 1 and others continue, the stopping group has higher exact-match accuracy than the full validation set evaluated at step 1 in **116/117 cases at threshold 3.0**, versus **60/92 at threshold 0**. These counts combine all targets and seeds of the five-step runs at both task sizes. Correct-answer counts were recovered from the logged group sizes and rounded rates. These separately trained runs show selection, not a controlled threshold change on a fixed model.

## 2. Complete early stopping removes later-step supervision

When every training sample stops at step 1, the loop ends before computing the later supervision losses. The shared weights still train on step 1. With partial collapse, samples that continue still provide supervision at the steps they reach.

Forcing step 2 during evaluation shows why this distinction matters:

| Training samples stopping at step 1 | Step 2 improves token accuracy: 4 digits | Step 2 improves token accuracy: 8 digits |
|---|---:|---:|
| Less than 50% | 98/119 | 395/398 |
| Partial collapse: 50% to less than 100% | 80/166 | 209/236 |
| Complete collapse: exactly 100% | 0/75 | 6/86 |

*Counts cover logged checkpoints of all multi-step runs in the 4-digit/100-epoch and 8-digit/200-epoch experiments. Classification uses exact training counts, not percentages rounded to 100%. Accuracy is measured on validation samples with halting disabled.*

Overall, step 2 lowers token accuracy in **155/161 completely collapsed checkpoints**. It usually helps under partial collapse at 8 digits, but the 4-digit results are mixed. Keeping later losses active is no guarantee of improvement. At the last saved checkpoint of each 8-digit soft-mean, five-step, threshold-0 run, forcing step 2 changes exact match from **86.44% to 70.54%** at seed 0, **86.88% to 72.22%** at seed 40, and **84.16% to 83.98%** at seed 100.

## 3. Extra supervision has no consistent accuracy advantage

Each cell reports **one-step / best five-step peak forced-depth exact match (%)**. Exact match requires the whole answer to be correct; forced-depth evaluation continues for a fixed number of outer steps while ignoring halting.

| Task | Halt target | Seed 0 | Seed 40 | Seed 100 |
|---|---|---:|---:|---:|
| 4-digit | Soft-mean | 99.44 / 98.90 | 98.96 / 99.44 | 98.10 / 99.30 |
| 4-digit | Geometric-mean | 97.80 / 99.44 | 98.74 / 99.64 | 99.28 / 99.00 |
| 4-digit | Binary exact-match | 99.74 / 98.72 | 99.72 / 99.18 | 99.28 / 99.36 |
| 8-digit | Soft-mean | 89.62 / 86.44 | 91.94 / 86.88 | 91.36 / 87.08 |
| 8-digit | Geometric-mean | 92.66 / 89.06 | 92.30 / 88.96 | 91.56 / 89.68 |
| 8-digit | Binary exact-match | 93.16 / 93.76 | 87.28 / 92.90 | 95.40 / 95.20 |

*Peaks select over saved validation checkpoints; five-step results also select over thresholds 0, 1.5 and 3.0 and forced depths 1–5. These are observed maxima, not averages or confidence intervals.*

Five steps win **five of nine** comparisons at 4 digits. At 8 digits, one step wins all **six graded-target comparisons**, while five steps win for binary exact-match at **two of three seeds**. The earlier suggestion that extra supervision does not help needs this qualification: the result depends on the target and seed.

Improvement within a model does not establish an advantage over one-step training. For 8-digit geometric-mean at seed 0, the best five-step checkpoint (threshold 3.0, epoch 180) rises from **78.12% at step 1 to 89.06% at step 5**, still below the one-step peak of **92.66%**. Later steps recover accuracy from a weaker first step. Averaging losses over executed steps reduces the direct weight of the first-step loss and could help explain this pattern. Testing that explanation requires an ablation.

Longer inputs do not necessarily require more dependent reasoning steps. The median longest run of carries is **2 in all three 8-digit validation sets**, although the sets also contain longer carry runs. This describes the data; it does not tell us how many steps the model needs.

## 4. What this adds, and what remains unresolved

These runs support the original early-stopping finding and show that forcing later steps usually hurts after complete collapse. The seeded code fixes the per-sample halt-loss mask, so changing the seed is not the only difference from the original experiments. Padding still contributes to the prediction-loss denominator and affects both training and the reported cross-entropy.

There are three seeds per configuration, with detailed measurements every ten epochs. Checkpoint counts are repeated observations within runs, not independent trials. The 8-digit model is larger (**1,277,037 versus 740,413 parameters**), so the comparison changes model size as well as task length. One supervision step still uses inner recursion. Neither its necessity nor an accuracy–compute advantage from adaptive halting has been established against matched baselines.
