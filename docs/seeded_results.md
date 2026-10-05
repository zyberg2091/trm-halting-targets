# Halting on 4-digit and 8-digit addition: three-seed results

This follow-up extends the [original study](technical_report.md) by repeating each configuration with seeds 0, 40 and 100 and adding 8-digit addition.

We compare three targets for the model’s stopping prediction:

- **Binary exact match:** 1 if the whole answer is correct, otherwise 0.
- **Soft-mean:** the fraction of answer tokens predicted correctly.
- **Geometric-mean:** the geometric mean of the probabilities assigned to the correct answer tokens.

Soft-mean and geometric-mean give partial credit and are called **graded targets**.

The experiments examine when the model stops, whether continuing improves its answer, and whether training with more supervision steps beats one-step training. One supervision step still includes the model’s internal recursion.

## 1. Experimental setup

Each run uses **45,000 training and 5,000 validation examples**. Within each seed and task size, all configurations share the same data, split and starting weights. Changing the seed changes both the data and initialization.

Each halt target is tested with five settings: one or sixteen supervision steps at threshold 0, and up to five steps at thresholds 0, 1.5 and 3.0. Across three targets, three seeds and two task sizes, this gives **90 runs**.

| Task | Model parameters | Training epochs |
|---|---:|---:|
| 4-digit addition | 740,413 | 100 |
| 8-digit addition | 1,277,037 | 200 |

The 8-digit model has larger input and output layers. Its recursive network is unchanged.

**Code correction.** The halt loss is now averaged only over samples that have not stopped. The original code averaged before applying this mask, so stopped samples could still influence the halt head. These experiments therefore include a code fix as well as fixed seeds. One issue remains: padding contributes zero prediction loss but still counts in its denominator.

**Measurements.** Every ten epochs, we record training stopping counts and validation accuracy. Whole-answer accuracy requires every non-padding position to be correct; token accuracy measures the fraction of individual positions correct.

We also evaluate with halting disabled, scoring answers after each step to check whether continuing helps. The **onset epoch** is the first logged epoch when at least half the training samples stop at step 1; **complete collapse** means all 45,000 do so during that epoch.

## 2. Graded targets lead to stopping while most answers are still wrong

With **soft-mean and geometric-mean**, first-step stopping becomes common before the model reliably produces correct whole answers. **Binary exact match** reaches the same stopping level later, at substantially higher accuracy.

The table shows when at least half the training samples first stop at step 1, and the model’s validation accuracy at that point. These runs allow up to five steps at threshold 0.

| Task | Halt target | Onset epoch | Whole-answer accuracy at onset |
|---|---|---:|---:|
| 4-digit | Soft-mean | 10 | 0.86–1.94% |
| 4-digit | Geometric-mean | 20 | 6.96–7.74% |
| 4-digit | Binary exact match | 40 | 54.32–93.94% |
| 8-digit | Soft-mean | 30 | 0.00% |
| 8-digit | Geometric-mean | 80–90 | 0.40–0.42% |
| 8-digit | Binary exact match | 100–140 | 39.32–48.16% |

*Ranges show the minimum and maximum across three seeds. Accuracy is measured at each run’s own onset epoch, using the model’s stopping decisions.*

The clearest case is 8-digit soft-mean: at least half the training samples stop after one step while **none of the validation answers is fully correct**, at all three seeds. Partial credit can therefore encourage stopping well before whole answers become reliable.

**A stricter threshold generally delays this behavior.** Complete first-step collapse is observed in 14 of 18 five-step runs at threshold 0, five at threshold 1.5, and none at threshold 3.0 during the logged training period. At threshold 3.0, some training examples still reach later steps at every logged epoch, preserving some feedback for those steps.

## 3. An extra step can make a good answer worse

In one 8-digit soft-mean run, whole-answer accuracy falls from **86.44% after step 1 to 70.54% after step 2** when we ignore the stopping decision and make the model continue.*

During that training epoch, all examples stopped after step 1. Training therefore provided feedback on their first answers but none on later answers. The shared network weights still changed, but training no longer checked whether another step improved the answer.

This pattern appears across the experiments. Combining **soft-mean, geometric-mean and binary exact match**, forcing a second step reduces token accuracy at **155 of 161 checkpoints** where every training sample stopped after step 1.

The results suggest that later answers can become unreliable without continued training feedback. They do not establish that losing this feedback caused the decline; that requires a controlled experiment.

*Example: seed 0, threshold 0, epoch 190, with up to five supervision steps.*

## 4. More supervision steps do not consistently beat one-step training

On 8-digit addition, **one-step training reaches higher peak whole-answer accuracy at all three seeds for soft-mean and geometric-mean**. Binary exact match gives mixed results: training with up to five steps wins at two of three seeds.

Extra steps can improve a model’s answer without making it better than a separately trained one-step model. The 8-digit geometric-mean run at seed 0 illustrates this:

| Model and evaluation depth | Whole-answer accuracy |
|---|---:|
| Five-step model, after step 1 | 78.12% |
| Same model, after step 5 | 89.06% |
| One-step model, at its best checkpoint | 92.66% |

The five-step model improves considerably as it continues, but remains below the one-step model. For 8-digit soft-mean, the best five-step result already comes from its first answer.

Both comparisons select the best logged checkpoint. The five-step side also searches three thresholds and five evaluation depths with halting disabled, giving it **15 times as many scores to choose from**. The graded-target one-step models remain ahead despite this broader search.

At 4 digits, results are mixed and close to 100%. Sixteen-step training, tested only at threshold 0, falls below the one-step peak in all nine 8-digit comparisons.

*The five-step example uses threshold 3.0 at epoch 180. Both of its scores come from the same checkpoint.*

## 5. Limitations

These results cover three seeds per configuration in a simplified recursive model on addition. Counts such as **155 of 161 checkpoints** describe repeated measurements within runs, not 161 independent experiments.

The 8-digit experiments change the input length, model size and training budget together, so they do not isolate the effect of task difficulty. Each stopping threshold also uses a separately trained model; changing only the threshold after training could give different results.

The reported peaks are selected validation scores, rather than results from an untouched test set. One supervision step still contains internal recursion, whose necessity has not been tested against a matched model without recursion. These experiments also do not establish an accuracy advantage for adaptive halting at equal computational cost.
