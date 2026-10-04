# Halting targets in a Tiny Recursive Model

Addition experiments with a simplified TRM-style MLP. The halt head predicts one of three targets, all excluding padding:

- `softmean`: fraction of answer tokens correct.
- `geomean`: geometric mean of correct-token probabilities.
- `binary_em`: whether the whole answer is correct.

Read the [three-seed results](docs/seeded_results.md) for 4-digit and 8-digit findings; the [revised report on the original study](docs/technical_report.md) covers the model, losses and unseeded runs. The [original repository snapshot](archive/original_unseeded/readme.md) preserves the earlier notebooks, logs and report without changes.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/` | Modular code for the seeded follow-up experiments. |
| `run.ipynb` | Entry point for a new training run; open from the repository root. |
| `notebooks/4digit/epochs_100/seed_{0,40,100}/` | Seeded 4-digit notebooks, with saved outputs. |
| `notebooks/8digit/epochs_100/seed_0/` | Earlier 100-epoch snapshots of the seed-0 runs. |
| `notebooks/8digit/epochs_200/seed_{0,40,100}/` | Extended seeded 8-digit notebooks, with saved outputs. |
| `logs/` | Text exports matching the newer notebook hierarchy. |
| `docs/` | Current reports and the reorganization record. |
| `scripts/` | Saved-output extraction and verification. |
| `archive/original_unseeded/` | All 25 files from GitHub commit `0294880`, including 3 original notebooks, 15 run summaries, 3 CSV tables and the original report. |
| `CODE_PROVENANCE.md` | Mapping from notebook cells to source modules. |

Each experiment leaf contains `softmean`, `geomean` and `binary_em` notebooks or logs. Keep the original unseeded notebooks in the archive: the log extractor scans every notebook under the current `notebooks/` tree. See [version comparison and preservation details](docs/reorganization.md).

## What is here

| Digits | Epochs | Seeds | Notebooks / logs | Main runs |
| ---: | ---: | --- | ---: | ---: |
| 4 | 100 | 0, 40, 100 | 9 / 9 | 45 |
| 8 | 100 | 0 | 3 / 3 | 15 |
| 8 | 200 | 0, 40, 100 | 9 / 9 | 45 |

Each notebook has five `(maximum supervision steps, halt-logit threshold)` settings: `(5, 0)`, `(5, 1.5)`, `(5, 3.0)`, `(1, 0)`, `(16, 0)`.

Shared settings: 45,000 training / 5,000 validation examples; batch size 32; Adam, `1e-4`; halt-loss weight `c=0.1`; `T=3`, `n=6`. One outer supervision step still includes inner recursion.

All 105 main-run outputs are complete. The 15 older 8-digit seed-0 runs duplicate the corresponding 200-epoch runs' first 100 epochs; they are snapshots, not independent experiments.

## Results to look for

- At threshold 0, majority training-sample step-1 stopping begins with softmean, then geomean, then binary exact match, across task sizes and seeds.
- After every training sample stops at step 1, forced step 2 lowers validation token accuracy in 155/161 logged checkpoints. Shared weights still train; skipped steps have no direct loss.
- At 8 digits, one supervision step beats the best five-step result in all six graded-target comparisons. Binary exact match favors five steps at two of three seeds.

Accuracy comparisons use peak saved exact match with halting disabled. Five-step results also select the best threshold and forced depth. Overlapping 100-epoch snapshots are excluded.

## Using the files

Paths: `notebooks/<task>/epochs_<budget>/seed_<seed>/<target>.ipynb`, with matching `.txt` files under `logs/`. Logs contain five runs, diagnostics and previews, labeled by cell and settings.

From the repository root, with Python 3.10+:

```bash
python scripts/extract_logs.py
python scripts/extract_logs.py --check
```

Extraction uses the standard library; training needs PyTorch and Jupyter/Colab. Outside Colab, skip the final runtime cell. Notebook prose was edited; code and saved outputs are preserved.

Epoch markers appear every epoch; detailed metrics every ten. `%never` includes last-step halts. Dependency versions are unpinned; model checkpoints are absent. Earlier unseeded experiments are available in [the local archive](archive/original_unseeded/readme.md) as well as Git history. The original and seeded implementations differ, including a correction to the per-sample halt-loss mask; do not treat the archived runs as extra seeds of the newer implementation.

## Running the source package

The source package is already combined with this repository. Open `run.ipynb` from the repository root with PyTorch and Jupyter or Colab available. Set the options before importing `src`; restart the kernel if they change after import. This starts new training. Exploratory diagnostics remain in the experiment notebooks. See [code provenance](CODE_PROVENANCE.md) for the source mappings.

**AI assistance:** The AI-generated `scripts/extract_logs.py` copies saved notebook outputs without rerunning training. Reports and documentation were drafted and edited with AI assistance.
