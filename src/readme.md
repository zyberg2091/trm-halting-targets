# Experiment modules

Start from [`run.ipynb`](../run.ipynb) in the repository root. It sets the options read by `config.py` and calls the training function. Importing the training module also builds the dataset.

| file | role |
|---|---|
| `config.py` | Sets the seed, operand digit count, epoch count, and halting target. |
| `model.py` | Defines `TinyRModel`. |
| `data.py` | Generates addition problems and carry-chain labels; defines `AdditionDataset`, the train/validation split, data loaders, and sequence dimensions. |
| `targets/softmean.py` | Defines `Q_LOSS` using the fraction of non-padding digits predicted correctly. |
| `targets/geomean.py` | Defines `Q_LOSS` using the geometric mean of the probabilities assigned to the correct non-padding tokens. |
| `targets/binary_em.py` | Defines `Q_LOSS` with target 1 when all non-padding digits are correct and 0 otherwise. |
| `targets/__init__.py` | Exposes the target selected by `config.TARGET`. |
| `train.py` | Defines `model_training_and_validation_with_mask`, including training with per-sample halting, validation with halting, and forced-depth validation without halting. |
| `data_check.py` | Prints counts for selected subsets of the generated dataset. |

The source notebooks are at `notebooks/<task>/epochs_<budget>/seed_<seed>/<target>.ipynb`. See [code provenance](../CODE_PROVENANCE.md) for the cell numbers and packaging changes: six configuration substitutions, imports, and the use of variable-length dataset and training cells. That document also records the AI assistance used in packaging and editing the comments and documentation.

Install PyTorch and use Jupyter or Colab. The original dependency versions were not recorded. Choose the settings before importing `src`; restart the kernel if you change them later.

The source package is already integrated. Keep `src/` and `run.ipynb` at the repository root so the `src.*` imports resolve. The original unseeded code and outputs are preserved separately under `archive/original_unseeded/`.
