# Code provenance

This package moves the experiment code out of the saved notebooks. The main changes are imports between modules, one place for run settings, and a switch between the three halting targets. The source cells and substitutions are listed below.

Packaging and documentation were prepared with AI assistance. Comments were edited as well; these files are not verbatim notebook exports.

## Using the combined repository

Open `run.ipynb` from the repository root in Jupyter or Colab with PyTorch installed.

The two uploaded packages have already been combined: `src/`, `run.ipynb`, and `CODE_PROVENANCE.md` sit beside `notebooks/`, `logs/`, and `docs/`, without the outer `src_package/` wrapper. The earlier unseeded repository is preserved under `archive/original_unseeded/`.

Set the run options before importing the modules. Changing them afterwards requires a kernel restart: importing `src.train` also creates the dataset.

The notebook paths in this document resolve from the repository root and refer to the seeded follow-up notebooks and saved logs. Reorganization changed documentation and paths only; the uploaded Python code and saved experiment notebooks and logs are unchanged. No trained checkpoints or pinned training environment are included. A fresh run may therefore differ from the saved results.

## Notebook sources

| module | notebook | cell | contents | non-blank lines |
|---|---|---:|---|---:|
| `src/model.py` | `notebooks/4digit/epochs_100/seed_0/softmean.ipynb` | 6 | TinyRModel | 28 |
| `src/data.py` | `notebooks/8digit/epochs_100/seed_0/softmean.ipynb` | 11 | generate_pair + operand generation | 15 |
| `src/data.py` | `notebooks/8digit/epochs_100/seed_0/softmean.ipynb` | 12 | dataset build with carry chains | 35 |
| `src/data.py` | `notebooks/8digit/epochs_100/seed_0/softmean.ipynb` | 13 | AdditionDataset, split, loaders | 43 |
| `src/train.py` | `notebooks/8digit/epochs_100/seed_0/softmean.ipynb` | 20 | model_training_and_validation_with_mask | 189 |
| `src/targets/softmean.py` | `notebooks/4digit/epochs_100/seed_0/softmean.ipynb` | 17 | Q_LOSS soft mean | 7 |
| `src/targets/geomean.py` | `notebooks/4digit/epochs_100/seed_0/geomean.ipynb` | 17 | Q_LOSS geometric mean | 8 |
| `src/targets/binary_em.py` | `notebooks/4digit/epochs_100/seed_0/binary_em.ipynb` | 16 | Q_LOSS binary exact match | 6 |
| `src/data_check.py` | `notebooks/4digit/epochs_100/seed_0/softmean.ipynb` | 23 | dataset sanity print | 10 |

## Package additions

- `src/config.py` collects the seed, digit count, epoch count, and halting target. The notebooks set these directly in their cells.
- `src/targets/__init__.py` selects a `Q_LOSS` function; each notebook had just one target definition.
- `src/__init__.py` is empty. It marks the package.
- Imports replace the shared notebook namespace.

## Configuration substitutions

| module | line in the notebook | line in `src/` | times |
|---|---|---|---:|
| `src/data.py` | `random.seed(0)` | `random.seed(SEED)` | 1 |
| `src/data.py` | `torch.manual_seed(0)` | `torch.manual_seed(SEED)` | 2 |
| `src/data.py` | `num_digits = 8` | `num_digits = NUM_DIGITS` | 1 |
| `src/train.py` | `  torch.manual_seed(0)` | `torch.manual_seed(SEED)` | 1 |
| `src/train.py` | `  epochs = 100` | `epochs = EPOCHS` | 1 |

These are the six literal-value substitutions. The defaults are `SEED=0`, `NUM_DIGITS=4`, and `EPOCHS=100`. Along with the imports and source-version choices below, they account for the changes needed to package the code.

## Use of the 8-digit dataset and training cells

The dataset and training modules use the 8-digit cells because those cells calculate the dimensions from `num_digits`. Input length is `2 * num_digits + 1`; output length is `num_digits + 1`. The 4-digit notebooks hard-code 9 and 5. The same difference appears in the carry buckets: `range(n_chain)` in the 8-digit cells, `range(5)` in the 4-digit cells. The 8-digit cells also print each bucket's sample count.

With `NUM_DIGITS = 4`, `data.py` and `train.py` give `inp_seq_len = 9`, `out_seq_len = 5`, and `n_chain = 5`, matching the 4-digit notebooks. The forced-depth header still prints bucket sizes.

## Separate diagnostic code

`diagnostic_per_sup_step` was left in the notebooks. It trains without halting and reports each supervision step separately. It was an exploratory check, outside the five main configurations, so it is not part of `src/`.
