# Histopathologic Cancer Detection Experiments

This repository contains a classical ML experiment pipeline for histopathologic cancer detection.

## Project Layout

```text
Histopathologic-Cancer-Detection/
├── main.py                     # Unified experiment runner
├── run_ensemble_models.py      # Ensemble experiments (RF/XGB/CAT + weighted/stacking)
├── models/
│   ├── logistic_regression.py
│   ├── random_forest.py
│   ├── naive_bayes.py
│   ├── xgboost_model.py
│   └── catboost_model.py
└── results/
  ├── results_full96.csv      # Existing benchmark results (used by --run-best)
  └── results_crop32.csv      # Existing benchmark results (used by --run-best)
```

## Data Requirements

The scripts expect pre-engineered tabular CSV files with:

- a target column named `label`
- an optional `id` column
- feature columns

You need:

- train CSV (`--train_path`)
- test CSV (`--test_path`)
- selected-features CSV (`--selected_features_path`)

The selected-features CSV should contain feature names in the first column.

## Install Dependencies

```bash
pip install pandas numpy scikit-learn optuna tqdm xgboost catboost
```

## Unified Runner

Use `main.py` as the single entrypoint.

### 1. Run all base experiments

```bash
python main.py \
  --run-all \
  --train_path /path/to/train.csv \
  --test_path /path/to/test.csv \
  --selected_features_path /path/to/selected_features.csv \
  --n-trials 10
```

This runs:

- logistic regression
- random forest
- naive bayes
- xgboost
- catboost

### 2. Run all experiments plus ensemble

```bash
python main.py \
  --run-all \
  --include-ensemble \
  --train_path /path/to/train.csv \
  --test_path /path/to/test.csv \
  --selected_features_path /path/to/selected_features.csv \
  --n-trials 10 \
  --n-trials-model 10 \
  --n-trials-weights 10
```

### 3. Run only the best model by ROC-AUC

```bash
python main.py \
  --run-best \
  --results-csv results/results_full96.csv \
  --train_path /path/to/train.csv \
  --test_path /path/to/test.csv \
  --selected_features_path /path/to/selected_features.csv \
  --n-trials 10
```

`--run-best` reads the `model`, `features`, and `roc_auc` columns from the results CSV and runs the top model.

If the best row is an `ensemble_*` model (for example `ensemble_stacking_lr`), the runner dispatches to `run_ensemble_models.py`.

## Main Arguments

- `--run-all`: run all base model experiments
- `--run-best`: run only best model according to `roc_auc` from `--results-csv`
- `--train_path`: path to training CSV (required)
- `--test_path`: path to test CSV (required)
- `--selected_features_path`: path to selected-features CSV (required)
- `--results-csv`: benchmark CSV used by `--run-best` (default: `results/results_full96.csv`)
- `--n-trials`: Optuna trials for model scripts that tune hyperparameters
- `--include-ensemble`: include `run_ensemble_models.py` when using `--run-all`
- `--n-trials-model`: model tuning trials for ensemble script
- `--n-trials-weights`: weight tuning trials for ensemble script
- `--ensemble-result-file`: output log file for ensemble run

## Outputs

- Per-model text logs such as:
  - `results_logistic_regression.txt`
  - `results_random_forest.txt`
  - `results_xgboost.txt`
- Ensemble log file:
  - `results_ensemble.txt` (or custom path via `--ensemble-result-file`)

## Notes

- `xgboost` and `catboost` must be installed to run those models.
- `run_ensemble_models.py` writes/reads `checkpoint.json` to resume ensemble modes (`all`, `selected`).
- When running `run_ensemble_models.py` directly, always pass `--train_path`, `--test_path`, and `--selected_features_path` explicitly.
