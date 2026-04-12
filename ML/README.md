# ML

`ML` is the classical machine learning pipeline for tabular feature experiments.

## Folder Layout

```text
ML/
├── main.py
├── run_ensemble_models.py
├── datasets/              # Place train/test feature CSVs here
├── models/
│   ├── logistic_regression.py
│   ├── random_forest.py
│   ├── naive_bayes.py
│   ├── xgboost_model.py
│   └── catboost_model.py
└── results/
    ├── results_full96.csv
    └── results_crop32.csv
```

## Required Inputs

* `--train_path`
* `--test_path`
* `--selected_features_path`

Input expectations for `ML.main`:

* `--train_path` and `--test_path` are tabular CSV files with the same feature columns.
* Both CSVs should include `label`; `id` is optional and ignored during feature selection.
* The selected-features CSV should contain feature names in its first column.
* For `--run-best`, `--results-csv` must include columns: `model`, `features`, `roc_auc`.

## Where To Place Files

Put your existing ML feature CSV files in:

* `ML/datasets/full96_train.csv`
* `ML/datasets/full96_test.csv`
* `ML/datasets/crop32_train.csv`
* `ML/datasets/crop32_test.csv`

Put selected-features files in:

* `EDA/outputs/reports/selected_features_Full96.csv`
* `EDA/outputs/reports/selected_features_Crop32.csv`

Put results summary files (used by `--run-best`) in:

* `ML/results/results_full96.csv`
* `ML/results/results_crop32.csv`

Quick checklist for the Full96 commands below:

* `ML/datasets/full96_train.csv`
* `ML/datasets/full96_test.csv`
* `EDA/outputs/reports/selected_features_Full96.csv`
* `ML/results/results_full96.csv` (only required for `--run-best`)

## Run Commands

From repository root:

Verified working commands:

```bash
python -m ML.main --run-all --train_path ML/datasets/full96_train.csv --test_path ML/datasets/full96_test.csv --selected_features_path EDA/outputs/reports/selected_features_Full96.csv
python -m ML.main --run-best --results-csv ML/results/results_full96.csv --train_path ML/datasets/full96_train.csv --test_path ML/datasets/full96_test.csv --selected_features_path EDA/outputs/reports/selected_features_Full96.csv
```

Behavior:

* Command 1 (`--run-all`): runs all base models sequentially.
* Command 2 (`--run-best`): selects one best model from `ML/results/results_full96.csv` using max `roc_auc`, then runs only that model.

Equivalent command from inside `ML/`:

```bash
python main.py --run-all --train_path datasets/full96_train.csv --test_path datasets/full96_test.csv --selected_features_path ../EDA/outputs/reports/selected_features_Full96.csv
```

## Main Flags

* `--run-all`: run logistic regression, random forest, naive Bayes, XGBoost, CatBoost.
* `--run-best`: use `--results-csv` to choose a single best model by highest `roc_auc`.
* `--train_path`: path to the training feature CSV.
* `--test_path`: path to the holdout/test feature CSV.
* `--selected_features_path`: path to selected-features CSV (first column is feature name list).
* `--results-csv`: model-comparison CSV read by `--run-best`.
* `--n-trials`: Optuna trial count for per-model tuning scripts (except naive Bayes).
* `--include-ensemble`: when using `--run-all`, also run `run_ensemble_models.py`.
* `--n-trials-model`: ensemble base-model tuning trials.
* `--n-trials-weights`: ensemble weight-search tuning trials.
* `--ensemble-result-file`: output text file for ensemble metrics.

## Outputs

Produces per-model text logs and optional ensemble log output file.

