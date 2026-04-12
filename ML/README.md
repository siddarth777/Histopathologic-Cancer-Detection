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

## Run Commands

From repository root:

```bash
python -m ML.main --run-all --train_path ML/datasets/full96_train.csv --test_path ML/datasets/full96_test.csv --selected_features_path EDA/outputs/reports/selected_features_Full96.csv
python -m ML.main --run-best --results-csv ML/results/results_full96.csv --train_path ML/datasets/full96_train.csv --test_path ML/datasets/full96_test.csv --selected_features_path EDA/outputs/reports/selected_features_Full96.csv
```

Equivalent command from inside `ML/`:

```bash
python main.py --run-all --train_path datasets/full96_train.csv --test_path datasets/full96_test.csv --selected_features_path ../EDA/outputs/reports/selected_features_Full96.csv
```

## Main Flags

* `--run-all`: run logistic regression, random forest, naive Bayes, XGBoost, and CatBoost; optionally ensemble.
* `--run-best`: read `--results-csv`, select row with highest `roc_auc`, and run only that model.
* `--train_path`: training CSV path.
* `--test_path`: testing/holdout CSV path.
* `--selected_features_path`: CSV listing selected feature names in first column.
* `--results-csv`: input summary CSV used only with `--run-best`.
* `--n-trials`: Optuna trial count for single-model scripts (except naive Bayes).
* `--include-ensemble`: include `run_ensemble_models.py` when using `--run-all`.
* `--n-trials-model`: Optuna trials for ensemble base-model tuning.
* `--n-trials-weights`: Optuna trials for ensemble weight tuning.
* `--ensemble-result-file`: output text file name for ensemble metrics.

## Outputs

Produces per-model text logs and optional ensemble log output file.

