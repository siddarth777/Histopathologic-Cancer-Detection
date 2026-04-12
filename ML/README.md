# ML

`ML` is the classical machine learning pipeline for tabular feature experiments.

## Folder Layout

```text
ML/
├── __main__.py
├── main.py
├── run_ensemble_models.py
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

The selected-features CSV should contain feature names in its first column.

## Generating `results_full96.csv`

To obtain `results_full96.csv`, either:

1. Run the `feature_extraction.py` script prior to using this pipeline.
2. Download the precomputed file from the provided [Google Drive link](https://drive.google.com/drive/folders/15_sl_u_LarPHjbz1vFc27FBD6rL23tYd?usp=sharing).

## Run Commands

From repository root:

```bash
python -m ML --run-all --train_path /path/train.csv --test_path /path/test.csv --selected_features_path /path/selected_features.csv
python -m ML --run-best --results-csv ML/results/results_full96.csv --train_path /path/train.csv --test_path /path/test.csv --selected_features_path /path/selected_features.csv
```

Equivalent command from inside `ML/`:

```bash
python main.py --run-all --train_path /path/train.csv --test_path /path/test.csv --selected_features_path /path/selected_features.csv
```

## Main Flags

* `--run-all`
* `--run-best`
* `--train_path`
* `--test_path`
* `--selected_features_path`
* `--results-csv`
* `--n-trials`
* `--include-ensemble`
* `--n-trials-model`
* `--n-trials-weights`
* `--ensemble-result-file`

## Outputs

Produces per-model text logs and optional ensemble log output file.

