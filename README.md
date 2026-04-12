# Histopathologic Cancer Detection

This repository includes three experiment tracks for the Kaggle histopathologic cancer detection task:

- EDA analysis in `EDA/`
- deep learning LDA and Optuna experiments in `DL_exp/`
- classical ML experiments in `ML/`

## Repository Layout

```text
Histopathologic-Cancer-Detection/
├── main.py                  # Top-level dispatcher (EDA + DL)
├── EDA/
│   ├── src/                 # EDA pipeline implementation
│   └── outputs/             # Historical EDA outputs
├── DL_exp/
│   ├── src_lda/             # LDA / regLDA training workflows
│   ├── src_optuna/          # Optuna tuning workflows
│   └── scripts/             # Utilities (e.g., Grad-CAM)
├── ML/
│   ├── main.py              # Classical ML runner
│   ├── models/              # Per-model scripts
│   └── results/             # Benchmark CSVs
├── csv/                     # Additional CSV artifacts
└── README.md
```

## Folder Guides

- `EDA/README.md`
- `DL_exp/README.md`
- `ML/README.md`

## Data Layout

### DL and EDA image data

For the deep learning and EDA pipelines, place image data under a dataset root (default: `data/` at repository root):

```text
data/
├── train/
└── train_labels.csv
```

Notes:

- `train_labels.csv` must include at least `id` and `label` columns.
- `data/train/` must contain image files matching the `id` values (for example `<id>.tif`).
- A `data/test/` folder is not required for the current DL and EDA workflows in this repository.

### ML tabular data

Place your ML CSV files in these folders:

```text
ML/
├── datasets/
│   ├── full96_train.csv
│   ├── full96_test.csv
│   ├── crop32_train.csv
│   └── crop32_test.csv
└── results/
    ├── results_full96.csv
    └── results_crop32.csv
```

Use selected-features files from:

```text
EDA/outputs/reports/
├── selected_features_Full96.csv
└── selected_features_Crop32.csv
```

ML runner expects:

- A training CSV via `--train_path` (must include `label` and feature columns).
- A testing/holdout CSV via `--test_path` (same schema as training CSV).
- A selected-features CSV via `--selected_features_path` (first column = feature names).

You can store these CSVs anywhere and pass absolute or relative paths.

Recommended starter set for the commands below:

- `ML/datasets/full96_train.csv`
- `ML/datasets/full96_test.csv`
- `EDA/outputs/reports/selected_features_Full96.csv`
- `ML/results/results_full96.csv` (needed for `--run-best`)

## Top-Level Dispatcher

`main.py` controls EDA and DL workflows:

```bash
python main.py --eda
python main.py --dl-lda --data-dir data --out-dir outputs
python main.py --dl-optuna --data-dir data --out-dir outputs --n-trials 20
python main.py --eda --dl-lda --dl-optuna
```

Flags for `main.py`:

- `--eda`: run EDA pipeline.
- `--dl-lda`: run DL LDA + RegLDA pipeline (`DL_exp.src_lda` with `--task all`).
- `--dl-optuna`: run DL Optuna tuning pipeline (`DL_exp.src_optuna`).
- `--data-dir`: dataset root for DL modules. Must contain `train/` and `train_labels.csv`.
- `--out-dir`: output root for DL artifacts (checkpoints, logs, Optuna outputs).
- `--model`: model for DL Optuna (`cnn`, `alexnet`, `resnet50`, `vgg16`, or `all`).
- `--n-trials`: Optuna trials per model.
- `--n-jobs`: parallel Optuna jobs.
- `--timeout-minutes`: optional Optuna timeout; omit to disable timeout.
- `--epochs`: epochs per Optuna trial.
- `--eda-batch-size`: batch size used by EDA.

## Direct DL Commands

Run DL modules directly from repository root:

```bash
python -m DL_exp.src_lda --task all --data-dir data --out-dir outputs
python -m DL_exp.src_optuna --model all --data-dir data --out-dir outputs --n-trials 20 --n-jobs 2 --epochs 5
```

DL module flags:

- `DL_exp.src_lda`
- `--task`: `task2`, `task3`, or `all`.
- `--data-dir`: dataset root (`train/` + `train_labels.csv`).
- `--out-dir`: output directory for logs/checkpoints/plots.

- `DL_exp.src_optuna`
- `--model`: `cnn`, `alexnet`, `resnet50`, `vgg16`, or `all`.
- `--data-dir`: dataset root (`train/` + `train_labels.csv`).
- `--out-dir`: output directory.
- `--n-trials`: trials per model.
- `--n-jobs`: parallel trials.
- `--timeout-minutes`: optional timeout.
- `--epochs`: epochs per trial.

## ML Runner

The classical ML pipeline is run from `ML/`:

Verified working commands from repository root:

```bash
python -m ML.main --run-all --train_path ML/datasets/full96_train.csv --test_path ML/datasets/full96_test.csv --selected_features_path EDA/outputs/reports/selected_features_Full96.csv
python -m ML.main --run-best --results-csv ML/results/results_full96.csv --train_path ML/datasets/full96_train.csv --test_path ML/datasets/full96_test.csv --selected_features_path EDA/outputs/reports/selected_features_Full96.csv
```

What these commands do:

- First command (`--run-all`): runs all base ML models (and ensemble only if you also pass `--include-ensemble`).
- Second command (`--run-best`): reads `ML/results/results_full96.csv`, picks the row with highest `roc_auc`, then runs only that model.

ML flags:

- `--run-all`: execute logistic regression, random forest, naive Bayes, XGBoost, CatBoost.
- `--run-best`: execute only one model selected from `--results-csv`.
- `--train_path`: path to training feature CSV (must include `label`).
- `--test_path`: path to holdout/test feature CSV (must include `label`).
- `--selected_features_path`: path to selected-features CSV (feature names in first column).
- `--results-csv`: summary CSV used only with `--run-best`; must include `model`, `features`, `roc_auc`.
- `--n-trials`: Optuna trial count for single-model scripts.
- `--include-ensemble`: include `ML/run_ensemble_models.py` when using `--run-all`.
- `--n-trials-model`: Optuna trials for each ensemble base learner.
- `--n-trials-weights`: Optuna trials for ensemble weight tuning.
- `--ensemble-result-file`: output text file for ensemble metrics.

## Environment

This repository is configured with `pyproject.toml`. A typical setup is:

```bash
uv sync
```

## License

This project is for educational and research purposes. Follow Kaggle dataset terms for usage rights.
