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
- `EDA/src/README.md`
- `DL_exp/README.md`
- `DL_exp/src_lda/README.md`
- `DL_exp/src_optuna/README.md`
- `ML/README.md`

## Data Layout

Expected dataset location at repository root:

```text
data/
├── train/
├── test/
└── train_labels.csv
```

## Top-Level Dispatcher

`main.py` controls EDA and DL workflows:

```bash
python main.py --eda
python main.py --dl-lda --data-dir data --out-dir outputs
python main.py --dl-optuna --data-dir data --out-dir outputs --n-trials 20
python main.py --eda --dl-lda --dl-optuna
```

Supported flags:

- `--eda`
- `--dl-lda`
- `--dl-optuna`
- `--data-dir`
- `--out-dir`
- `--model`
- `--n-trials`
- `--n-jobs`
- `--timeout-minutes`
- `--epochs`
- `--eda-batch-size`

## ML Runner

The classical ML pipeline is run from `ML/`:

```bash
python -m ML --run-all --train_path /path/train.csv --test_path /path/test.csv --selected_features_path /path/selected_features.csv
python -m ML --run-best --results-csv ML/results/results_full96.csv --train_path /path/train.csv --test_path /path/test.csv --selected_features_path /path/selected_features.csv
```

## Environment

This repository is configured with `pyproject.toml`. A typical setup is:

```bash
uv sync
```

## License

This project is for educational and research purposes. Follow Kaggle dataset terms for usage rights.
