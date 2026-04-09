# Histopathologic Cancer Detection

Deep learning pipelines for histopathologic cancer detection, organized around three top-level workflows:

- EDA exploration
- LDA-based model training
- Optuna-based hyperparameter tuning

## Repository Layout

```text
Histopathologic-Cancer-Detection/
├── main.py              # Top-level pipeline dispatcher
├── EDA/                 # Exploratory data analysis package
└── DL_exp/              # LDA and Optuna training packages
```

## Requirements

Typical Python packages used by the project include:

- torch
- torchvision
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- tqdm
- pillow
- optuna

Install them with your preferred environment manager.

## Data

The project expects the Kaggle histopathologic cancer detection dataset in the project root:

```text
Histopathologic-Cancer-Detection/
└── data/
    ├── train/
    ├── test/
    └── train_labels.csv
```

Both the EDA pipeline and the DL pipelines read from this same `data/` directory.

### Download Methods

#### 1. Kaggle CLI

```bash
# Authenticate first if needed
kaggle competitions download -c histopathologic-cancer-detection

# Extract to the project-root data directory
mkdir -p data
unzip histopathologic-cancer-detection.zip -d data/
```

#### 2. KaggleHub

```python
import kagglehub

kagglehub.competition_download('histopathologic-cancer-detection')
```

If KaggleHub downloads to a cache location, copy or move the extracted dataset so your project has:

```text
data/train/
data/test/
data/train_labels.csv
```

## Top-Level Runner

Use `main.py` to launch the desired pipeline(s):

```bash
python main.py --eda
python main.py --dl-lda
python main.py --dl-optuna
python main.py --eda --dl-lda --dl-optuna
```

### `main.py` Flags

- `--eda` runs the EDA pipeline
- `--dl-lda` runs the LDA pipeline
- `--dl-optuna` runs the Optuna pipeline
- `--data-dir` sets the shared dataset directory for DL pipelines
- `--out-dir` sets the output directory for DL pipelines
- `--model` chooses a single Optuna model or `all`
- `--n-trials` sets Optuna trial count per model
- `--n-jobs` sets Optuna parallel jobs
- `--timeout-minutes` sets an Optuna timeout
- `--epochs` sets epochs per Optuna trial
- `--eda-batch-size` sets the EDA batch size

## EDA

The EDA pipeline runs the package entrypoint under `EDA/eda/src/eda.py`, reads from `data/`, and generates dataset summaries and visualizations under `outputs/eda/`.

Example:

```bash
python main.py --eda
```

## LDA Pipeline

The LDA pipeline runs the single-GPU training workflows in `DL_exp/src_lda/`.

It executes:

- Task 2: LDA projection on backbone features
- Task 3: Regularized LDA projection on backbone features

Example:

```bash
python main.py --dl-lda --data-dir data --out-dir outputs
```

You can also run the package directly:

```bash
python -m DL_exp.src_lda --task all --data-dir data --out-dir outputs
```

## Optuna Pipeline

The Optuna pipeline tunes model hyperparameters and writes trial artifacts under `outputs/optuna/`.

Example:

```bash
python main.py --dl-optuna --data-dir data --out-dir outputs --n-trials 20
```

You can also run the package directly:

```bash
python -m DL_exp.src_optuna --model all --data-dir data --out-dir outputs --n-trials 20
```

### Optuna Outputs

For each model, Optuna writes:

```text
outputs/optuna/<model>/
├── trials.csv
├── top_trials.csv
├── best_params.json
└── trial_XXXX/
    ├── log.csv
    └── <model>_best.pth
```

## Outputs

Typical generated artifacts include:

- `outputs/eda/` for EDA results
- `outputs/optuna/` for Optuna tuning outputs
- model checkpoints and CSV logs from the DL pipelines

## Notes

- `--dl-lda` and `--dl-optuna` run the model sweeps sequentially.
- `--dl-optuna` defaults to tuning all models when `--model` is not provided.
- `--eda-batch-size` only affects the EDA pipeline.

## License

This project is for educational and research purposes. Refer to the Kaggle competition terms for dataset usage rights.
