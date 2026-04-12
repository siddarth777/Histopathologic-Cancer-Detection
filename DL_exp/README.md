# DL_exp

`DL_exp` contains deep learning experiments for this project.

## Subfolders

- `src_lda/`: LDA and regularized LDA training workflows
- `src_optuna/`: Optuna hyperparameter tuning workflows
- `scripts/`: utility scripts (for example Grad-CAM)
- `outputs/`: generated DL artifacts
- `plots/`: generated plots

## Run Commands

From repository root:

```bash
python -m DL_exp.src_lda --task all --data-dir data --out-dir outputs
python -m DL_exp.src_optuna --model all --data-dir data --out-dir outputs --n-trials 20 --n-jobs 2 --epochs 5
```

Or through top-level dispatcher:

```bash
python main.py --dl-lda
python main.py --dl-optuna
```

## Inputs

- `data/train/`
- `data/train_labels.csv`

`data/test/` is not required for the current DL workflows in this repository.

`train_labels.csv` should contain at least `id` and `label` columns.

## Flags

`python -m DL_exp.src_lda`

- `--task`: `task2`, `task3`, or `all`.
- `--data-dir`: dataset root containing `train/` and `train_labels.csv`.
- `--out-dir`: output root for checkpoints/logs/plots.

`python -m DL_exp.src_optuna`

- `--model`: `cnn`, `alexnet`, `resnet50`, `vgg16`, or `all`.
- `--data-dir`: dataset root containing `train/` and `train_labels.csv`.
- `--out-dir`: output root for Optuna artifacts.
- `--n-trials`: number of trials per model.
- `--n-jobs`: parallel Optuna jobs.
- `--timeout-minutes`: optional timeout in minutes.
- `--epochs`: epochs per trial.

## Outputs

Primary artifacts are written under output directories passed via `--out-dir`, including:

- Optuna trials and best-params files
- model checkpoints
- logs and plots

## Related Docs

- `../README.md`