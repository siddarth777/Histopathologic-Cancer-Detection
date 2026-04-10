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
python -m DL_exp.src_optuna --model all --data-dir data --out-dir outputs --n-trials 20
```

Or through top-level dispatcher:

```bash
python main.py --dl-lda
python main.py --dl-optuna
```

## Inputs

- `data/train/`
- `data/test/`
- `data/train_labels.csv`

## Outputs

Primary artifacts are written under output directories passed via `--out-dir`, including:

- Optuna trials and best-params files
- model checkpoints
- logs and plots

## Related Docs

- `src_lda/README.md`
- `src_optuna/README.md`
- `../README.md`