# Histopathologic Cancer Detection

A deep learning project for detecting metastatic cancer in histopathologic scans of lymph node sections.

## Project Structure

```
Histopathologic-Cancer-Detection/
├── src/                    # Source code (models, data, utilities)
├── src_lda/                # LDA / CNN+MLP experiment scripts (single-GPU)
├── src_optuna/             # Hyperparameter tuning with Optuna
├── csv/                    # CSV files and training logs
├── plots/                  # Training plots and visualizations
├── scripts/                # Utility scripts (e.g., gradcam_cnn.py)
├── outputs/                # Model outputs and results
└── README.md
```

## Quick Start

### 1. Install Dependencies
```bash
source .venv/bin/activate
uv pip install torch torchvision pandas scikit-learn scipy matplotlib seaborn tqdm pillow
```

### 2. Download Data from Kaggle
```bash
# Install kaggle hub
pip install kagglehub

# Login to Kaggle
kagglehub login

# Download the dataset
kagglehub download dataset histopathologic-cancer-detection

# Extract the data
unzip histopathologic-cancer-detection.zip -d data/
```

### 3. Run LDA Experiments
```bash
# Task 2: LDA on backbone features (all model variants)
python -m src_lda.task2_lda --data-dir data --out-dir outputs

# Task 3: Regularized LDA on backbone features (all model variants)
python -m src_lda.task3_reglda --data-dir data --out-dir outputs

# Run both tasks sequentially
python -m src_lda.run_all --task all --data-dir data --out-dir outputs
```

### 4. Hyperparameter Tuning with Optuna
```bash
# Tune all models (default: 20 trials, 2 jobs in parallel)
python -m src_optuna --data-dir data --out-dir outputs --n-trials 20

# Tune a single model
python -m src_optuna --model resnet50 --data-dir data --out-dir outputs --n-trials 20

# Custom configuration (50 trials, timeout 120 mins, 3 epochs per trial)
python -m src_optuna --n-trials 50 --timeout-minutes 120 --epochs 3 --data-dir data --out-dir outputs

# Recover best params from completed trials
python src_optuna/rebuild_best_params.py --optuna-dir outputs/optuna --force
```

**Optuna Output Structure:**
```
outputs/optuna/<model>/
├── trials.csv              # All trials with metrics
├── top_trials.csv          # Top 5 trials
├── best_params.json        # Best hyperparameters + best metrics
└── trial_XXXX/             # Individual trial folders
    ├── log.csv             # Trial training logs
    └── <model>_best.pth    # Best checkpoint for that trial
```

## Test Commands

### Environment Validation
```bash
# Check GPU availability
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# Check output directories
python -c "import os; dirs = ['csv', 'plots', 'outputs']; print('\n'.join([f'{d}: {os.path.exists(d)}' for d in dirs]))"
```

### Quick LDA Task Test
```bash
# Test Task 2 with a single model
python -m src_lda.task2_lda --model cnn --data-dir data --out-dir outputs/test

# Test Task 3 with a single model
python -m src_lda.task3_reglda --model resnet50 --data-dir data --out-dir outputs/test
```

### Quick Optuna Test
```bash
# Test single-model tuning with 2 trials
python -m src_optuna --model cnn --n-trials 2 --data-dir data --out-dir outputs/test
```

### Code Quality
```bash
# Check for syntax errors
python -m py_compile src_lda/*.py
python -m py_compile src_optuna/*.py

# Check with flake8 (if installed)
flake8 src_lda/ --max-line-length=120
flake8 src_optuna/ --max-line-length=120
## Data Structure

The dataset contains:
- `train/`: Training images (220,025 images)
- `test/`: Test images (57,458 images)
- `train_labels.csv`: Labels for training images

Each image is a 96x96 pixel RGB image with a binary label indicating presence of metastatic tissue.

## Models

This project trains and evaluates the following models:
- CNN (Custom Convolutional Neural Network)
- AlexNet
- ResNet50
- VGG16

The src_lda package provides two LDA-based training pipelines for single-GPU setups:
- **Task 2: LDA** — Extract backbone features → LDA projection → train MLP head
- **Task 3: Regularized LDA** — Same as Task 2 but with automatic shrinkage estimation

Both tasks sweep across all models: CNN, AlexNet, ResNet50, and VGG16.

The src_optuna package provides Optuna-based hyperparameter tuning for all models, searching over:
- Learning rate (log-scale: 1e-5 to 1e-2)
- Weight decay (log-scale: 1e-6 to 1e-2)
- Optimizer (AdamW or SGD with conditional momentum)

Results include trial metrics, top-5 trials ranking, and best hyperparameters with validation performance.

## Output Structure

Training outputs are organized in `outputs/`:
- `outputs/optuna/<model>/` — Optuna tuning results (trials, best params, checkpoints)
- `outputs/eda/` — Exploratory data analysis plots
- `outputs/` — Task 2 and Task 3 logs and checkpoints (by default)

CSV logs are saved in `csv/` and plots in `plots/`

## License

This project is for educational and research purposes. Please refer to the Kaggle competition terms for usage rights of the dataset.