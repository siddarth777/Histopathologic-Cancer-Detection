# Histopathologic Cancer Detection

A deep learning project for detecting metastatic cancer in histopathologic scans of lymph node sections.

## Project Structure

```
Histopathologic-Cancer-Detection/
├── src/                    # Source code
├── src_lda/                # Distributed LDA / CNN+MLP experiment scripts
├── data/                  # Raw dataset
├── csv/                   # CSV files and logs
├── plots/                 # Training plots
├── logs/                  # Training logs
├── outputs/               # Model outputs
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

### 3. Run the Project
```bash
# Run exploratory data analysis
python -m src.eda

# Train all models
python -m src.training

# Generate summary report
python -m src.summary
```

### 4. Run the Distributed LDA Scripts
```bash
# Task 1: Raw Image -> CNN feature vector -> MLP -> class prediction
python -m src_lda.task1 --world-size 2

# Task 2: LDA on backbone features, then run the same model sweep
python -m src_lda.task2_lda --world-size 2

# Task 3: Regularized LDA (shrinkage=auto), then run the same model sweep
python -m src_lda.task3_reglda --world-size 2

# Combined runner
python -m src_lda.run_all --task all --world-size 2
```

## Test Commands

### Data Validation
```bash
# Check if data directory exists
python -c "import os; print(f'Data directory exists: {os.path.exists(\'data\') and os.path.isdir(\'data\')}')"

# Check if CSV files are generated
python -c "import os; print(f'CSV files exist: {len([f for f in os.listdir(\'csv\') if f.endswith(\'.csv\')])} files found')"

# Check if plot directory has images
python -c "import os; print(f'Plot files exist: {len([f for f in os.listdir(\'plots\') if f.endswith(\'.png\')])} files found')"

# Check if logs are generated
python -c "import os; print(f'Log files exist: {len([f for f in os.listdir(\'logs\') if f.endswith(\'.log\')])} files found')"
```

### Model Testing
```bash
# Test individual models
python -m src.training --model cnn
python -m src.training --model alexnet
python -m src.training --model resnet50
python -m src.training --model vgg16

# Distributed LDA tracks
python -m src_lda.task2_lda --model cnn --world-size 2
python -m src_lda.task3_reglda --model resnet50 --world-size 2

# Test with different configurations
python -m src.training --epochs 1 --batch_size 32
python -m src.training --seed 123
```

### Error Checking
```bash
# Check for syntax errors in all Python files
python -m py_compile src/*.py
python -m py_compile src/*/*.py

# Run flake8 for code style
flake8 src/ --max-line-length=120

# Run mypy for type checking
mypy src/ --ignore-missing-imports

# Run pytest for unit tests (if tests exist)
pytest tests/ -v
```

### Performance Testing
```bash
# Check GPU availability
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# Profile training time
python -m src.training --epochs 1 --profile
```

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

The src_lda package adds distributed-only variants for the same backbone sweep, plus a CNN + MLP pipeline and two transform tracks:
- Task 1: CNN feature vector + MLP head
- Task 2: LDA over backbone features
- Task 3: Regularized LDA over backbone features

## Output

Training outputs are saved in:
- `csv/`: Training metrics and logs
- `plots/`: Training curves and visualizations
- `logs/`: Detailed training logs
- `outputs/`: Model predictions and saved models

## License

This project is for educational and research purposes. Please refer to the Kaggle competition terms for usage rights of the dataset.