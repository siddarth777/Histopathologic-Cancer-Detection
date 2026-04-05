# Histopathologic Cancer Detection

A deep learning project for detecting metastatic cancer in histopathologic scans of lymph node sections.

## Project Structure

```
Histopathologic-Cancer-Detection/
├── src/                    # Source code
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
pip install -r requirements.txt
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

## Output

Training outputs are saved in:
- `csv/`: Training metrics and logs
- `plots/`: Training curves and visualizations
- `logs/`: Detailed training logs
- `outputs/`: Model predictions and saved models

## License

This project is for educational and research purposes. Please refer to the Kaggle competition terms for usage rights of the dataset.