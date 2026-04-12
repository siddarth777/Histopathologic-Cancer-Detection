import os
from pathlib import Path

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

TRAIN_DIR = os.path.join(DATA_DIR, "train") + os.sep
TEST_DIR = os.path.join(DATA_DIR, "test") + os.sep
LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")
PLOT_DIR = os.path.join(OUTPUT_DIR, "eda") + os.sep
MODEL_DIR = os.path.join(OUTPUT_DIR, "eda", "models") + os.sep
REPORT_DIR = os.path.join(OUTPUT_DIR, "eda", "reports") + os.sep

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
