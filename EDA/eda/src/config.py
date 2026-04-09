import os
from pathlib import Path

SEED = 42

BASE_DIR = os.path.join(Path(__file__).resolve().parent.parent, "data")
TRAIN_DIR = os.path.join(BASE_DIR, "train") + os.sep
TEST_DIR = os.path.join(BASE_DIR, "data", "test") + os.sep
LABELS_CSV = os.path.join(BASE_DIR, "train_labels.csv")
PLOT_DIR = os.path.join(BASE_DIR, "outputs", "plots") + os.sep
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models") + os.sep
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports") + os.sep

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
