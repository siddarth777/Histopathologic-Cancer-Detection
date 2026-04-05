import os

SEED = 42
TRAIN_DIR = 'train/'
TEST_DIR = 'data/test/'
LABELS_CSV = 'train_labels.csv'
PLOT_DIR = 'outputs/plots/'
MODEL_DIR = 'outputs/models/'

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
