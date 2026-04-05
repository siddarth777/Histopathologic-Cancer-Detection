import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from config import TRAIN_DIR, LABELS_CSV, SEED

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

def load_full_image(filepath, size=(96, 96)):
    img = Image.open(filepath).convert("RGB").resize(size)
    return np.array(img) / 255.0

def load_center_crop(filepath, crop_size=32):
    img = np.array(Image.open(filepath).convert("RGB"))
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    half = crop_size // 2
    crop = img[cy - half:cy + half, cx - half:cx + half]
    return crop / 255.0

def load_data_and_sample(sample_n=5000):
    df = pd.read_csv(LABELS_CSV)
    df["filepath"] = df["id"].apply(lambda x: os.path.join(TRAIN_DIR, x + ".tif"))
    
    df_sample = pd.concat([
        df[df["label"] == 0].sample(min(sample_n // 2, (df["label"] == 0).sum()), random_state=SEED),
        df[df["label"] == 1].sample(min(sample_n // 2, (df["label"] == 1).sum()), random_state=SEED),
    ]).reset_index(drop=True)
    
    return df, df_sample
