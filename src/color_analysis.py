import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from config import PLOT_DIR

def extract_color_features(filepaths, loader_fn):
    records = []
    for fp in tqdm(filepaths, desc="Extracting color features", leave=False):
        img_rgb = (loader_fn(fp) * 255).astype(np.uint8)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        records.append({
            "H": img_hsv[:,:,0].mean(), "S": img_hsv[:,:,1].mean(),
            "V": img_hsv[:,:,2].mean(),
            "L": img_lab[:,:,0].mean(), "A": img_lab[:,:,1].mean(),
            "B_lab": img_lab[:,:,2].mean(),
        })
    return pd.DataFrame(records)

def plot_color_analysis(df_sample, load_full_image, load_center_crop):
    for track_name, loader in [("Full96", load_full_image), ("Crop32", load_center_crop)]:
        dfs = []
        for cls in [0, 1]:
            subset = df_sample[df_sample["label"] == cls]
            feats = extract_color_features(subset["filepath"].tolist(), loader)
            feats["label"] = cls
            dfs.append(feats)
        color_df = pd.concat(dfs, ignore_index=True)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for i, col in enumerate(["H","S","V","L","A","B_lab"]):
            ax = axes[i // 3][i % 3]
            for cls, color in [(0, "#378ADD"), (1, "#D85A30")]:
                data = color_df[color_df["label"] == cls][col]
                sns.kdeplot(data, ax=ax, color=color, label=f"Class {cls}", fill=True, alpha=0.3)
            ax.set_title(f"{col} Distribution — {track_name}")
            ax.legend()
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}04_color_kde_{track_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 6))
        for cls, color in [(0, "#378ADD"), (1, "#D85A30")]:
            subset = color_df[color_df["label"] == cls]
            ax.scatter(subset["A"], subset["B_lab"], alpha=0.3, s=5, c=color,
                       label=f"{'Positive' if cls else 'Negative'}")
        ax.set_xlabel("A channel (LAB)")
        ax.set_ylabel("B channel (LAB)")
        ax.set_title(f"LAB Color Space — A vs B — {track_name}")
        ax.legend()
        plt.savefig(f"{PLOT_DIR}04_lab_scatter_{track_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
