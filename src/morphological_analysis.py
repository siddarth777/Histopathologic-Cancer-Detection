import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from skimage import color as skcolor
from config import PLOT_DIR

def laplacian_variance(img_array):
    gray = (skcolor.rgb2gray(img_array) * 255).astype(np.uint8)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def edge_density(img_array):
    gray = (skcolor.rgb2gray(img_array) * 255).astype(np.uint8)
    edges = cv2.Canny(gray, 50, 150)
    return edges.mean() / 255.0

def plot_morphological_analysis(df_sample, load_full_image, load_center_crop):
    for track_name, loader in [("Full96", load_full_image), ("Crop32", load_center_crop)]:
        records = []
        img_by_class = {0: [], 1: []}

        for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc=f"Morpho {track_name}", leave=False):
            img = loader(row["filepath"])
            cls = row["label"]
            img_by_class[cls].append(img)
            records.append({
                "label":       cls,
                "sharpness":   laplacian_variance(img),
                "edge_density":edge_density(img),
            })

        morph_df = pd.DataFrame(records)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for feat, ax in zip(["sharpness", "edge_density"], axes):
            for cls, color in [(0, "#378ADD"), (1, "#D85A30")]:
                sns.kdeplot(morph_df[morph_df["label"] == cls][feat], ax=ax,
                            color=color, label=f"Class {cls}", fill=True, alpha=0.3)
            ax.set_title(f"{feat.title()} — {track_name}")
            ax.legend()
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}06_morpho_dists_{track_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for cls in [0, 1]:
            mean_img = np.mean(img_by_class[cls], axis=0)
            axes[cls].imshow(np.clip(mean_img, 0, 1))
            axes[cls].set_title(f"Mean Image — {'Positive' if cls else 'Negative'}\n{track_name}")
            axes[cls].axis("off")
        plt.savefig(f"{PLOT_DIR}06_mean_images_{track_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for cls in [0, 1]:
            var_map = np.var(img_by_class[cls], axis=0).mean(axis=-1)
            im = axes[cls].imshow(var_map, cmap="hot")
            axes[cls].set_title(f"Pixel Variance — {'Positive' if cls else 'Negative'}\n{track_name}")
            axes[cls].axis("off")
            plt.colorbar(im, ax=axes[cls])
        plt.savefig(f"{PLOT_DIR}06_variance_maps_{track_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
