import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from skimage import color as skcolor
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from EDA.eda.src.config import PLOT_DIR, SEED
from EDA.eda.src.batch_utils import iter_dataframe_batches

def extract_glcm_features(img_array):
    gray = (skcolor.rgb2gray(img_array) * 255).astype(np.uint8)
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    return {
        "glcm_contrast":     graycoprops(glcm, "contrast").mean(),
        "glcm_dissimilarity":graycoprops(glcm, "dissimilarity").mean(),
        "glcm_homogeneity":  graycoprops(glcm, "homogeneity").mean(),
        "glcm_energy":       graycoprops(glcm, "energy").mean(),
        "glcm_correlation":  graycoprops(glcm, "correlation").mean(),
    }

def extract_lbp_features(img_array, n_points=24, radius=3):
    gray = skcolor.rgb2gray(img_array)
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2,
                           range=(0, n_points + 2), density=True)
    return hist

def plot_texture_analysis(df, load_full_image, load_center_crop, batch_size=1024):
    for track_name, loader in [("Full96", load_full_image), ("Crop32", load_center_crop)]:
        records = []
        for batch in iter_dataframe_batches(df, batch_size):
            for _, row in tqdm(batch.iterrows(), total=len(batch), desc=f"Texture {track_name}", leave=False):
                img = loader(row["filepath"])
                feats = extract_glcm_features(img)
                feats["label"] = row["label"]
                records.append(feats)
        tex_df = pd.DataFrame(records)

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        for i, col in enumerate(["glcm_contrast","glcm_dissimilarity",
                                  "glcm_homogeneity","glcm_energy","glcm_correlation"]):
            for cls, color in [(0, "#378ADD"), (1, "#D85A30")]:
                sns.kdeplot(tex_df[tex_df["label"] == cls][col], ax=axes[i],
                            color=color, label=f"Class {cls}", fill=True, alpha=0.3)
            axes[i].set_title(col.replace("glcm_", "").title() + f"\n{track_name}")
            axes[i].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}05_glcm_{track_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # LBP
        n_points_lbp = 24
        radius_lbp = 3
        n_bins_lbp = n_points_lbp + 2 

        lbp_hists_0, lbp_hists_1 = [], []
        for batch in iter_dataframe_batches(df, batch_size):
            for _, row in tqdm(batch.iterrows(), total=len(batch), desc=f"LBP {track_name}", leave=False):
                img = loader(row["filepath"])
                lbp_hist = extract_lbp_features(img, n_points=n_points_lbp, radius=radius_lbp)
                if row["label"] == 0:
                    lbp_hists_0.append(lbp_hist)
                else:
                    lbp_hists_1.append(lbp_hist)
        lbp_hists_0 = np.array(lbp_hists_0)
        lbp_hists_1 = np.array(lbp_hists_1)

        # Plot 1: Mean LBP histogram
        fig, ax = plt.subplots(figsize=(12, 5))
        bins_x = np.arange(n_bins_lbp)
        mean_h0 = lbp_hists_0.mean(axis=0)
        mean_h1 = lbp_hists_1.mean(axis=0)
        bar_w = 0.35
        ax.bar(bins_x - bar_w/2, mean_h0, bar_w, color="#378ADD", alpha=0.7, label="Negative (0)")
        ax.bar(bins_x + bar_w/2, mean_h1, bar_w, color="#D85A30", alpha=0.7, label="Positive (1)")
        ax.set_xlabel("LBP Bin (Uniform Pattern Index)")
        ax.set_ylabel("Mean Normalized Frequency")
        ax.set_title(f"Mean LBP Histogram by Class — {track_name}\n(n_points={n_points_lbp}, radius={radius_lbp})")
        ax.legend()
        ax.set_xticks(bins_x)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}05_lbp_histogram_{track_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Plot 2: LBP Entropy
        def lbp_entropy(hist):
            h = hist[hist > 0]
            return -np.sum(h * np.log2(h))

        entropy_0 = np.array([lbp_entropy(h) for h in lbp_hists_0])
        entropy_1 = np.array([lbp_entropy(h) for h in lbp_hists_1])

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.kdeplot(entropy_0, ax=ax, color="#378ADD", label="Negative (0)", fill=True, alpha=0.3)
        sns.kdeplot(entropy_1, ax=ax, color="#D85A30", label="Positive (1)", fill=True, alpha=0.3)
        ax.set_xlabel("LBP Histogram Entropy")
        ax.set_ylabel("Density")
        ax.set_title(f"LBP Texture Entropy Distribution by Class — {track_name}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}05_lbp_entropy_{track_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Plot 3: Samples
        fig, axes = plt.subplots(2, 8, figsize=(20, 5))
        for cls, row_idx in [(0, 0), (1, 1)]:
            subset = df[df["label"] == cls].sample(min(4, (df["label"] == cls).sum()), random_state=SEED)
            for j, (_, srow) in enumerate(subset.iterrows()):
                img = loader(srow["filepath"])
                gray = skcolor.rgb2gray(img)
                lbp_img = local_binary_pattern(gray, n_points_lbp, radius_lbp, method="uniform")
                axes[row_idx][j*2].imshow(img)
                axes[row_idx][j*2].axis("off")
                if j == 0:
                    axes[row_idx][j*2].set_title(f"{'Neg' if cls==0 else 'Pos'} Original", fontsize=8)
                axes[row_idx][j*2+1].imshow(lbp_img, cmap="viridis")
                axes[row_idx][j*2+1].axis("off")
                if j == 0:
                    axes[row_idx][j*2+1].set_title(f"{'Neg' if cls==0 else 'Pos'} LBP", fontsize=8)
        plt.suptitle(f"Sample LBP Transformations — {track_name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}05_lbp_samples_{track_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
