import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import PLOT_DIR
from batch_utils import iter_dataframe_batches

def compute_channel_stats(filepaths, loader_fn):
    means, stds = [], []
    for fp in tqdm(filepaths, desc="Computing stats"):
        img = loader_fn(fp)
        means.append(img.mean(axis=(0, 1)))
        stds.append(img.std(axis=(0, 1)))
    return np.array(means), np.array(stds)

def plot_pixel_stats(df, load_full_image, load_center_crop, batch_size=1024):
    for track_name, loader in [("Full96", load_full_image), ("Crop32", load_center_crop)]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        channels = ["Red", "Green", "Blue"]
        colors = ["#E24B4A", "#639922", "#378ADD"]

        for cls in [0, 1]:
            subset = df[df["label"] == cls]
            mean_chunks = []
            std_chunks = []
            for batch in iter_dataframe_batches(subset, batch_size):
                subset_paths = batch["filepath"].tolist()
                means, stds = compute_channel_stats(subset_paths, loader)
                mean_chunks.append(means)
                std_chunks.append(stds)
            means = np.vstack(mean_chunks)
            stds = np.vstack(std_chunks)
            row = cls

            for ch_idx, (ch_name, color) in enumerate(zip(channels, colors)):
                axes[row][ch_idx].hist(means[:, ch_idx], bins=50, color=color, alpha=0.7,
                                       label=f"Class {cls}")
                axes[row][ch_idx].set_title(f"{'Positive' if cls else 'Negative'} — {ch_name} Mean\n({track_name})")
                axes[row][ch_idx].set_xlabel("Mean Pixel Intensity")
                axes[row][ch_idx].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}03_channel_histograms_{track_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for cls in [0, 1]:
            subset = df[df["label"] == cls]
            mean_chunks = []
            for batch in iter_dataframe_batches(subset, batch_size):
                subset_paths = batch["filepath"].tolist()
                means, _ = compute_channel_stats(subset_paths, loader)
                mean_chunks.append(means)
            means = np.vstack(mean_chunks)
            for ch_idx, ch_name in enumerate(channels):
                axes[ch_idx].boxplot(means[:, ch_idx], positions=[cls],
                                      patch_artist=True,
                                      boxprops=dict(facecolor=colors[ch_idx], alpha=0.6))
                axes[ch_idx].set_title(f"{ch_name} Channel Mean — {track_name}")
                axes[ch_idx].set_xticks([0, 1])
                axes[ch_idx].set_xticklabels(["Negative", "Positive"])
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}03_channel_boxplots_{track_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
