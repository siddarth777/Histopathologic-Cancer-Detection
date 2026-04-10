import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from EDA.src.config import PLOT_DIR, SEED

def plot_class_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = df["label"].value_counts()

    axes[0].bar(["Negative (0)", "Positive (1)"], counts.values, color=["#378ADD", "#D85A30"])
    axes[0].set_title("Class Distribution — Count")
    axes[0].set_ylabel("Number of Samples")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 200, str(v), ha="center", fontsize=11)

    axes[1].pie(counts.values, labels=["Negative", "Positive"],
                autopct="%1.1f%%", colors=["#378ADD", "#D85A30"], startangle=90)
    axes[1].set_title("Class Distribution — Proportion")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}01_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_sample_images(df, load_full, load_crop):
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    for cls, label_name in [(0, "Negative"), (1, "Positive")]:
        subset = df[df["label"] == cls].sample(8, random_state=SEED)
        row_offset = 0 if cls == 0 else 2
        for i, (_, row) in enumerate(subset.iterrows()):
            axes[row_offset][i].imshow(load_full(row["filepath"]))
            axes[row_offset][i].axis("off")
            if i == 0:
                axes[row_offset][i].set_title(f"{label_name}\nFull (96x96)", fontsize=9)
                
            axes[row_offset + 1][i].imshow(load_crop(row["filepath"]))
            axes[row_offset + 1][i].axis("off")
            if i == 0:
                axes[row_offset + 1][i].set_title(f"{label_name}\nCrop (32x32)", fontsize=9)

    plt.suptitle("Sample Images — Full vs Center Crop by Class", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}02_sample_images.png", dpi=150, bbox_inches="tight")
    plt.close()
