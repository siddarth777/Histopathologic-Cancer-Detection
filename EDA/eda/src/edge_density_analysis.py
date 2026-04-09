import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from EDA.eda.src.morphological_analysis import edge_density
from EDA.eda.src.config import PLOT_DIR, REPORT_DIR
from EDA.eda.src.batch_utils import iter_dataframe_batches


def plot_edge_density_comparison(ed_df, track_name, out_dir=None):
    """Create a GLCM-style comparison plot: histogram + KDE + box, all in one figure."""
    if out_dir is None:
        out_dir = PLOT_DIR
    os.makedirs(out_dir, exist_ok=True)

    neg = ed_df[ed_df["label"] == 0]["edge_density"]
    pos = ed_df[ed_df["label"] == 1]["edge_density"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: overlaid histograms
    ax = axes[0]
    ax.hist(neg, bins=40, alpha=0.5, color="#378ADD", label="Negative (0)", density=True)
    ax.hist(pos, bins=40, alpha=0.5, color="#D85A30", label="Positive (1)", density=True)
    ax.set_xlabel("Edge Density")
    ax.set_ylabel("Density")
    ax.set_title(f"Histogram — {track_name}")
    ax.legend()

    # Panel 2: overlaid KDE
    ax = axes[1]
    sns.kdeplot(neg, ax=ax, color="#378ADD", label="Negative (0)", fill=True, alpha=0.3)
    sns.kdeplot(pos, ax=ax, color="#D85A30", label="Positive (1)", fill=True, alpha=0.3)
    ax.set_xlabel("Edge Density")
    ax.set_ylabel("Density")
    ax.set_title(f"KDE — {track_name}")
    ax.legend()

    # Panel 3: box plot
    ax = axes[2]
    ed_df_plot = ed_df.copy()
    ed_df_plot["Class"] = ed_df_plot["label"].map({0: "Negative (0)", 1: "Positive (1)"})
    sns.boxplot(
        data=ed_df_plot, x="Class", y="edge_density",
        hue="Class", palette={"Negative (0)": "#378ADD", "Positive (1)": "#D85A30"},
        ax=ax, legend=False,
    )
    ax.set_ylabel("Edge Density")
    ax.set_title(f"Box Plot — {track_name}")

    # Add mean annotations to box plot
    for i, cls_val in enumerate([0, 1]):
        vals = ed_df[ed_df["label"] == cls_val]["edge_density"]
        ax.annotate(
            f"μ={vals.mean():.4f}\nσ={vals.std():.4f}",
            xy=(i, vals.mean()), xytext=(0.3, 0),
            textcoords="offset fontsize", ha="left", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    plt.suptitle(f"14 — Edge Density: Positive vs Negative — {track_name}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"14_edge_density_comparison_{track_name}.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()


def generate_edge_density_report(df, loader, track_name, batch_size=1024):
    """Compute edge density for every image and produce distribution plots."""
    records = []

    for batch in iter_dataframe_batches(df, batch_size):
        for _, row in tqdm(
            batch.iterrows(), total=len(batch),
            desc=f"Edge Density {track_name}", leave=False,
        ):
            img = loader(row["filepath"])
            records.append({
                "label": int(row["label"]),
                "edge_density": edge_density(img),
            })

    ed_df = pd.DataFrame(records)
    ed_df.to_csv(f"{REPORT_DIR}edge_density_{track_name}.csv", index=False)

    # --- Summary statistics ---
    summary = (
        ed_df.groupby("label")["edge_density"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )
    summary["label"] = summary["label"].map({0: "Negative", 1: "Positive"})
    summary.to_csv(
        f"{REPORT_DIR}edge_density_summary_{track_name}.csv", index=False,
    )

    # --- Comparison plot (histogram + KDE + box) ---
    plot_edge_density_comparison(ed_df, track_name)

    return ed_df


def plot_from_csv(csv_path, track_name, out_dir):
    """Generate comparison plots from an existing edge density CSV."""
    ed_df = pd.read_csv(csv_path)
    plot_edge_density_comparison(ed_df, track_name, out_dir=out_dir)
    print(f"  Saved comparison plot for {track_name} → {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Edge density plots from CSV")
    parser.add_argument("--csv", required=True, help="Path to edge_density CSV")
    parser.add_argument("--track", required=True, help="Track name (e.g. Full96)")
    parser.add_argument("--out_dir", required=True, help="Output directory for plots")
    args = parser.parse_args()

    plot_from_csv(args.csv, args.track, args.out_dir)

