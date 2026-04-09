import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from EDA.eda.src.morphological_analysis import edge_density
from EDA.eda.src.config import PLOT_DIR, REPORT_DIR
from EDA.eda.src.batch_utils import iter_dataframe_batches


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

    # --- KDE distribution plot (class-wise) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, color, lbl in [(0, "#378ADD", "Negative"), (1, "#D85A30", "Positive")]:
        sns.kdeplot(
            ed_df[ed_df["label"] == cls]["edge_density"],
            ax=ax, color=color, label=lbl, fill=True, alpha=0.3,
        )
    ax.set_xlabel("Edge Density")
    ax.set_ylabel("Density")
    ax.set_title(f"14 — Edge Density Distribution — {track_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        f"{PLOT_DIR}14_edge_density_dist_{track_name}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # --- Box plot (class-wise) ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ed_df["class_label"] = ed_df["label"].map({0: "Negative", 1: "Positive"})
    sns.boxplot(
        data=ed_df, x="class_label", y="edge_density",
        palette={"Negative": "#378ADD", "Positive": "#D85A30"}, ax=ax,
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Edge Density")
    ax.set_title(f"14 — Edge Density Box Plot — {track_name}")
    plt.tight_layout()
    plt.savefig(
        f"{PLOT_DIR}14_edge_density_box_{track_name}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    return ed_df
