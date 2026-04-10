import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from EDA.src.texture_analysis import extract_glcm_features, extract_lbp_features
from EDA.src.morphological_analysis import laplacian_variance, edge_density
from EDA.src.config import PLOT_DIR, REPORT_DIR
from EDA.src.batch_utils import iter_dataframe_batches


def _lbp_entropy(hist):
    h = hist[hist > 0]
    return float(-np.sum(h * np.log2(h)))


def _symmetric_kl(values_neg, values_pos, bins=50, eps=1e-10):
    min_v = min(float(np.min(values_neg)), float(np.min(values_pos)))
    max_v = max(float(np.max(values_neg)), float(np.max(values_pos)))

    if np.isclose(min_v, max_v):
        return 0.0

    p, _ = np.histogram(values_neg, bins=bins, range=(min_v, max_v), density=True)
    q, _ = np.histogram(values_pos, bins=bins, range=(min_v, max_v), density=True)

    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()

    return float(0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))))


def _extract_feature_row(img):
    img_uint8 = (img * 255).astype(np.uint8)
    row = {}

    rgb_mean = img.mean(axis=(0, 1))
    rgb_std = img.std(axis=(0, 1))
    row["R_mean"], row["G_mean"], row["B_mean"] = rgb_mean
    row["R_std"], row["G_std"], row["B_std"] = rgb_std

    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    row["H_mean"] = hsv[:, :, 0].mean() / 180.0
    row["S_mean"] = hsv[:, :, 1].mean() / 255.0
    row["V_mean"] = hsv[:, :, 2].mean() / 255.0

    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    row["L_mean"] = lab[:, :, 0].mean() / 255.0
    row["A_mean"] = lab[:, :, 1].mean() / 255.0
    row["B_lab_mean"] = lab[:, :, 2].mean() / 255.0

    glcm = extract_glcm_features(img)
    row["GLCM_contrast"] = glcm["glcm_contrast"]
    row["GLCM_dissimilarity"] = glcm["glcm_dissimilarity"]
    row["GLCM_homogeneity"] = glcm["glcm_homogeneity"]
    row["GLCM_energy"] = glcm["glcm_energy"]
    row["GLCM_correlation"] = glcm["glcm_correlation"]

    lbp = extract_lbp_features(img)
    for i, v in enumerate(lbp):
        row[f"LBP_bin{i:02d}"] = float(v)
    row["LBP_entropy"] = _lbp_entropy(lbp)

    row["Sharpness"] = laplacian_variance(img)
    row["Edge_density"] = edge_density(img)

    return row


def generate_kl_reports(df, loader, track_name, batch_size=1024):
    records = []
    labels = []

    for batch in iter_dataframe_batches(df, batch_size):
        for _, row in tqdm(batch.iterrows(), total=len(batch), desc=f"KL {track_name}", leave=False):
            img = loader(row["filepath"])
            records.append(_extract_feature_row(img))
            labels.append(int(row["label"]))

    feature_df = pd.DataFrame(records)
    labels = np.array(labels)

    kl_rows = []
    for col in feature_df.columns:
        neg_vals = feature_df.loc[labels == 0, col].to_numpy()
        pos_vals = feature_df.loc[labels == 1, col].to_numpy()
        kl_rows.append({"Feature": col, "KL_Divergence": _symmetric_kl(neg_vals, pos_vals)})

    kl_df = pd.DataFrame(kl_rows).sort_values("KL_Divergence", ascending=False).reset_index(drop=True)
    kl_df.to_csv(f"{REPORT_DIR}kl_divergence_{track_name}.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, max(6, len(kl_df) * 0.35)))
    ax.barh(kl_df["Feature"], kl_df["KL_Divergence"], color="#2A9D8F")
    ax.invert_yaxis()
    ax.set_xlabel("Symmetric KL Divergence")
    ax.set_title(f"13 — Symmetric KL Divergence (Individual Metrics, Decreasing) — {track_name}")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}13_kl_divergence_{track_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    return kl_df
