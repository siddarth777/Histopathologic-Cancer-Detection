import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from config import PLOT_DIR, SEED

def run_pca_analysis(df_sample, loader, track_name, n_components=100):
    print(f"\n[PCA] Loading images for {track_name}...")
    X, y = [], []
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), leave=False):
        img = loader(row["filepath"])
        X.append(img.ravel())
        y.append(row["label"])
    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)

    # Scree plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(range(1, n_components + 1), pca.explained_variance_ratio_, "o-", ms=3)
    axes[0].set_title(f"Explained Variance per Component — {track_name}")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(range(1, n_components + 1), cum_var, "o-", ms=3, color="#D85A30")
    axes[1].axhline(0.90, color="gray", linestyle="--", label="90% variance")
    axes[1].set_title(f"Cumulative Explained Variance — {track_name}")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Variance")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}07_pca_scree_{track_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Components visualization
    img_shape = loader(df_sample.iloc[0]["filepath"]).shape
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i, ax in enumerate(axes.ravel()):
        comp = pca.components_[i].reshape(img_shape)
        comp_norm = (comp - comp.min()) / (comp.max() - comp.min())
        ax.imshow(comp_norm)
        ax.set_title(f"PC {i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)")
        ax.axis("off")
    plt.suptitle(f"Top 9 Principal Components — {track_name}")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}07_pca_components_{track_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls, color, label in [(0, "#378ADD", "Negative"), (1, "#D85A30", "Positive")]:
        mask = y == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.3, s=5, c=color, label=label)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"PCA 2D Projection — {track_name}")
    ax.legend()
    plt.savefig(f"{PLOT_DIR}07_pca_scatter_{track_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    return pca, scaler, X_pca, X_scaled, y

def run_lda_analysis(X_pca, y, track_name, n_pca_components=50):
    X_input = X_pca[:, :n_pca_components]
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X_input, y)

    fig, ax = plt.subplots(figsize=(10, 4))
    for cls, color, label in [(0, "#378ADD", "Negative"), (1, "#D85A30", "Positive")]:
        sns.kdeplot(X_lda[y == cls, 0], ax=ax, color=color, label=label, fill=True, alpha=0.4)
    ax.set_title(f"LDA 1D Projection — {track_name}")
    ax.set_xlabel("LDA Component 1")
    ax.legend()
    plt.savefig(f"{PLOT_DIR}08_lda_1d_{track_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls, color, label in [(0, "#378ADD", "Negative"), (1, "#D85A30", "Positive")]:
        mask = y == cls
        ax.scatter(X_lda[mask, 0], X_pca[mask, 1], alpha=0.3, s=5, c=color, label=label)
    ax.set_xlabel("LDA Component 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"LDA Component vs PC2 — {track_name}")
    ax.legend()
    plt.savefig(f"{PLOT_DIR}08_lda_2d_{track_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    means = [X_lda[y == c, 0].mean() for c in [0, 1]]
    vars_ = [X_lda[y == c, 0].var() for c in [0, 1]]
    separation = abs(means[1] - means[0]) / (np.sqrt(vars_[0]) + np.sqrt(vars_[1]) + 1e-8)
    print(f"[{track_name}] LDA Separation Score: {separation:.4f}")
    return lda, X_lda, separation
