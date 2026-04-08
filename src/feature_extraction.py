import os, sys, random, warnings, time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2
from skimage import color as skcolor
from skimage.feature import local_binary_pattern
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TRAIN_DIR  = "train/"
LABELS_CSV = "train_labels.csv"
OUT_DIR    = "cancer_eda/outputs/datasets/"
MODEL_DIR  = "cancer_eda/outputs/models/"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Image loaders
def load_full_image(filepath, size=(96, 96)):
    img = Image.open(filepath).convert("RGB").resize(size)
    return np.array(img) / 255.0

def load_center_crop(filepath, crop_size=32):
    img = np.array(Image.open(filepath).convert("RGB"))
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    half = crop_size // 2
    return img[cy-half:cy+half, cx-half:cx+half] / 255.0

# ── Feature extraction 
    ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"]
    + ["H_mean", "S_mean", "V_mean"]
    + ["L_mean", "A_mean", "B_lab_mean"]
    + ["GLCM_contrast", "GLCM_dissimilarity", "GLCM_homogeneity",
       "GLCM_energy", "GLCM_correlation"]
    + [f"LBP_bin{i:02d}" for i in range(26)]
    + ["LBP_entropy"]
    + ["Sharpness", "Edge_density"]
    + [f"PCA_{i+1}" for i in range(50)]
)

def extract_features(filepath, loader, pca_model, scaler_model):
    """Extract the complete feature vector for a single image."""
    img = loader(filepath)
    img_uint8 = (img * 255).astype(np.uint8)
    feats = []

    # RGB stats (6)
    feats.extend(img.mean(axis=(0,1)).tolist())
    feats.extend(img.std(axis=(0,1)).tolist())

    # HSV stats (3)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    feats.append(hsv[:,:,0].mean() / 180.0)
    feats.append(hsv[:,:,1].mean() / 255.0)
    feats.append(hsv[:,:,2].mean() / 255.0)

    # LAB stats (3)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    feats.append(lab[:,:,0].mean() / 255.0)
    feats.append(lab[:,:,1].mean() / 255.0)
    feats.append(lab[:,:,2].mean() / 255.0)

    # GLCM (5)
    gray = (skcolor.rgb2gray(img) * 255).astype(np.uint8)
    glcm = graycomatrix(gray, distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        feats.append(graycoprops(glcm, prop).mean())

    # LBP histogram (26) + entropy (1)
    n_pts, rad = 24, 3
    gray_f = skcolor.rgb2gray(img)
    lbp = local_binary_pattern(gray_f, n_pts, rad, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_pts+2,
                                range=(0, n_pts+2), density=True)
    feats.extend(lbp_hist.tolist())
    h_pos = lbp_hist[lbp_hist > 0]
    feats.append(-np.sum(h_pos * np.log2(h_pos)))  # entropy

    # Morphological (2)
    feats.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 50, 150)
    feats.append(edges.mean() / 255.0)

    # PCA (50)
    flat = scaler_model.transform(img.ravel().reshape(1, -1))
    pca_vec = pca_model.transform(flat)[0, :50]
    feats.extend(pca_vec.tolist())

    return feats

print("Loading labels...")
df = pd.read_csv(LABELS_CSV)
df["filepath"] = df["id"].apply(lambda x: os.path.join(TRAIN_DIR, x + ".tif"))
df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)
print(f"Total images: {len(df)}")

df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"])
df_train = df_train.reset_index(drop=True)
df_test  = df_test.reset_index(drop=True)
print(f"Train: {len(df_train)}  |  Test: {len(df_test)}")

SAMPLE_FIT = 5000
df_fit = pd.concat([
    df_train[df_train["label"]==0].sample(SAMPLE_FIT//2, random_state=SEED),
    df_train[df_train["label"]==1].sample(SAMPLE_FIT//2, random_state=SEED),
]).reset_index(drop=True)

def fit_pca_scaler(df_fit, loader, track_name):
    print(f"Fitting PCA/Scaler for {track_name} on {len(df_fit)} images...")
    X_fit = []
    for _, row in tqdm(df_fit.iterrows(), total=len(df_fit), desc=f"PCA fit {track_name}"):
        X_fit.append(loader(row["filepath"]).ravel())
    X_fit = np.array(X_fit, dtype=np.float32)
    scaler = StandardScaler().fit(X_fit)
    X_scaled = scaler.transform(X_fit)
    pca = PCA(n_components=100, random_state=SEED).fit(X_scaled)
    print(f"  Variance captured by 50 components: {sum(pca.explained_variance_ratio_[:50])*100:.1f}%")
    return pca, scaler

pca_full, scaler_full = fit_pca_scaler(df_fit, load_full_image, "Full96")
pca_crop, scaler_crop = fit_pca_scaler(df_fit, load_center_crop, "Crop32")

def extract_dataset(df_subset, loader, pca_model, scaler_model, track_name, split_name):
    """Extract features for an entire dataframe and return as DataFrame."""
    all_feats = []
    t0 = time.time()
    for idx, (_, row) in enumerate(tqdm(df_subset.iterrows(), total=len(df_subset),
                                         desc=f"{track_name} {split_name}")):
        feats = extract_features(row["filepath"], loader, pca_model, scaler_model)
        all_feats.append(feats)
    
    result_df = pd.DataFrame(all_feats, columns=FEATURE_NAMES)
    result_df.insert(0, "id", df_subset["id"].values)
    result_df["label"] = df_subset["label"].values
    elapsed = time.time() - t0
    print(f"  {track_name} {split_name}: {len(result_df)} rows, {len(FEATURE_NAMES)} features, {elapsed:.0f}s")
    return result_df

for track_name, loader, pca, scaler in [
    ("full96", load_full_image, pca_full, scaler_full),
    ("crop32", load_center_crop, pca_crop, scaler_crop),
]:
    for split_name, df_split in [("train", df_train), ("test", df_test)]:
        out_path = f"{OUT_DIR}{track_name}_{split_name}.csv"
        feat_df = extract_dataset(df_split, loader, pca, scaler, track_name, split_name)
        feat_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}  ({feat_df.shape})")

print("\nAll 4 CSV files generated successfully.")
