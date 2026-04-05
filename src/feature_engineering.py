import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from texture_analysis import extract_glcm_features, extract_lbp_features
from config import SEED

def build_feature_vector(filepath, loader, pca_model, scaler_model):
    img = loader(filepath)
    img_uint8 = (img * 255).astype(np.uint8)

    rgb_mean = img.mean(axis=(0,1))
    rgb_std  = img.std(axis=(0,1))

    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    hsv_mean = hsv.mean(axis=(0,1)) / 255.0

    glcm_feats = extract_glcm_features(img)
    glcm_vec = np.array(list(glcm_feats.values()))

    lbp_vec = extract_lbp_features(img)

    flat = scaler_model.transform(img.ravel().reshape(1, -1))
    pca_vec = pca_model.transform(flat)[0, :50]

    return np.concatenate([rgb_mean, rgb_std, hsv_mean, glcm_vec, lbp_vec, pca_vec])

def engineer_features(df_sample, loader, pca_model, scaler_model, track_name):
    print(f"Building {track_name} feature matrix...")
    X_feat, y_feat = [], []
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), leave=False):
        feat = build_feature_vector(row["filepath"], loader, pca_model, scaler_model)
        X_feat.append(feat)
        y_feat.append(row["label"])
        
    X_feat = np.array(X_feat)
    y_feat = np.array(y_feat)
    
    return train_test_split(X_feat, y_feat, test_size=0.2, random_state=SEED, stratify=y_feat)
