import os
import argparse
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
from sklearn.naive_bayes import GaussianNB

def log(text, result_file):
    with open(result_file, "a") as f:
        f.write(text + "\n")
        f.flush()

def evaluate(model, Xtr, Xte, y_train, y_test, name, mode, result_file):
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xte)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    log("="*70, result_file)
    log(f"MODEL: {name} | FEATURES: {mode}", result_file)
    log(f"Accuracy: {acc:.6f}", result_file)
    log(f"Precision: {prec:.6f}", result_file)
    log(f"Recall: {rec:.6f}", result_file)
    log(f"F1 Score: {f1:.6f}", result_file)

    if proba is not None:
        try:
            auc = roc_auc_score(y_test, proba)
            log(f"ROC AUC: {auc:.6f}", result_file)
        except:
            pass

    log("Confusion Matrix:", result_file)
    log(np.array2string(cm), result_file)
    log("="*70 + "\n", result_file)

def main():
    parser = argparse.ArgumentParser(description="Run Naive Bayes model")
    parser.add_argument("--train_path", required=True, help="Path to training CSV")
    parser.add_argument("--test_path", required=True, help="Path to testing CSV")
    parser.add_argument("--selected_features_path", required=True, help="Path to selected features CSV")
    parser.add_argument("--result_file", default="results_naive_bayes.txt", help="Path to results txt file")
    args = parser.parse_args()

    TARGET_COL = "label"
    if not os.path.exists(args.result_file):
        with open(args.result_file, "w") as f:
            f.write("=== NAIVE BAYES RESULTS ===\n\n")

    print("Loading train/test datasets...")
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    selected_df = pd.read_csv(args.selected_features_path)
    SELECTED_FEATURES = selected_df.iloc[:, 0].tolist()

    drop_cols = ["id", TARGET_COL]
    ALL_FEATURES = [c for c in train_df.columns if c not in drop_cols]
    SELECTED_FEATURES = [c for c in SELECTED_FEATURES if c in ALL_FEATURES]

    print(f"All features: {len(ALL_FEATURES)}, Selected features: {len(SELECTED_FEATURES)}")

    X_train_all = train_df[ALL_FEATURES].copy()
    X_test_all = test_df[ALL_FEATURES].copy()
    X_train_sel = train_df[SELECTED_FEATURES].copy()
    X_test_sel = test_df[SELECTED_FEATURES].copy()

    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]

    X_train_all = X_train_all.fillna(X_train_all.median())
    X_test_all = X_test_all.fillna(X_train_all.median())
    X_train_sel = X_train_sel.fillna(X_train_sel.median())
    X_test_sel = X_test_sel.fillna(X_train_sel.median())

    for mode in ["all", "selected"]:
        print(f"\nRunning naive_bayes for mode: {mode}")
        log(f"\n>>> STARTING naive_bayes __{mode}", args.result_file)

        Xtr = X_train_all if mode == "all" else X_train_sel
        Xte = X_test_all if mode == "all" else X_test_sel

        model = GaussianNB()
        evaluate(model, Xtr, Xte, y_train, y_test, "naive_bayes", mode, args.result_file)

    print("\nNaive Bayes evaluations done.")

if __name__ == "__main__":
    main()
