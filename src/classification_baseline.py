import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.preprocessing import StandardScaler as SS

from config import PLOT_DIR, MODEL_DIR, SEED

CLASSIFIERS = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=SEED),
    "SVM (RBF)":            SVC(kernel="rbf", probability=True, random_state=SEED),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=SEED),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=SEED),
    "K-Nearest Neighbors":  KNeighborsClassifier(n_neighbors=5),
}

def evaluate_classifiers(X_tr, X_val, y_tr, y_val, track_name):
    sc = SS()
    X_tr_s  = sc.fit_transform(X_tr)
    X_val_s = sc.transform(X_val)

    results = []
    fig_cm, axes_cm = plt.subplots(1, 5, figsize=(22, 4))
    fig_roc, roc_ax = plt.subplots(figsize=(8, 6))

    for i, (name, clf) in enumerate(CLASSIFIERS.items()):
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_val_s)
        y_prob = clf.predict_proba(X_val_s)[:, 1] if hasattr(clf, "predict_proba") \
                 else clf.decision_function(X_val_s)

        acc  = accuracy_score(y_val, y_pred)
        auc  = roc_auc_score(y_val, y_prob)
        f1   = f1_score(y_val, y_pred)
        results.append({"Classifier": name, "Accuracy": acc, "AUC-ROC": auc, "F1": f1})

        cm = confusion_matrix(y_val, y_pred)
        ConfusionMatrixDisplay(cm).plot(ax=axes_cm[i], colorbar=False)
        axes_cm[i].set_title(f"{name}\nAcc={acc:.3f}", fontsize=8)

        RocCurveDisplay.from_predictions(y_val, y_prob, ax=roc_ax, name=name, alpha=0.7)

        joblib.dump(clf, f"{MODEL_DIR}{track_name}_{name.replace(' ','_')}.pkl")

    fig_cm.suptitle(f"Confusion Matrices — {track_name}", fontsize=12)
    fig_cm.tight_layout()
    fig_cm.savefig(f"{PLOT_DIR}10_confusion_{track_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig_cm)

    roc_ax.set_title(f"ROC Curves — {track_name}")
    fig_roc.tight_layout()
    fig_roc.savefig(f"{PLOT_DIR}10_roc_{track_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig_roc)

    res_df = pd.DataFrame(results).sort_values("AUC-ROC", ascending=False)
    res_df.to_csv(f"cancer_eda/outputs/reports/results_{track_name}.csv", index=False)
    return res_df
