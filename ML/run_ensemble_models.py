import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import optuna
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

parser = argparse.ArgumentParser(description="Ensemble Pipeline")
parser.add_argument("--train_path")
parser.add_argument("--test_path")
parser.add_argument("--selected_features_path", default="")
parser.add_argument("--result_file", default="results.txt")
parser.add_argument("--n_trials_model", type=int, default=10)
parser.add_argument("--n_trials_weights", type=int, default=10)
args = parser.parse_args()

TRAIN_PATH = args.train_path
TEST_PATH = args.test_path
SELECTED_FEATURES_PATH = args.selected_features_path
RESULT_FILE = args.result_file
N_TRIALS_MODEL = args.n_trials_model
N_TRIALS_WEIGHTS = args.n_trials_weights
TARGET_COL = "label"
CHECKPOINT_FILE = "checkpoint.json"

if not os.path.exists(RESULT_FILE):
    with open(RESULT_FILE, "w") as f:
        f.write("ENSEMBLE RESULTS\n\n")

def log(text):
    with open(RESULT_FILE, "a") as f:
        f.write(text + "\n")
        f.flush()

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        return json.load(open(CHECKPOINT_FILE))
    return []

def save_checkpoint(done):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(done, f)

done = load_checkpoint()

print("Loading data...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
selected_df = pd.read_csv(SELECTED_FEATURES_PATH)
SELECTED_FEATURES = selected_df.iloc[:, 0].tolist()

drop_cols = ["id", TARGET_COL]
ALL_FEATURES = [c for c in train_df.columns if c not in drop_cols]
SELECTED_FEATURES = [c for c in SELECTED_FEATURES if c in ALL_FEATURES]

X_train_all = train_df[ALL_FEATURES].fillna(train_df[ALL_FEATURES].median())
X_test_all = test_df[ALL_FEATURES].fillna(train_df[ALL_FEATURES].median())
X_train_sel = train_df[SELECTED_FEATURES].fillna(train_df[SELECTED_FEATURES].median())
X_test_sel = test_df[SELECTED_FEATURES].fillna(train_df[SELECTED_FEATURES].median())

y_train = train_df[TARGET_COL]
y_test = test_df[TARGET_COL]

def tune_model(name, Xtr, Xval, ytr, yval):
    print(f"\nTuning {name.upper()}...")
    log(f"\nTUNING {name.upper()}")

    def objective(trial):
        if name == "rf":
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                random_state=42
            )
        elif name == "xgb":
            model = XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("lr", 0.01, 0.3),
                eval_metric="logloss"
            )
        elif name == "cat":
            model = CatBoostClassifier(
                iterations=trial.suggest_int("iterations", 50, 200),
                depth=trial.suggest_int("depth", 3, 10),
                learning_rate=trial.suggest_float("lr", 0.01, 0.3),
                verbose=0
            )
        model.fit(Xtr, ytr)
        preds = model.predict_proba(Xval)[:, 1]
        return roc_auc_score(yval, preds)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS_MODEL)
    print(f"Best params for {name.upper()}: {study.best_params}")
    log(f"Best params for {name.upper()}: {study.best_params}")
    return study.best_params

def build_model(name, params):
    if name == "rf":
        return RandomForestClassifier(**params, random_state=42)
    elif name == "xgb":
        return XGBClassifier(**params, eval_metric="logloss")
    elif name == "cat":
        params = params.copy()
        params["learning_rate"] = params.pop("lr")
        return CatBoostClassifier(**params, verbose=0)

def tune_weights(preds_dict, yval):
    print("\nTuning ensemble weights...")
    log("\nTUNING ENSEMBLE WEIGHTS")
    keys = list(preds_dict.keys())

    def objective(trial):
        weights = np.array([trial.suggest_float(k, 0, 1) for k in keys])
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        weights /= weights.sum()
        final = sum(weights[i] * preds_dict[k] for i, k in enumerate(keys))
        return roc_auc_score(yval, final)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS_WEIGHTS)
    print("Best weights:", study.best_params)
    log(f"Best weights: {study.best_params}")
    return study.best_params

def evaluate(preds, name):
    y_pred = (preds > 0.5).astype(int)
    log("="*70)
    log(name)
    log(f"AUC: {roc_auc_score(y_test, preds):.6f}")
    log(f"Accuracy: {accuracy_score(y_test, y_pred):.6f}")
    log(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.6f}")
    log(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.6f}")
    log(f"F1: {f1_score(y_test, y_pred, zero_division=0):.6f}")
    log(np.array2string(confusion_matrix(y_test, y_pred)))
    log("="*70 + "\n")

for mode in ["all", "selected"]:
    if mode in done:
        continue

    print(f"\nMODE: {mode.upper()}")
    log(f"MODE: {mode.upper()}")

    X = X_train_all if mode == "all" else X_train_sel
    X_test = X_test_all if mode == "all" else X_test_sel

    Xtr, Xval, ytr, yval = train_test_split(X, y_train, test_size=0.2, stratify=y_train, random_state=42)

    params_rf = tune_model("rf", Xtr, Xval, ytr, yval)
    params_xgb = tune_model("xgb", Xtr, Xval, ytr, yval)
    params_cat = tune_model("cat", Xtr, Xval, ytr, yval)

    rf = build_model("rf", params_rf)
    xgb = build_model("xgb", params_xgb)
    cat = build_model("cat", params_cat)

    rf.fit(Xtr, ytr)
    xgb.fit(Xtr, ytr)
    cat.fit(Xtr, ytr)

    val_preds = {
        "rf": rf.predict_proba(Xval)[:, 1],
        "xgb": xgb.predict_proba(Xval)[:, 1],
        "cat": cat.predict_proba(Xval)[:, 1],
    }

    rf.fit(X, y_train)
    xgb.fit(X, y_train)
    cat.fit(X, y_train)

    test_preds = {
        "rf": rf.predict_proba(X_test)[:, 1],
        "xgb": xgb.predict_proba(X_test)[:, 1],
        "cat": cat.predict_proba(X_test)[:, 1],
    }

    evaluate((test_preds["rf"] + test_preds["xgb"]) / 2, "RF + XGB")
    evaluate((test_preds["xgb"] + test_preds["cat"]) / 2, "XGB + CAT")
    evaluate((test_preds["cat"] + test_preds["rf"]) / 2, "CAT + RF")

    all_equal = sum(test_preds.values()) / 3
    evaluate(all_equal, "ALL (EQUAL)")

    weights = tune_weights(val_preds, yval)
    w = np.array([weights[k] for k in test_preds.keys()])
    if w.sum() == 0:
        w = np.ones_like(w)
    w /= w.sum()

    weighted = sum(w[i] * test_preds[k] for i, k in enumerate(test_preds))
    evaluate(weighted, "ALL (WEIGHTED)")

    stack_X = np.column_stack(list(val_preds.values()))
    meta = LogisticRegression(max_iter=1000)
    meta.fit(stack_X, yval)

    stack_test = np.column_stack(list(test_preds.values()))
    final_stack = meta.predict_proba(stack_test)[:, 1]

    evaluate(final_stack, "STACKING (LR)")

    done.append(mode)
    save_checkpoint(done)

print("\nDONE")