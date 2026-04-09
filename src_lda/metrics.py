from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def compute_binary_metrics(labels: Sequence[float], probs: Sequence[float]) -> dict:
    labels_arr = np.asarray(labels, dtype=np.float32)
    probs_arr = np.asarray(probs, dtype=np.float32)
    preds = (probs_arr >= 0.5).astype(np.int32)
    acc = float((preds == labels_arr).mean()) if len(labels_arr) else 0.0
    auc = float(roc_auc_score(labels_arr, probs_arr)) if len(np.unique(labels_arr)) > 1 else 0.5
    f1 = float(f1_score(labels_arr, preds, zero_division=0)) if len(labels_arr) else 0.0
    return {'acc': acc, 'auc': auc, 'f1': f1}
