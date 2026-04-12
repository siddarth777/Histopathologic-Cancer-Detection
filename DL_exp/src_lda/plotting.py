from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def plot_training_curves(log_csv_path: str, run_name: str, out_dir: str) -> None:
    if not os.path.exists(log_csv_path):
        return

    df = pd.read_csv(log_csv_path)
    if df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    metrics = ['loss', 'acc', 'auc', 'f1']

    for ax, metric in zip(axes, metrics):
        for phase, color in (('train', '#1f77b4'), ('val', '#d62728')):
            phase_df = df[df['phase'] == phase]
            if metric in phase_df.columns and not phase_df.empty:
                ax.plot(phase_df['epoch'], phase_df[metric], label=phase, color=color)
        ax.set_title(metric.upper())
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f'Training Curves: {run_name}')
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f'{run_name}_curves.png')
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_and_roc(labels, probs, run_name: str, out_dir: str) -> None:
    labels_arr = np.asarray(labels, dtype=np.int32)
    probs_arr = np.asarray(probs, dtype=np.float32)
    if labels_arr.size == 0 or probs_arr.size == 0:
        return

    preds_arr = (probs_arr >= 0.5).astype(np.int32)
    cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax_cm = axes[0]
    im = ax_cm.imshow(cm, cmap='Blues')
    ax_cm.set_title('Confusion Matrix')
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    ax_roc = axes[1]
    if len(np.unique(labels_arr)) > 1:
        fpr, tpr, _ = roc_curve(labels_arr, probs_arr)
        auc = float(roc_auc_score(labels_arr, probs_arr))
        ax_roc.plot(fpr, tpr, color='#2ca02c', label=f'AUC={auc:.4f}')
    else:
        ax_roc.plot([0, 1], [0, 1], color='#2ca02c', label='AUC=0.5000')
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax_roc.set_title('ROC Curve')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.grid(True, alpha=0.3)
    ax_roc.legend(loc='lower right')

    fig.suptitle(f'Validation Diagnostics: {run_name}')
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f'{run_name}_confusion_roc.png')
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
