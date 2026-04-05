import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def plot_training_curves(log_csv: str, model_name: str, out_dir: str):
    df = pd.read_csv(log_csv)
    tr = df[df['phase'] == 'train']
    vl = df[df['phase'] == 'val']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{model_name.upper()} — Training Curves', fontsize=14, fontweight='bold')

    for ax, metric, title in zip(axes, ['loss', 'auc', 'acc'], ['Loss', 'ROC-AUC', 'Accuracy']):
        ax.plot(tr['epoch'], tr[metric], 'o-', label='Train', lw=2)
        ax.plot(vl['epoch'], vl[metric], 's--', label='Val', lw=2)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, f'{model_name}_curves.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  [PLOT] saved → {out}')


def plot_confusion_and_roc(labels, probs, model_name: str, out_dir: str):
    preds = [1 if p >= 0.5 else 0 for p in probs]
    cm = confusion_matrix(labels, preds)
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{model_name.upper()} — Validation Diagnostics', fontsize=14, fontweight='bold')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    ax1.set_title('Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    ax2.plot(fpr, tpr, lw=2, label=f'AUC = {auc:.4f}')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, f'{model_name}_diagnostics.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  [PLOT] saved → {out}')
