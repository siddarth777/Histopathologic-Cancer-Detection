import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def print_summary(out_dir: str, model_names: list):
    print('\n' + '=' * 70)
    print('  FINAL RESULTS SUMMARY')
    print('=' * 70)
    print(f'  {"Model":<12} {"Best Val AUC":>14} {"Best Val Acc":>14} {"Best Val F1":>12}')
    print('  ' + '-' * 54)

    summary = []
    for name in model_names:
        ckpt = os.path.join(out_dir, f'{name}_best.pth')
        if os.path.exists(ckpt):
            data = torch.load(ckpt, map_location='cpu')
            row = {'model': name, 'auc': data['val_auc'],
                   'acc': data['val_acc'], 'f1': data['val_f1']}
            summary.append(row)
            print(f'  {name:<12} {data["val_auc"]:>14.4f} {data["val_acc"]:>14.4f} {data["val_f1"]:>12.4f}')
        else:
            print(f'  {name:<12}  (checkpoint not found)')

    json_path = os.path.join(out_dir, 'summary.json')
    pd.DataFrame(summary).to_json(json_path, orient='records', indent=2)
    print(f'\n  Summary saved → {json_path}')

    if summary:
        df_sum = pd.DataFrame(summary).set_index('model')
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df_sum))
        w = 0.25
        ax.bar(x - w, df_sum['auc'], w, label='AUC', color='steelblue')
        ax.bar(x, df_sum['acc'], w, label='Accuracy', color='seagreen')
        ax.bar(x + w, df_sum['f1'], w, label='F1 Score', color='tomato')
        ax.set_xticks(x)
        ax.set_xticklabels([n.upper() for n in df_sum.index])
        ax.set_ylim(0, 1.05)
        ax.set_title('Model Comparison — Validation Metrics', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=8, padding=2)
        plt.tight_layout()
        cmp_path = os.path.join(out_dir, 'model_comparison.png')
        plt.savefig(cmp_path, dpi=130, bbox_inches='tight')
        plt.close()
        print(f'  Comparison chart → {cmp_path}')
