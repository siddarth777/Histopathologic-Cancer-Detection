import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .config import CFG


def run_eda(data_dir: str, out_dir: str):
    print('\n' + '=' * 70)
    print('  EXPLORATORY DATA ANALYSIS')
    print('=' * 70)

    train_csv = os.path.join(data_dir, 'train_labels.csv')
    train_img = os.path.join(data_dir, 'train')
    df = pd.read_csv(train_csv)

    print(f'\n[EDA] Total samples : {len(df):,}')
    vc = df['label'].value_counts()
    print(f'[EDA] Class balance  : 0={vc.get(0,0):,}  1={vc.get(1,0):,}  ratio={vc.get(1,0)/len(df)*100:.1f}% positive')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('EDA — Dataset Overview', fontsize=14, fontweight='bold')

    ax = axes[0]
    counts = df['label'].value_counts().sort_index()
    bars = ax.bar(['Negative (0)', 'Positive (1)'], counts.values,
                  color=['steelblue', 'tomato'], edgecolor='white', width=0.5)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f'{v:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title('Class Distribution')
    ax.set_ylabel('Sample Count')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    ax.axis('off')
    ax.set_title('Sample Patches (top: negative · bottom: positive)')
    sample_neg = df[df['label'] == 0].sample(4, random_state=CFG['seed'])
    sample_pos = df[df['label'] == 1].sample(4, random_state=CFG['seed'])
    combined = pd.concat([sample_neg, sample_pos])

    inner = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=axes[1].get_subplotspec(),
                                             hspace=0.05, wspace=0.05)
    for i, (_, row) in enumerate(combined.iterrows()):
        sub = fig.add_subplot(inner[i // 4, i % 4])
        try:
            img = Image.open(os.path.join(train_img, row['id'] + '.tif'))
            sub.imshow(img)
        except Exception:
            sub.text(0.5, 0.5, 'N/A', ha='center', va='center')
        sub.axis('off')
        color = 'tomato' if row['label'] == 1 else 'steelblue'
        for spine in sub.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    ax = axes[2]
    sample_ids = df.sample(min(500, len(df)), random_state=CFG['seed'])
    r_vals, g_vals, b_vals = [], [], []
    for _, row in sample_ids.iterrows():
        try:
            arr = np.array(Image.open(
                os.path.join(train_img, row['id'] + '.tif')).convert('RGB'))
            r_vals.append(arr[:, :, 0].mean())
            g_vals.append(arr[:, :, 1].mean())
            b_vals.append(arr[:, :, 2].mean())
        except Exception:
            pass

    ax.hist(r_vals, bins=40, alpha=0.6, color='red', label='R', density=True)
    ax.hist(g_vals, bins=40, alpha=0.6, color='green', label='G', density=True)
    ax.hist(b_vals, bins=40, alpha=0.6, color='blue', label='B', density=True)
    ax.set_title('Mean Pixel Intensity (500 sample patches)')
    ax.set_xlabel('Mean Intensity')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    eda_path = os.path.join(out_dir, 'eda_overview.png')
    plt.savefig(eda_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'[EDA] overview plot saved → {eda_path}')

    print('\n[EDA] Computing per-class pixel statistics...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('EDA — Per-Class Pixel Analysis', fontsize=14, fontweight='bold')

    for cls_label, color, label_txt in [(0, 'steelblue', 'Negative'), (1, 'tomato', 'Positive')]:
        subset = df[df['label'] == cls_label].sample(min(200, len(df)), random_state=0)
        means, stds = [], []
        for _, row in subset.iterrows():
            try:
                arr = np.array(Image.open(
                    os.path.join(train_img, row['id'] + '.tif')).convert('RGB'))
                means.append(arr.mean())
                stds.append(arr.std())
            except Exception:
                pass
        axes[0].hist(means, bins=30, alpha=0.6, color=color, label=label_txt, density=True)
        axes[1].hist(stds, bins=30, alpha=0.6, color=color, label=label_txt, density=True)

    for ax, title, xlabel in zip(axes,
                                 ['Mean Pixel Value per Patch', 'Std Pixel Value per Patch'],
                                 ['Mean', 'Std']):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    eda2_path = os.path.join(out_dir, 'eda_per_class_stats.png')
    plt.savefig(eda2_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'[EDA] per-class stats plot saved → {eda2_path}')

    print('[EDA] Analysing center 32x32 tumor region vs full patch...')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('EDA — Center Region (32×32) vs Full Patch Analysis', fontsize=14, fontweight='bold')

    center_means_by_class = {0: [], 1: []}
    full_means_by_class = {0: [], 1: []}

    sample100 = df.sample(min(300, len(df)), random_state=0)
    for _, row in sample100.iterrows():
        try:
            arr = np.array(Image.open(
                os.path.join(train_img, row['id'] + '.tif')).convert('RGB'))
            h, w = arr.shape[:2]
            cx, cy = w // 2, h // 2
            center = arr[cy - 16:cy + 16, cx - 16:cx + 16]
            center_means_by_class[int(row['label'])].append(center.mean())
            full_means_by_class[int(row['label'])].append(arr.mean())
        except Exception:
            pass

    colors = {0: 'steelblue', 1: 'tomato'}
    labels_map = {0: 'Negative', 1: 'Positive'}
    for cls in [0, 1]:
        axes[0].hist(center_means_by_class[cls], bins=25, alpha=0.6,
                     color=colors[cls], label=labels_map[cls], density=True)
        axes[1].hist(full_means_by_class[cls], bins=25, alpha=0.6,
                     color=colors[cls], label=labels_map[cls], density=True)

    for ax, title in zip(axes, ['Center 32×32 Mean Intensity', 'Full Patch Mean Intensity']):
        ax.set_title(title)
        ax.set_xlabel('Mean Pixel Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    eda3_path = os.path.join(out_dir, 'eda_center_region.png')
    plt.savefig(eda3_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'[EDA] center-region plot saved → {eda3_path}')

    print('[EDA] Complete.\n')
