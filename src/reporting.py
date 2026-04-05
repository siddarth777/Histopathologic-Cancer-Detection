import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PLOT_DIR

def generate_comparative_report(results_full, results_crop, sep_full, sep_crop, total_len, pos_cnt, neg_cnt):
    merged = results_full.merge(results_crop, on="Classifier",
                                 suffixes=(" Full96", " Crop32"))
    
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(merged))
    w = 0.35
    ax.bar(x - w/2, merged["AUC-ROC Full96"], w, label="Full96", color="#378ADD")
    ax.bar(x + w/2, merged["AUC-ROC Crop32"], w, label="Crop32", color="#D85A30")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["Classifier"], rotation=15, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC-ROC Comparison — Full96 vs Crop32")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}11_auc_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    best_full = results_full.iloc[0]["Classifier"]
    best_crop = results_crop.iloc[0]["Classifier"]

    pos_pct = pos_cnt / total_len * 100
    neg_pct = neg_cnt / total_len * 100

    report_md = f"""# Histopathologic Cancer Detection

## Dataset Summary
- Total training samples: {total_len}
- Positive (tumor): {pos_cnt} ({pos_pct:.1f}%)
- Negative: {neg_cnt} ({neg_pct:.1f}%)

## LDA Findings
- Separation Full96={sep_full:.4f} | Crop32={sep_crop:.4f}

## Classification Results

### Full Image (96×96)
{results_full.to_markdown(index=False)}

### Center Crop (32×32)
{results_crop.to_markdown(index=False)}

## Best Models
- Full96: **{best_full}** (AUC = {results_full.iloc[0]['AUC-ROC']:.4f})
- Crop32: **{best_crop}** (AUC = {results_crop.iloc[0]['AUC-ROC']:.4f})
"""

    with open("cancer_eda/outputs/reports/final_report.md", "w") as f:
        f.write(report_md)
