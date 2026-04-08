import pandas as pd
import os

def consolidate():
    files = [
        "outputs/reports/results_Crop32.csv",
        "outputs/reports/results_Full96.csv",
        "outputs/reports/results_Crop32_selected.csv",
        "outputs/reports/results_Full96_selected.csv"
    ]
    
    dfs = []
    for f in files:
        if not os.path.exists(f):
            print(f"Warning: {f} not found.")
            continue
        
        # Determine track based on filename
        if "selected" in f:
            t_name = "Crop32 Selected Features" if "Crop32" in f else "Full96 Selected Features"
        else:
            t_name = "Crop32 All Features" if "Crop32" in f else "Full96 All Features"
            
        df = pd.read_csv(f)
        df.insert(0, "Dataset_Track", t_name)
        dfs.append(df)
        
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        # Sort by AUC-ROC descending
        final_df = final_df.sort_values(by="AUC-ROC", ascending=False)
        out_path = "outputs/reports/all_models_comparison.csv"
        final_df.to_csv(out_path, index=False)
        print(f"Successfully consolidated {len(dfs)} reports into {out_path}")
    else:
        print("No report files found to consolidate.")

if __name__ == "__main__":
    consolidate()
