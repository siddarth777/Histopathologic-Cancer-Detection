
import gc
import json
import os
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from ..src_lda.utils import seed_everything


def load_train_dataframe(data_dir: str, seed: int, val_split: float):
    csv_path = os.path.join(data_dir, 'train_labels.csv')
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        random_state=seed,
        stratify=df['label'],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def trial_seed(base_seed: int, trial_number: int) -> int:
    return int(base_seed) + int(trial_number)


def set_trial_seed(base_seed: int, trial_number: int):
    seed_everything(trial_seed(base_seed, trial_number))


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def trial_dir(out_dir: str, model_name: str, trial_number: int) -> str:
    path = os.path.join(out_dir, 'optuna', model_name, f'trial_{trial_number:04d}')
    os.makedirs(path, exist_ok=True)
    return path


def model_out_dir(out_dir: str, model_name: str) -> str:
    path = os.path.join(out_dir, 'optuna', model_name)
    os.makedirs(path, exist_ok=True)
    return path


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json(path: str, data: dict[str, Any]):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
