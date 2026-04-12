
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .config import CFG


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs(*paths: str) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def make_output_dirs(root: str = '.') -> dict:
    csv_dir = os.path.join(root, CFG['csv_dir'])
    plot_dir = os.path.join(root, CFG['plot_dir'])
    log_dir = os.path.join(root, CFG['log_dir'])
    out_dir = os.path.join(root, CFG['out_dir'])
    ensure_dirs(csv_dir, plot_dir, log_dir, out_dir)
    return {'csv': csv_dir, 'plot': plot_dir, 'log': log_dir, 'out': out_dir}


def load_train_dataframe(data_dir: str, seed: int, val_split: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_csv = os.path.join(data_dir, 'train_labels.csv')
    df = pd.read_csv(train_csv)
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        random_state=seed,
        stratify=df['label'],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def save_json(path: str, payload: dict) -> None:
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


@dataclass(frozen=True)
class ExperimentPaths:
    csv_dir: str
    plot_dir: str
    log_dir: str
    out_dir: str

    @classmethod
    def from_root(cls, root: str = '.') -> 'ExperimentPaths':
        dirs = make_output_dirs(root)
        return cls(dirs['csv'], dirs['plot'], dirs['log'], dirs['out'])
