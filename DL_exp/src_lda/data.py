from __future__ import annotations

import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data import CancerDataset

from .config import CFG


def build_image_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame):
    img_root = os.path.join(CFG['data_dir'], 'train')
    train_ds = CancerDataset(train_df, img_root, 'train')
    val_ds = CancerDataset(val_df, img_root, 'val')
    return train_ds, val_ds


def build_image_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame, batch_size: int | None = None):
    batch_size = batch_size or CFG['batch_size']
    train_ds, val_ds = build_image_datasets(train_df, val_df)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=CFG['num_workers'],
        pin_memory=CFG['pin_memory'],
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=CFG['num_workers'],
        pin_memory=CFG['pin_memory'],
        drop_last=False,
    )
    return train_loader, val_loader


def make_feature_loader(features: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    labels = labels.float()
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    dataset = TensorDataset(features.float(), labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
