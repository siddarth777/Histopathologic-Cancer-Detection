
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms as T

from .config import CFG


class CancerDataset(Dataset):
    """Image dataset for histopathologic cancer detection."""

    def __init__(self, df: pd.DataFrame, img_root: str, split: str):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.split = split
        self.transform = self._build_transform(split)

    def _build_transform(self, split: str):
        common = [
            T.Resize((CFG['img_size'], CFG['img_size'])),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if split == 'train':
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                *common,
            ])
        return T.Compose(common)

    def _resolve_image_path(self, image_id: str) -> str:
        candidate = os.path.join(self.img_root, image_id)
        if os.path.isfile(candidate):
            return candidate

        for ext in ('.tif', '.png', '.jpg', '.jpeg'):
            path = os.path.join(self.img_root, f'{image_id}{ext}')
            if os.path.isfile(path):
                return path

        raise FileNotFoundError(f'Image not found for id={image_id} in {self.img_root}')

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row['id']) if 'id' in row else str(row.iloc[0])
        label = float(row['label']) if 'label' in row else 0.0

        image_path = self._resolve_image_path(image_id)
        with Image.open(image_path) as image:
            image = image.convert('RGB')
            image_tensor = self.transform(image)

        return image_tensor, torch.tensor(label, dtype=torch.float32), image_id


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
