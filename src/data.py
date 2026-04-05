import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class CancerDataset(Dataset):
    """Loads 96x96 pathology patches; applies per-split augmentations."""

    MEAN = [0.7009, 0.5384, 0.6916]
    STD = [0.2125, 0.2432, 0.1939]

    def __init__(self, df: pd.DataFrame, img_dir: str, split: str = 'train'):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.split = split
        self.tfm = self._build_transforms()

    def _build_transforms(self):
        if self.split == 'train':
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(20),
                T.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.1, hue=0.05),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                T.ToTensor(),
                T.Normalize(self.MEAN, self.STD),
            ])
        return T.Compose([
            T.ToTensor(),
            T.Normalize(self.MEAN, self.STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fpath = os.path.join(self.img_dir, row['id'] + '.tif')
        img = Image.open(fpath).convert('RGB')
        img = self.tfm(img)
        label = int(row['label']) if 'label' in row else -1
        return img, label, row['id']
