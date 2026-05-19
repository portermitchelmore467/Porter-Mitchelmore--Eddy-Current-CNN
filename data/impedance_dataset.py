# impedance_dataset.py

import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import random


class CrackDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, image_key="image_path", label_key="label"):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Build full image path
        img_fp = os.path.join(self.root_dir, row[self.image_key])

        # Load image
        img = Image.open(img_fp).convert("RGB")

        # Load label
        label = int(row[self.label_key])

        # Apply transforms if provided
        if self.transform:
            img = self.transform(img)

        return img, label


def load_splits(metadata_csv, root_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    df = pd.read_csv(metadata_csv)

    # Train vs temp split
    train_df, tv_df = train_test_split(
        df,
        test_size=1 - train_ratio,
        stratify=df["label"],
        random_state=seed
    )

    # Validation vs test split
    val_size = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        tv_df,
        test_size=1 - val_size,
        stratify=tv_df["label"],
        random_state=seed
    )

    return train_df, val_df, test_df


def build_transforms(img_size=224, grayscale=True, augment_cfg=None, is_train=True):
    to_gray = [T.Grayscale(num_output_channels=1)] if grayscale else []

    base = [
        T.Resize((img_size, img_size)),
        *to_gray,
        T.ToTensor(),
    ]

    # No augmentation for validation/test
    if not is_train or not augment_cfg:
        normalize = T.Normalize(
            mean=[0.5] if grayscale else [0.485, 0.456, 0.406],
            std=[0.5] if grayscale else [0.229, 0.224, 0.225]
        )
        return T.Compose(base + [normalize])

    # Training augmentations
    aug = []

    if augment_cfg.get("flip_h", False):
        aug.append(T.RandomHorizontalFlip(p=0.5))

    if augment_cfg.get("flip_v", False):
        aug.append(T.RandomVerticalFlip(p=0.5))

    deg = augment_cfg.get("rotate_deg", 0)
    if deg > 0:
        aug.append(T.RandomRotation(degrees=deg))

    translate_pct = augment_cfg.get("translate_pct", 0.0)
    if translate_pct > 0:
        aug.append(T.RandomAffine(degrees=0, translate=(translate_pct, translate_pct)))

    brightness = augment_cfg.get("brightness", 0.0)
    contrast = augment_cfg.get("contrast", 0.0)
    if brightness > 0 or contrast > 0:
        aug.append(T.ColorJitter(brightness=brightness, contrast=contrast))

    cutout_pct = augment_cfg.get("cutout_pct", 0.0)

    normalize = T.Normalize(
        mean=[0.5] if grayscale else [0.485, 0.456, 0.406],
        std=[0.5] if grayscale else [0.229, 0.224, 0.225]
    )

    pipeline = base + aug + [normalize]

    if cutout_pct > 0:
        pipeline.append(RandomCutout(cutout_pct))

    return T.Compose(pipeline)


class RandomCutout(torch.nn.Module):
    def __init__(self, pct: float = 0.1):
        super().__init__()
        self.pct = pct

    def forward(self, x):
        c, h, w = x.shape
        ch = max(1, int(h * self.pct))
        cw = max(1, int(w * self.pct))

        # Skip cutout if it would remove everything
        if ch >= h or cw >= w:
            return x

        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)

        x[:, top:top + ch, left:left + cw] = 0
        return x