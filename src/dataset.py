"""
Step 2: PyTorch Dataset with circular ROI extraction + patch sampling.

Each POM image has a circular field of view. We:
1. Detect the circular ROI automatically
2. Extract random 224x224 patches from within the circle
3. Apply augmentations (rotation, flip - NOT aggressive color changes)
4. Return (image_patch, scalar_features, delta_n)
"""

import os
import random
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def detect_circular_roi(img_np: np.ndarray, threshold: int = 30) -> Tuple[int, int, int]:
    """
    Detect the circular field of view in a POM image.
    Returns (cx, cy, radius) of the bright circular region.
    """
    gray = np.mean(img_np, axis=2).astype(np.uint8)
    mask = gray > threshold

    # Find bounding box of bright region
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        # Fallback: use center of image
        h, w = img_np.shape[:2]
        return w // 2, h // 2, min(w, h) // 3

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cy = (rmin + rmax) // 2
    cx = (cmin + cmax) // 2
    radius = min(rmax - rmin, cmax - cmin) // 2

    # Shrink by 10% to avoid edge artifacts
    radius = int(radius * 0.85)
    return int(cx), int(cy), int(radius)


def sample_patches_from_circle(
    img: Image.Image,
    cx: int, cy: int, radius: int,
    patch_size: int = 224,
    n_patches: int = 20,
) -> List[Image.Image]:
    """
    Sample random patches from within the circular ROI.
    Each patch is fully contained within the circle.
    """
    patches = []
    half = patch_size // 2
    # Maximum distance from center where a patch can be placed
    max_dist = radius - half * 1.42  # account for diagonal

    if max_dist < half:
        # Circle too small; just center-crop
        left = max(0, cx - half)
        top = max(0, cy - half)
        patch = img.crop((left, top, left + patch_size, top + patch_size))
        return [patch] * n_patches

    for _ in range(n_patches * 3):  # oversample to get enough valid patches
        if len(patches) >= n_patches:
            break
        # Random point within circle
        angle = random.uniform(0, 2 * np.pi)
        dist = random.uniform(0, max_dist)
        px = int(cx + dist * np.cos(angle))
        py = int(cy + dist * np.sin(angle))

        # Check bounds
        left = px - half
        top = py - half
        right = px + half
        bottom = py + half

        w, h = img.size
        if left < 0 or top < 0 or right > w or bottom > h:
            continue

        patch = img.crop((left, top, right, bottom))
        patches.append(patch)

    # If we couldn't get enough, duplicate
    while len(patches) < n_patches:
        patches.append(patches[len(patches) % max(1, len(patches) - 1)])

    return patches[:n_patches]


class POMPatchDataset(Dataset):
    """
    Dataset that extracts patches from POM images.

    For training: extracts n_patches random patches per image (with augmentation).
    For validation: extracts center crops (deterministic).
    """

    def __init__(
        self,
        csv_path: str,
        mode: str = "train",
        patch_size: int = 224,
        n_patches: int = 25,
        transform: Optional[T.Compose] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.mode = mode
        self.patch_size = patch_size
        self.n_patches = n_patches if mode == "train" else 5  # fewer for val

        # Default transforms
        if transform is not None:
            self.transform = transform
        elif mode == "train":
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(180),
                # MILD color jitter - color IS the signal!
                T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.03, hue=0.01),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        # Precompute: extract all patches and build flat index
        self.samples = []  # List of (image_idx, patch_idx)
        self._patches_cache = {}
        self._scalars_cache = {}
        self._targets_cache = {}

        print(f"Building {mode} dataset with {self.n_patches} patches/image...")
        for idx, row in self.df.iterrows():
            img = Image.open(row["image_path"]).convert("RGB")
            img_np = np.array(img)
            cx, cy, r = detect_circular_roi(img_np)

            patches = sample_patches_from_circle(
                img, cx, cy, r,
                patch_size=self.patch_size,
                n_patches=self.n_patches,
            )

            self._patches_cache[idx] = patches
            self._scalars_cache[idx] = np.array([
                row["temperature_C"] / 60.0,       # normalize temperature
                row["thickness_um"] / 15.0,         # normalize thickness
                row["order"] / 6.0,                 # normalize order
                row.get("reduced_temp", 0.8),       # T/Tc
            ], dtype=np.float32)
            self._targets_cache[idx] = np.float32(row["delta_n"])

            for p_idx in range(len(patches)):
                self.samples.append((idx, p_idx))

        print(f"  Total patches: {len(self.samples)} "
              f"({len(self.df)} images x ~{self.n_patches} patches)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, flat_idx):
        img_idx, patch_idx = self.samples[flat_idx]

        patch_img = self._patches_cache[img_idx][patch_idx]
        scalars = torch.from_numpy(self._scalars_cache[img_idx])
        target = torch.tensor(self._targets_cache[img_idx])

        if self.transform:
            patch_tensor = self.transform(patch_img)
        else:
            patch_tensor = T.ToTensor()(patch_img)

        return patch_tensor, scalars, target

    def get_image_index(self, flat_idx):
        """Get the original image index for a flat sample index."""
        return self.samples[flat_idx][0]


class POMWholeImageDataset(Dataset):
    """
    Dataset that returns center-cropped whole images (for feature extraction baseline).
    """

    def __init__(self, csv_path: str, crop_size: int = 224):
        self.df = pd.read_csv(csv_path)
        self.crop_size = crop_size
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img_np = np.array(img)
        cx, cy, r = detect_circular_roi(img_np)

        # Center crop from ROI center
        half = self.crop_size // 2
        w, h = img.size
        left = max(0, min(cx - half, w - self.crop_size))
        top = max(0, min(cy - half, h - self.crop_size))
        patch = img.crop((left, top, left + self.crop_size, top + self.crop_size))

        patch_tensor = self.transform(patch)
        scalars = torch.tensor([
            row["temperature_C"] / 60.0,
            row["thickness_um"] / 15.0,
            row["order"] / 6.0,
            row.get("reduced_temp", 0.8),
        ], dtype=torch.float32)
        target = torch.tensor(row["delta_n"], dtype=torch.float32)

        return patch_tensor, scalars, target


if __name__ == "__main__":
    # Quick test
    ds = POMPatchDataset("data/dataset_unified.csv", mode="train", n_patches=5)
    print(f"Dataset size: {len(ds)}")
    img, scalars, target = ds[0]
    print(f"Image shape: {img.shape}, scalars: {scalars}, target: {target}")
