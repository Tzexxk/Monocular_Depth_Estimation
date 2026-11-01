import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def detect_highlight(rgb_tensor):
    rgb = rgb_tensor.numpy().transpose(1, 2, 0)
    rgb8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    hsv = cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV)

    mask = (
        (hsv[..., 2] >= 220)
        & (hsv[..., 1] <= 100)
    ).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )
    mask = cv2.dilate(mask, kernel, iterations=1)

    return torch.from_numpy(
        mask.astype(np.float32)[None].copy()
    )


class HighlightDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        sample["highlight"] = detect_highlight(sample["rgb"])
        return sample

