"""The model-independent dataset shared by the baseline and pretrained model."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .io import read_pair
from .manifest import load_manifest


class C3VDDataset(Dataset):
    """Read a prepared manifest without resizing, augmentation, or model normalization."""

    def __init__(self, data_root, split):
        self.data_root = Path(data_root).resolve()
        self.split = split
        self.manifest_path = self.data_root / "manifests" / f"{split}.csv"
        self.samples = load_manifest(self.data_root, split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        rgb, _, depth_mm, mask = read_pair(
            self.data_root / row["rgb_path"], self.data_root / row["depth_path"],
        )
        image = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32).div_(255.0)
        return {
            "image": image,
            "depth_mm": torch.from_numpy(depth_mm),
            "valid_mask": torch.from_numpy(mask),
            "sample_id": row["sample_id"],
            "sequence": row["sequence"],
        }
