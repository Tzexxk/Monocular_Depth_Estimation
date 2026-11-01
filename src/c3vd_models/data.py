from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

def load_dataset(data_root, split):
    from c3vd_data.dataset import C3VDDataset
    from c3vd_data.manifest import load_manifest
    load_manifest(data_root, split, complete=True)
    return C3VDDataset(data_root, split)


class RefinementDataset(Dataset):

    def __init__(self, base_dataset, prediction_dir, a, b):
        self.base = base_dataset
        self.prediction_dir = Path(prediction_dir)
        self.a = float(a)
        self.b = float(b)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]

        frame_id = sample["sample_id"].split("/")[-1]

        relative = np.load(
            self.prediction_dir / f"{frame_id}.npy", allow_pickle=False
        ).astype(np.float32)

        if relative.shape != tuple(sample["depth_mm"].shape) or not np.isfinite(relative).all():
            raise ValueError("Relative prediction must match the original GT grid and be finite")

        denominator = self.a * relative + self.b

        if not np.isfinite(denominator).all():
            raise ValueError("Invalid calibration denominator")

        if np.any(denominator <= 0):
            raise ValueError("Non-positive denominator")

        pred_mm = 1.0 / denominator

        rgb = sample["image"].float()
        gt_mm = sample["depth_mm"].float()
        valid = sample["valid_mask"].bool()

        pred = torch.from_numpy(
            (pred_mm[None] / 100.0).copy()
        )

        gt = (gt_mm / 100.0).unsqueeze(0)
        mask = valid.unsqueeze(0)

        return {
            "rgb": rgb,
            "pred": pred,
            "gt": gt,
            "mask": mask,
            "sample_id": sample["sample_id"]
        }


def build_refinement_splits(dev_dataset, pred_dir, a, b):
    from .highlight import HighlightDataset
    full = HighlightDataset(RefinementDataset(dev_dataset, Path(pred_dir), a, b))
    frames = [int(row["frame_id"]) for row in dev_dataset.samples]
    return (Subset(full, [i for i, frame in enumerate(frames) if frame < 40]),
            Subset(full, [i for i, frame in enumerate(frames) if 50 <= frame <= 60]))
