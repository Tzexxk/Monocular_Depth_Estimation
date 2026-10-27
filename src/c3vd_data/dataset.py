"""The model-independent dataset shared by the baseline and pretrained model."""

import csv
from pathlib import Path, PurePosixPath

import torch
from torch.utils.data import Dataset

from .io import SPLITS, read_pair


class C3VDDataset(Dataset):
    """Read a prepared manifest without resizing, augmentation, or model normalization."""

    def __init__(self, data_root, split):
        if split not in SPLITS:
            raise ValueError(f"split must be 'dev' or 'test'; got {split!r}")
        self.data_root = Path(data_root).resolve()
        self.split = split
        sequence, _ = SPLITS[split]
        self.manifest_path = self.data_root / "manifests" / f"{split}.csv"
        required = {"sample_id", "sequence", "frame_id", "rgb_path", "depth_path"}
        with self.manifest_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing manifest columns in {self.manifest_path}: {sorted(required)}")
            self.samples = list(reader)
        if not self.samples:
            raise ValueError(f"Empty manifest: {self.manifest_path}")
        seen = set()
        for row in self.samples:
            frame_id = row.get("frame_id") or ""
            if row.get("sequence") != sequence or not frame_id.isascii() or not frame_id.isdecimal():
                raise ValueError(f"Invalid sequence or frame ID for {split}: {row}")
            if row.get("sample_id") != f"{sequence}/{frame_id}" or int(frame_id) in seen:
                raise ValueError(f"Invalid or duplicate sample ID: {row}")
            seen.add(int(frame_id))
            for key, suffix in (("rgb_path", "_color.png"), ("depth_path", "_depth.tiff")):
                relative = PurePosixPath(row.get(key) or "")
                path = (self.data_root / relative).resolve()
                name = relative.name
                file_frame = name.removesuffix(suffix)
                if (
                    relative.is_absolute() or ".." in relative.parts
                    or relative.parts[:2] != ("raw", sequence)
                    or not path.is_relative_to(self.data_root)
                    or not name.endswith(suffix) or not file_frame.isascii()
                    or not file_frame.isdecimal() or int(file_frame) != int(frame_id)
                ):
                    raise ValueError(f"Invalid {key} for {row['sample_id']}: {relative}")
                if not path.is_file():
                    raise FileNotFoundError(path)

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
