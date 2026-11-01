"""Portable manifest validation shared by tensor loading and NumPy evaluation."""

import csv
from pathlib import Path, PurePosixPath

from .io import SPLITS


def load_manifest(data_root, split, *, complete=False):
    if split not in SPLITS:
        raise ValueError(f"split must be 'dev' or 'test'; got {split!r}")
    root = Path(data_root).resolve()
    sequence, count = SPLITS[split]
    path = root / "manifests" / f"{split}.csv"
    required = {"sample_id", "sequence", "frame_id", "rgb_path", "depth_path"}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing manifest columns in {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    seen = set()
    for row in rows:
        frame = row.get("frame_id") or ""
        if row["sequence"] != sequence or not frame.isascii() or not frame.isdecimal():
            raise ValueError(f"Invalid sequence or frame ID for {split}: {row}")
        if row["sample_id"] != f"{sequence}/{frame}" or int(frame) in seen:
            raise ValueError(f"Invalid or duplicate sample ID: {row}")
        seen.add(int(frame))
        for key, suffix in (("rgb_path", "_color.png"), ("depth_path", "_depth.tiff")):
            relative = PurePosixPath(row.get(key) or "")
            target = (root / relative).resolve()
            file_frame = relative.name.removesuffix(suffix)
            if (
                relative.is_absolute() or ".." in relative.parts
                or relative.parts[:2] != ("raw", sequence)
                or not target.is_relative_to(root)
                or not relative.name.endswith(suffix)
                or not file_frame.isascii() or not file_frame.isdecimal()
                or int(file_frame) != int(frame)
            ):
                raise ValueError(f"Invalid {key} for {row['sample_id']}: {relative}")
            if not target.is_file():
                raise FileNotFoundError(target)
    if complete and (
        len(rows) != count or seen != set(range(count))
        or any(row["frame_id"] != f"{int(row['frame_id']):04d}" for row in rows)
    ):
        raise ValueError(f"{split}: expected exactly {count} canonical frame IDs")
    return rows
