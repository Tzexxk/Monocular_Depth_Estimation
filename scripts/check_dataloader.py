"""Traverse both audited C3VD splits through the shared CPU DataLoader."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from c3vd_data.dataset import C3VDDataset
from c3vd_data.io import SPLITS


def check_split(data_root, split, audit):
    dataset = C3VDDataset(data_root, split)
    sequence, expected = SPLITS[split]
    digest = hashlib.sha256(dataset.manifest_path.read_bytes()).hexdigest()
    if digest != audit["manifests"][split]["sha256"]:
        raise ValueError(f"Manifest changed after stage 2: {dataset.manifest_path}")
    if len(dataset) != expected:
        raise ValueError(f"{split}: expected {expected} samples, got {len(dataset)}")
    source = next(item for item in audit["sequences"] if item["sequence"] == sequence)
    height, width = source["height"], source["width"]
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    seen, valid_pixels = [], 0
    for index, batch in enumerate(loader):
        image, depth, mask = batch["image"], batch["depth_mm"], batch["valid_mask"]
        sample_id = batch["sample_id"][0]
        if (
            image.shape != (1, 3, height, width) or image.dtype != torch.float32
            or depth.shape != (1, height, width) or depth.dtype != torch.float32
            or mask.shape != depth.shape or mask.dtype != torch.bool
        ):
            raise ValueError(f"Invalid tensor shape or dtype: {sample_id}")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError(f"Invalid RGB values: {sample_id}")
        if not torch.isfinite(depth).all() or depth.min() < 0 or depth.max() > 100:
            raise ValueError(f"Invalid depth values: {sample_id}")
        if not mask.any() or not torch.equal(mask, (depth > 0) & (depth < 100)):
            raise ValueError(f"Invalid depth mask: {sample_id}")
        if sample_id != dataset.samples[index]["sample_id"] or batch["sequence"] != [sequence]:
            raise ValueError(f"Unexpected sample order or sequence: {sample_id}")
        seen.append(sample_id)
        valid_pixels += int(mask.sum())
        if (index + 1) % 20 == 0 or index + 1 == expected:
            print(f"[{split}] Read {index + 1}/{expected} samples", flush=True)
    if len(seen) != expected or len(set(seen)) != expected:
        raise ValueError(f"Missing or repeated samples: {split}")
    return {
        "split": split, "sequence": sequence, "status": "pass", "samples": len(seen),
        "manifest_sha256": digest, "first_sample_id": seen[0], "last_sample_id": seen[-1],
        "image_shape": [3, height, width], "depth_shape": [height, width],
        "image_dtype": "float32", "depth_dtype": "float32", "mask_dtype": "bool",
        "valid_pixels_total": valid_pixels,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args(argv)
    data_root, output_root = args.data_root.resolve(), args.output_root.resolve()
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "fail", "data_root": str(data_root), "torch_version": torch.__version__,
        "device": "cpu", "batch_size": 1, "num_workers": 0, "shuffle": False, "splits": [],
    }
    path = output_root / "checks" / "dataloader.json"
    try:
        audit_path = output_root / "checks" / "data_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit["status"] != "pass" or Path(audit["data_root"]).resolve() != data_root:
            raise ValueError("Stage 2 must have a passing audit for this data root; rerun prepare_data.py")
        report["data_audit_sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
        for split in SPLITS:
            report["splits"].append(check_split(data_root, split, audit))
        report["status"] = "pass"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print(f"[FAIL] {report['error']}", flush=True)
    finally:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {path}", flush=True)
    print(f"Overall: {report['status'].upper()}", flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
