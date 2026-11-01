"""Evaluate existing predictions with shared original GT and write traceable results."""

import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from .metrics import depth_metrics, summarize
from .protocol import digest, load_manifest, read_depth


ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def runtime_record():
    packages = {}
    for name in ("numpy", "Pillow", "torch", "torchvision", "scipy", "opencv-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    torch = sys.modules.get("torch")
    numerical_settings = None if torch is None else {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "git_revision": revision, "working_tree_dirty": dirty,
        "numerical_settings": numerical_settings,
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): digest(path)
            for directory in (ROOT / "src", ROOT / "scripts")
            for path in sorted(directory.rglob("*.py"))
        },
    }


def evaluate(data_root, split, predictions, output_dir, method, metadata):
    """Reject any missing, extra, malformed or failed prediction; never change the mask."""
    data_root, predictions, output_dir = map(Path, (data_root, predictions, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite results: {output_dir}")
    if metadata.get("calibration_split") not in ("dev", "none"):
        raise ValueError("Primary evaluation permits dev calibration or none only")
    for key in ("model", "weights_sha256", "preprocessing", "calibration"):
        if key not in metadata:
            raise ValueError(f"Missing metadata: {key}")
    if metadata["model"] != method:
        raise ValueError("Method name differs from prediction metadata")
    samples = load_manifest(data_root, split)
    expected = {row["sample_id"] + ".npy" for row in samples}
    actual = {p.relative_to(predictions).as_posix() for p in predictions.rglob("*.npy")}
    if expected != actual:
        raise ValueError(f"Prediction set mismatch; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    rows, prediction_hashes, depth_hashes = [], {}, {}
    for sample in samples:
        _, gt, mask = read_depth(data_root, sample)
        path = predictions / (sample["sample_id"] + ".npy")
        pred = np.load(path, allow_pickle=False)
        if pred.dtype != np.float32:
            raise ValueError(f"Expected float32 millimeter prediction: {path}, got {pred.dtype}")
        rows.append({"sample_id": sample["sample_id"], **depth_metrics(pred, gt, mask)})
        prediction_hashes[sample["sample_id"]] = digest(path)
        if "prediction_sha256" in metadata and metadata["prediction_sha256"].get(sample["sample_id"]) != prediction_hashes[sample["sample_id"]]:
            raise ValueError(f"Prediction differs from generation metadata: {sample['sample_id']}")
        depth_hashes[sample["sample_id"]] = digest(data_root / sample["depth_path"])
    summary = {
        "method": method, "split": split, **summarize(rows),
        "depth_unit": "mm", "valid_mask": "0 < raw_uint16 < 65535",
        "gt_source": "shared original depth_mm; no normalization round trip",
        "prediction_clipping": False,
        "verification": "existing predictions re-evaluated; inference not verified by this command",
    }
    config = {
        "method": method, "split": split, "metadata": metadata,
        "manifest_sha256": digest(data_root / "manifests" / f"{split}.csv"),
        "prediction_sha256": prediction_hashes, "depth_sha256": depth_hashes,
        "alignment": "none during evaluation", "runtime": runtime_record(),
    }
    output_dir.mkdir(parents=True)
    (output_dir / "figures").mkdir()
    write_csv(output_dir / "per_frame.csv", rows)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "config.json", config)
    return summary
