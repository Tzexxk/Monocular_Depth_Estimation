"""Audit the two prescribed C3VD ZIP archives and prepare paired frame manifests."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import zipfile
import zlib

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Repository scripts can run before an editable installation.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from c3vd_data.io import SPLITS, decode_depth, read_pair


SEQUENCES = {sequence: (split, count) for split, (sequence, count) in SPLITS.items()}
FRAME_PATTERN = re.compile(r"^(\d+)_(color\.png|depth\.tiff)$")
MANIFEST_FIELDS = ["sample_id", "sequence", "frame_id", "rgb_path", "depth_path"]
AUDIT_FIELDS = [
    "sequence", "frame_id", "width", "height", "valid_pixels", "total_pixels",
    "valid_fraction", "zero_pixels", "saturated_pixels",
]


def index_pairs(infos):
    """Match by numeric frame ID, rejecting duplicate IDs even with different padding."""
    indexed = {"color.png": {}, "depth.tiff": {}}
    for info in infos:
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename.replace("\\", "/"))
        match = FRAME_PATTERN.fullmatch(path.name)
        if not match:
            continue
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive member: {info.filename}")
        frame_id, kind = match.groups()
        number = int(frame_id)
        if number in indexed[kind]:
            raise ValueError(f"Duplicate {kind} frame ID {number}: {info.filename}")
        indexed[kind][number] = (frame_id, info)
    rgb_ids, depth_ids = set(indexed["color.png"]), set(indexed["depth.tiff"])
    if not rgb_ids or rgb_ids != depth_ids:
        raise ValueError(
            f"Incomplete RGB/depth pairs: missing depth IDs={sorted(rgb_ids - depth_ids)}, "
            f"missing RGB IDs={sorted(depth_ids - rgb_ids)}; RGB={len(rgb_ids)}, depth={len(depth_ids)}"
        )
    return [
        (indexed["color.png"][number][0], indexed["color.png"][number][1], indexed["depth.tiff"][number][1])
        for number in sorted(rgb_ids)
    ]


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_member(archive, info, target):
    """Reuse matching files; refuse to overwrite files that differ from the archive."""
    if target.exists():
        checksum = 0
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum = zlib.crc32(block, checksum)
        if target.stat().st_size != info.file_size or checksum != info.CRC:
            raise ValueError(f"Existing file differs from ZIP; review it before rerunning: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            with archive.open(info) as source:
                shutil.copyfileobj(source, stream)
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def audit_sequence(data_root, sequence, expected_count):
    archive_path = data_root / "downloads" / f"{sequence}.zip"
    print(f"[{sequence}] Checking archive: {archive_path}", flush=True)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Download the official archive first: {archive_path}")
    digest = sha256_file(archive_path)
    rows, frame_audit = [], []
    shape = None
    with zipfile.ZipFile(archive_path) as archive:
        print(f"[{sequence}] Checking ZIP CRC (all archive members)...", flush=True)
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC failed: {bad_member}")
        pairs = index_pairs(archive.infolist())
        if len(pairs) != expected_count:
            raise ValueError(f"{sequence}: expected {expected_count} pairs, found {len(pairs)}")
        for index, (frame_id, rgb_info, depth_info) in enumerate(pairs):
            raw_dir = data_root / "raw" / sequence
            rgb_path = raw_dir / PurePosixPath(rgb_info.filename.replace("\\", "/")).name
            depth_path = raw_dir / PurePosixPath(depth_info.filename.replace("\\", "/")).name
            extract_member(archive, rgb_info, rgb_path)
            extract_member(archive, depth_info, depth_path)
            rgb, raw, _, mask = read_pair(rgb_path, depth_path)
            if shape is None:
                shape = raw.shape
            if raw.shape != shape:
                raise ValueError(f"Frame dimensions changed within {sequence}: {frame_id}")
            rows.append({
                "sample_id": f"{sequence}/{frame_id}", "sequence": sequence, "frame_id": frame_id,
                "rgb_path": rgb_path.relative_to(data_root).as_posix(),
                "depth_path": depth_path.relative_to(data_root).as_posix(),
            })
            frame_audit.append({
                "sequence": sequence, "frame_id": frame_id,
                "width": rgb.shape[1], "height": rgb.shape[0],
                "valid_pixels": int(mask.sum()), "total_pixels": int(raw.size),
                "valid_fraction": float(mask.mean()),
                "zero_pixels": int((raw == 0).sum()),
                "saturated_pixels": int((raw == 65535).sum()),
            })
            if (index + 1) % 20 == 0 or index + 1 == len(pairs):
                print(f"[{sequence}] Audited {index + 1}/{len(pairs)} pairs", flush=True)
    details = {
        "sequence": sequence, "status": "pass", "archive_sha256": digest,
        "archive_bytes": archive_path.stat().st_size, "archive_crc": "pass",
        "expected_pairs": expected_count, "actual_pairs": len(rows),
        "height": shape[0], "width": shape[1], "depth_dtype": "uint16",
        "valid_fraction_min": min(item["valid_fraction"] for item in frame_audit),
        "valid_fraction_max": max(item["valid_fraction"] for item in frame_audit),
    }
    return rows, frame_audit, details


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_preview(data_root, row, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb, _, depth_mm, mask = read_pair(data_root / row["rgb_path"], data_root / row["depth_path"])
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    color_map = plt.get_cmap("viridis").copy()
    color_map.set_bad("lightgray")
    plotted = axes[1].imshow(np.ma.masked_where(~mask, depth_mm), cmap=color_map, vmin=0, vmax=100)
    axes[1].set_title("Valid z-depth (mm)")
    figure.colorbar(plotted, ax=axes[1], shrink=0.7)
    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Valid mask (white = valid)")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(row["sample_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{row['sequence']}_{row['frame_id']}.png"
    try:
        figure.savefig(path, dpi=150)
    finally:
        plt.close(figure)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args(argv)
    data_root, output_root = args.data_root.resolve(), args.output_root.resolve()
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "fail", "data_root": str(data_root), "sequences": [],
        "depth_decode": "raw_uint16 / 65535 * 100 millimeters",
        "valid_mask": "0 < raw_uint16 < 65535", "previews": [], "manifests": {},
    }
    # Preflight both archives before writing raw files or manifests.
    report_path = output_root / "checks" / "data_audit.json"
    try:
        missing = [str(data_root / "downloads" / f"{name}.zip") for name in SEQUENCES
                   if not (data_root / "downloads" / f"{name}.zip").is_file()]
        if missing:
            raise FileNotFoundError("Missing ZIP archives: " + ", ".join(missing))
        manifests, audits = {}, []
        for sequence, (split, expected_count) in SEQUENCES.items():
            rows, checks, details = audit_sequence(data_root, sequence, expected_count)
            details["split"] = split
            report["sequences"].append(details)
            manifests[split] = rows
            audits.extend(checks)
        dev_rows = manifests["dev"]
        for index in sorted({0, len(dev_rows) // 2, len(dev_rows) - 1}):
            preview = save_preview(data_root, dev_rows[index], output_root / "previews")
            report["previews"].append(str(preview))
        for split, rows in manifests.items():
            path = data_root / "manifests" / f"{split}.csv"
            write_csv(path, MANIFEST_FIELDS, rows)
            report["manifests"][split] = {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}
        write_csv(output_root / "checks" / "data_frames.csv", AUDIT_FIELDS, audits)
        report["status"] = "pass"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print(f"[FAIL] {report['error']}", flush=True)
    finally:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}", flush=True)
    print(f"Overall: {report['status'].upper()}", flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
