"""Fit one pooled dev median or reuse a verified frozen constant; export and evaluate test predictions."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from c3vd_eval.protocol import digest, fit_constant, load_manifest, read_depth
from c3vd_eval.runner import evaluate, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--constant-config", type=Path, help="Reuse a frozen dev fit after verifying its input hashes")
    args = parser.parse_args()
    predictions = args.output_root / "predictions" / "constant_median"
    results = args.output_root / "results" / "constant_median"
    fitted = args.output_root / "calibration" / "constant_median.json"
    for path in (predictions, results, fitted):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    if args.constant_config:
        model = json.loads(args.constant_config.read_text(encoding="utf-8"))
        if model["fit_split"] != "dev" or model["fit_frames"] != 61:
            raise ValueError("Frozen baseline must be fitted on the complete dev split")
        if digest(args.data_root / "manifests/dev.csv") != model["dev_manifest_sha256"]:
            raise ValueError("Frozen-fit development manifest differs")
        for sample in load_manifest(args.data_root, "dev"):
            if digest(args.data_root / sample["depth_path"]) != model["dev_depth_sha256"][sample["sample_id"]]:
                raise ValueError(f"Frozen-fit GT differs: {sample['sample_id']}")
    else:
        model, _ = fit_constant(args.data_root)
    constant = model["constant_mm"]
    if not np.isfinite(constant) or constant <= 0:
        raise ValueError("Invalid frozen constant")
    # Preserve C's ordering: freeze the fitting artifact before reading test GT.
    write_json(fitted, model)
    predictions.mkdir(parents=True)
    for sample in load_manifest(args.data_root, "test"):
        _, gt, _ = read_depth(args.data_root, sample)
        path = predictions / (sample["sample_id"] + ".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.save(stream, np.full(gt.shape, constant, dtype=np.float32), allow_pickle=False)
    metadata = {
        "model": "constant_median", "weights_sha256": {},
        "preprocessing": "none; constant on original pixel grid",
        "calibration_split": "dev", "calibration": model,
        "prediction_origin": "exported from frozen pooled development median",
    }
    print(json.dumps(evaluate(args.data_root, "test", predictions, results, "constant_median", metadata), indent=2))


if __name__ == "__main__":
    main()
