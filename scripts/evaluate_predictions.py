"""Strictly evaluate complete dev/test millimeter predictions on the original GT grid."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from c3vd_eval.runner import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--predictions", type=Path, required=True, help="Directory containing SEQUENCE/FRAME_ID.npy")
    parser.add_argument("--output-dir", type=Path, required=True, help="New result directory; existing paths are refused")
    parser.add_argument("--method", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    print(json.dumps(evaluate(
        args.data_root, args.split, args.predictions, args.output_dir, args.method, metadata,
    ), indent=2))


if __name__ == "__main__":
    main()
