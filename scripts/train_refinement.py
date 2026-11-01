"""Train the returned residual architecture on the fixed development train/validation split."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("simple_cnn", "multiscale_b", "gated_c0", "highlight_c1"), required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--relative-root", type=Path, default=ROOT / "outputs/intermediate/da_v2_relative")
    parser.add_argument("--calibration", type=Path, default=ROOT / "configs/inverse_calibration_clean.json")
    parser.add_argument("--output-root", type=Path, required=True, help="New experiment root; keep historical checkpoints separate")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    from c3vd_eval.runner import runtime_record, write_csv, write_json
    from c3vd_eval.protocol import digest
    from c3vd_models.artifacts import load_calibration
    from c3vd_models.data import load_dataset, build_refinement_splits
    from c3vd_models.training import train_model

    if args.output_root.exists():
        raise FileExistsError(f"Use a new training output directory: {args.output_root}")
    a, b = load_calibration(args.calibration)
    train, val = build_refinement_splits(
        load_dataset(args.data_root, "dev"), args.relative_root / "trans_t1_a", a, b,
    )
    checkpoint, history = train_model(args.model, train, val, args.output_root, seed=args.seed, device=args.device)
    write_csv(args.output_root / "training.csv", history)
    write_json(args.output_root / "training.json", {
        "model": args.model, "seed": args.seed, "checkpoint_sha256": digest(checkpoint),
        "calibration_sha256": digest(args.calibration), "calibration_a": a, "calibration_b": b,
        "training_frames": list(range(40)), "validation_frames": list(range(50, 61)),
        "runtime": runtime_record(),
        "verification": "new training run; numerical equivalence to archived Colab training is not asserted",
    })
    print(checkpoint)


if __name__ == "__main__":
    main()
