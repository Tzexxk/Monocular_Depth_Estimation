"""Run the pinned official DA V2 Relative Small on shared full-view inputs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--repo", type=Path, default=ROOT / "weights/Depth-Anything-V2")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/depth_anything_v2_vits.pth")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--strict-fp32", action="store_true", help="Disable TF32 for an explicit full-float32 inference comparison")
    parser.add_argument("--max-frames", type=int, help="Small verification subset; use a separate output root")
    args = parser.parse_args()
    import torch
    if args.strict_fp32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    from torch.utils.data import Subset
    from c3vd_eval.protocol import digest
    from c3vd_eval.runner import runtime_record, write_json
    from c3vd_models.data import load_dataset
    from c3vd_models.depth_anything import infer_sequence, load_model

    config = json.loads((ROOT / "configs/da_v2_inference_metadata.json").read_text())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()
    if commit != config["repository_commit"]:
        raise ValueError("Official model source commit differs from the recorded experiment")
    if subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=args.repo).returncode:
        raise ValueError("Official model source has tracked modifications")
    if digest(args.weights) != config["checkpoint_sha256"]:
        raise ValueError("DA checkpoint SHA-256 differs")
    dataset = load_dataset(args.data_root, args.split)
    if args.max_frames is not None:
        if not 1 <= args.max_frames <= len(dataset):
            raise ValueError("max-frames is outside the split")
        dataset = Subset(dataset, range(args.max_frames))
    sequence = "trans_t1_a" if args.split == "dev" else "trans_t2_b"
    output = args.output_root / "intermediate/da_v2_relative" / sequence
    if output.exists():
        raise FileExistsError(output)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model(args.repo, args.weights, device)
    sample_ids = infer_sequence(model, dataset, output, input_size=config["input_size"])
    write_json(output / "inference.json", {
        "model": config, "split": args.split, "frames": len(dataset), "device": device,
        "sample_ids": sample_ids,
        "strict_fp32_requested": args.strict_fp32,
        "manifest_sha256": digest(args.data_root / "manifests" / f"{args.split}.csv"),
        "prediction_sha256": {p.name: digest(p) for p in sorted(output.glob("*.npy"))},
        "output_resize": "bilinear, align_corners=True (official infer_image)",
        "runtime": runtime_record(),
    })
    print(f"Exported {len(dataset)} relative predictions to {output}")


if __name__ == "__main__":
    main()
