"""Check shared manifests, strict exports, original GT use and checkpoint compatibility."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from c3vd_data.io import read_pair
from c3vd_data.manifest import load_manifest
from c3vd_eval.metrics import depth_metrics
from c3vd_eval.runner import evaluate


class IntegrationTests(unittest.TestCase):
    def test_complete_prediction_contract_and_original_gt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, predictions = root / "data", root / "predictions"
            raw_dir = data / "raw/trans_t2_b"
            raw_dir.mkdir(parents=True)
            (data / "manifests").mkdir()
            (predictions / "trans_t2_b").mkdir(parents=True)
            rows = []
            expected = []
            for index in range(2):
                frame = f"{index:04d}"
                raw = np.array([[0, 11311, 65535], [1000, 20000, 50000]], np.uint16)
                Image.fromarray(raw).save(raw_dir / f"{frame}_depth.tiff")
                Image.fromarray(np.zeros((2, 3, 3), np.uint8)).save(raw_dir / f"{frame}_color.png")
                row = {
                    "sample_id": f"trans_t2_b/{frame}", "sequence": "trans_t2_b", "frame_id": frame,
                    "depth_path": f"raw/trans_t2_b/{frame}_depth.tiff",
                    "rgb_path": f"raw/trans_t2_b/{frame}_color.png",
                }
                rows.append(row)
                _, _, gt, mask = read_pair(data / row["rgb_path"], data / row["depth_path"])
                pred = np.full(gt.shape, 17.259479522705078, np.float32)
                np.save(predictions / (row["sample_id"] + ".npy"), pred)
                expected.append(depth_metrics(pred, gt, mask))
            manifest = data / "manifests/test.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            metadata = {
                "model": "fixture", "weights_sha256": {}, "preprocessing": "none",
                "calibration_split": "none", "calibration": {},
            }

            def read_original(data_root, row):
                _, raw, depth, mask = read_pair(data_root / row["rgb_path"], data_root / row["depth_path"])
                return raw, depth, mask

            with patch("c3vd_data.manifest.SPLITS", {"test": ("trans_t2_b", 2)}), patch(
                "c3vd_eval.runner.read_depth", side_effect=read_original,
            ):
                summary = evaluate(data, "test", predictions, root / "results", "fixture", metadata)
                self.assertEqual(summary["frames"], 2)
                for key in ("absrel", "rmse_mm", "delta1"):
                    self.assertEqual(summary[key], np.mean([r[key] for r in expected]))
                with self.assertRaises(FileExistsError):
                    evaluate(data, "test", predictions, root / "results", "fixture", metadata)
                with self.assertRaisesRegex(ValueError, "Method name"):
                    evaluate(data, "test", predictions, root / "mislabeled", "other_model", metadata)
                with self.assertRaisesRegex(ValueError, "generation metadata"):
                    evaluate(data, "test", predictions, root / "changed_prediction", "fixture", {
                        **metadata, "prediction_sha256": {},
                    })
                bad_path = predictions / "trans_t2_b/0001.npy"
                np.save(bad_path, np.ones((2, 3), np.float64))
                with self.assertRaisesRegex(ValueError, "float32"):
                    evaluate(data, "test", predictions, root / "wrong_type", "fixture", metadata)
                np.save(bad_path, np.zeros((2, 3), np.float32))
                with self.assertRaisesRegex(ValueError, "ALL valid"):
                    evaluate(data, "test", predictions, root / "invalid", "fixture", metadata)
                np.save(predictions / "trans_t2_b/0002.npy", np.ones((2, 3), np.float32))
                with self.assertRaisesRegex(ValueError, "set mismatch"):
                    evaluate(data, "test", predictions, root / "extra", "fixture", metadata)
                with self.assertRaisesRegex(ValueError, "dev calibration"):
                    evaluate(data, "test", predictions, root / "test_gt", "fixture", {
                        **metadata, "calibration_split": "test",
                    })
                rows[1]["frame_id"] = "0002"
                rows[1]["sample_id"] = "trans_t2_b/0002"
                rows[1]["rgb_path"] = rows[0]["rgb_path"]
                with manifest.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    load_manifest(data, "test", complete=True)

    def test_returned_architectures_and_exclusive_prediction_save(self):
        from c3vd_models.artifacts import load_checkpoint, save_prediction
        from c3vd_models.models import MODELS
        import torch

        expected = {"simple_cnn": 10721, "multiscale_b": 472097, "gated_c0": 485841, "highlight_c1": 485841}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, constructor in MODELS.items():
                model = constructor()
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected[name])
                path = root / f"{name}.pth"
                torch.save({"model_state_dict": model.state_dict()}, path)
                restored = load_checkpoint(name, path)
                self.assertFalse(restored.training)
            path = root / "prediction.npy"
            prediction = np.full((1080, 1350), 10, np.float32)
            save_prediction(path, prediction)
            with self.assertRaises(FileExistsError):
                save_prediction(path, prediction + 1)
            np.testing.assert_array_equal(np.load(path, allow_pickle=False), prediction)


if __name__ == "__main__":
    unittest.main()
