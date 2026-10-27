"""Synthetic checks for the shared sample contract and audited loader traversal."""

from contextlib import redirect_stdout
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from c3vd_data.dataset import C3VDDataset

SPEC = importlib.util.spec_from_file_location("check_dataloader", PROJECT_ROOT / "scripts" / "check_dataloader.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        directory = self.root / "raw" / "trans_t1_a"
        directory.mkdir(parents=True)
        rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]] * 2, dtype=np.uint8)
        self.raw = np.array([[0, 1, 32768], [65534, 65535, 10]], dtype=np.uint16)
        self.rows = []
        for index in range(2):
            frame = f"{index:04d}"
            Image.fromarray(rgb).save(directory / f"{frame}_color.png")
            Image.fromarray(self.raw).save(directory / f"{frame}_depth.tiff")
            self.rows.append({
                "sample_id": f"trans_t1_a/{frame}", "sequence": "trans_t1_a", "frame_id": frame,
                "rgb_path": f"raw/trans_t1_a/{frame}_color.png",
                "depth_path": f"raw/trans_t1_a/{frame}_depth.tiff",
            })
        self.manifest = self.root / "manifests" / "dev.csv"
        self.manifest.parent.mkdir()
        self.write_manifest(self.rows)

    def write_manifest(self, rows):
        with self.manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def test_sample_contract_preserves_rgb_and_depth(self):
        sample = C3VDDataset(self.root, "dev")[0]
        self.assertEqual(sample["sample_id"], "trans_t1_a/0000")
        self.assertEqual(sample["image"].shape, (3, 2, 3))
        self.assertEqual(sample["image"].dtype, torch.float32)
        torch.testing.assert_close(sample["image"][:, 0, 0], torch.tensor([1., 0., 0.]))
        torch.testing.assert_close(sample["image"][:, 0, 2], torch.tensor([0., 0., 1.]))
        self.assertEqual(sample["depth_mm"].dtype, torch.float32)
        torch.testing.assert_close(
            sample["depth_mm"], torch.from_numpy(self.raw.astype(np.float32) / 65535 * 100),
        )
        self.assertEqual(sample["valid_mask"].dtype, torch.bool)
        self.assertEqual(sample["valid_mask"].tolist(), [[False, True, True], [True, False, True]])

    def test_dataloader_keeps_order_and_adds_batch_dimension(self):
        loader = DataLoader(C3VDDataset(self.root, "dev"), batch_size=1, num_workers=0, shuffle=False)
        batches = list(loader)
        self.assertEqual([item["sample_id"][0] for item in batches], [row["sample_id"] for row in self.rows])
        self.assertEqual(batches[0]["image"].shape, (1, 3, 2, 3))
        self.assertEqual(batches[0]["depth_mm"].shape, (1, 2, 3))

    def test_invalid_splits_ids_and_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "split must"):
            C3VDDataset(self.root, "train")
        for key, value in [
            ("sequence", "trans_t2_b"), ("sample_id", "wrong/0000"),
            ("rgb_path", "../0000_color.png"), ("depth_path", "raw/trans_t1_a/0001_depth.tiff"),
        ]:
            with self.subTest(key=key):
                rows = [dict(row) for row in self.rows]
                rows[0][key] = value
                self.write_manifest(rows)
                with self.assertRaises(ValueError):
                    C3VDDataset(self.root, "dev")
        self.write_manifest([self.rows[0], self.rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            C3VDDataset(self.root, "dev")

    def test_traversal_report_requires_the_audited_manifest(self):
        output = self.root / "outputs"
        (output / "checks").mkdir(parents=True)
        audit = {
            "status": "pass", "data_root": str(self.root),
            "manifests": {"dev": {"sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest()}},
            "sequences": [{"sequence": "trans_t1_a", "height": 2, "width": 3}],
        }
        (output / "checks" / "data_audit.json").write_text(json.dumps(audit), encoding="utf-8")
        arguments = ["--data-root", str(self.root), "--output-root", str(output)]
        with patch.object(checker, "SPLITS", {"dev": ("trans_t1_a", 2)}), redirect_stdout(io.StringIO()):
            self.assertEqual(checker.main(arguments), 0)
            path = output / "checks" / "dataloader.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["splits"][0]["samples"], 2)
            self.assertEqual(report["splits"][0]["valid_pixels_total"], 8)
            self.write_manifest(list(reversed(self.rows)))
            self.assertEqual(checker.main(arguments), 1)
            self.assertIn("Manifest changed", json.loads(path.read_text(encoding="utf-8"))["error"])


if __name__ == "__main__":
    unittest.main()
