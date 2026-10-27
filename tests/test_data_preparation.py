"""Synthetic checks only: no real C3VD downloads or GPU are required."""

from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
preparation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preparation)


def image_bytes(values, format):
    with io.BytesIO() as stream:
        Image.fromarray(values).save(stream, format=format)
        return stream.getvalue()


def make_archive(data_root, sequence, count):
    destination = data_root / "downloads" / f"{sequence}.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    rgb = image_bytes(np.full((3, 4, 3), 100, dtype=np.uint8), "PNG")
    depth = image_bytes(np.tile(np.array([0, 1, 32768, 65535], dtype=np.uint16), (3, 1)), "TIFF")
    with zipfile.ZipFile(destination, "w") as archive:
        for number in range(count):
            archive.writestr(f"{sequence}/{number:04d}_color.png", rgb)
            archive.writestr(f"{sequence}/{number:04d}_depth.tiff", depth)
        archive.writestr(f"{sequence}/0000_normals.tiff", b"unused modality")
    return destination


class DataPreparationTests(unittest.TestCase):
    def test_depth_preserves_boundary_values_and_rejects_8bit(self):
        raw = np.array([[0, 1, 32768, 65534, 65535]], dtype=np.uint16)
        depth, mask = preparation.decode_depth(raw)
        self.assertEqual(depth.dtype, np.float32)
        np.testing.assert_allclose(depth, raw.astype(np.float64) / 65535 * 100, rtol=1e-6)
        self.assertEqual(mask.tolist(), [[False, True, True, True, False]])
        with self.assertRaisesRegex(ValueError, "uint16"):
            preparation.decode_depth(np.array([[1]], dtype=np.uint8))

    def test_pairing_rejects_missing_duplicate_and_unsafe_members(self):
        cases = [
            (["0000_color.png"], "Incomplete"),
            (["0000_color.png", "0_color.png", "0000_depth.tiff"], "Duplicate"),
            (["../0000_color.png", "0000_depth.tiff"], "Unsafe"),
        ]
        for filenames, message in cases:
            with self.subTest(filenames=filenames):
                with self.assertRaisesRegex(ValueError, message):
                    preparation.index_pairs([zipfile.ZipInfo(name) for name in filenames])

    def test_size_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            rgb_path, depth_path = Path(directory) / "rgb.png", Path(directory) / "depth.tiff"
            Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(rgb_path)
            Image.fromarray(np.ones((2, 4), dtype=np.uint16)).save(depth_path)
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                preparation.read_pair(rgb_path, depth_path)

    def test_changed_existing_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            data_root = Path(directory)
            make_archive(data_root, "trans_t1_a", 1)
            preparation.audit_sequence(data_root, "trans_t1_a", 1)
            # A second run reuses byte-identical files.
            preparation.audit_sequence(data_root, "trans_t1_a", 1)
            target = data_root / "raw" / "trans_t1_a" / "0000_color.png"
            target.write_bytes(b"changed locally")
            with self.assertRaisesRegex(ValueError, "Existing file differs"):
                preparation.audit_sequence(data_root, "trans_t1_a", 1)
            self.assertEqual(target.read_bytes(), b"changed locally")

    def test_complete_pipeline_and_missing_archive_report(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            data_root, output_root = root / "data", root / "outputs"
            arguments = [
                sys.executable, "-X", "utf8", str(SCRIPT),
                "--data-root", str(data_root), "--output-root", str(output_root),
            ]
            # Exercise the real CLI and keep matplotlib's native libraries in its own process.
            result = subprocess.run(arguments, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            report_path = output_root / "checks" / "data_audit.json"
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8"))["status"], "fail")
            for sequence, (_, count) in preparation.SEQUENCES.items():
                make_archive(data_root, sequence, count)
            result = subprocess.run(arguments, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "pass")
            self.assertEqual(len(report["previews"]), 3)
            self.assertTrue(all(Path(path).is_file() for path in report["previews"]))
            for sequence, (split, count) in preparation.SEQUENCES.items():
                with (data_root / "manifests" / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), count)
                self.assertEqual(rows[0]["sample_id"], f"{sequence}/0000")
                self.assertTrue(all(row["sequence"] == sequence for row in rows))
                self.assertTrue(all(not Path(row["rgb_path"]).is_absolute() for row in rows))
            self.assertEqual(
                len(list((data_root / "raw").rglob("*_normals.tiff"))), 0,
            )


if __name__ == "__main__":
    unittest.main()
