"""Check report persistence and exit codes without accessing a real GPU."""

from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_environment.py"
SPEC = importlib.util.spec_from_file_location("check_environment", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class EnvironmentReportTests(unittest.TestCase):
    def run_report(self, cuda_error=None, require_cuda=False):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            for name in ("check_interpreter", "check_dependency", "check_tiff"):
                stack.enter_context(patch.object(checker, name, return_value={"mocked": True}))
            compute = stack.enter_context(patch.object(
                checker, "check_compute", side_effect=cuda_error, return_value={"mocked": True},
            ))
            output = Path(directory) / "nested" / "environment.json"
            arguments = ["--output", str(output)]
            if require_cuda:
                arguments.append("--require-cuda")
            with redirect_stdout(io.StringIO()) as console:
                code = checker.main(arguments)
            compute.assert_called_once_with(require_cuda=require_cuda)
            return code, json.loads(output.read_text(encoding="utf-8")), console.getvalue()

    def test_cuda_failure_still_saves_other_results(self):
        code, report, console = self.run_report(
            RuntimeError("simulated CUDA failure"), require_cuda=True,
        )
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "fail")
        checks = {check["name"]: check for check in report["checks"]}
        self.assertEqual(checks["uint16_tiff_roundtrip"]["status"], "pass")
        self.assertTrue(report["require_cuda"])
        self.assertIn("simulated CUDA failure", checks["tensor_matrix_multiplication"]["error"])
        self.assertIn("Overall: FAIL", console)

    def test_success_report_returns_zero(self):
        with (
            patch.object(checker.sys, "prefix", str(Path("custom-env").resolve())),
            patch.object(checker.sys, "version_info", (3, 11, 16)),
            patch.object(checker.Path, "is_dir", return_value=True),
        ):
            self.assertEqual(checker.check_interpreter()["environment"], "custom-env")
        code, report, console = self.run_report()
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(all(check["status"] == "pass" for check in report["checks"]))
        self.assertIn("Overall: PASS", console)

    def test_require_cuda_rejects_a_cpu_only_installation(self):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            version=SimpleNamespace(cuda=None),
        )
        with patch.dict("sys.modules", {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                checker.check_compute(require_cuda=True)


if __name__ == "__main__":
    unittest.main()
