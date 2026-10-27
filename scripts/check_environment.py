"""Check the shared project environment; optionally require working CUDA."""

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = {
    "numpy": "numpy",
    "Pillow": "PIL.Image",
    "torch": "torch",
    "torchvision": "torchvision",
    "matplotlib": "matplotlib",
}


def run_check(name, operation):
    """Keep a failed check in the report so later checks can still run."""
    try:
        return {"name": name, "status": "pass", "details": operation()}
    except Exception as error:
        return {
            "name": name,
            "status": "fail",
            "error": f"{type(error).__name__}: {error}",
        }


def check_interpreter():
    prefix = Path(sys.prefix)
    if not (prefix / "conda-meta").is_dir():
        raise RuntimeError(f"No conda-meta directory under {prefix}")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("This project uses Python 3.11")
    return {"environment": prefix.name, "executable": sys.executable}


def check_dependency(distribution, module_name):
    module = importlib.import_module(module_name)
    version = importlib.metadata.version(distribution)
    module_path = Path(module.__file__).resolve()
    return {"version": version, "module_path": str(module_path)}


def check_tiff():
    import numpy as np
    from PIL import Image, features

    if not features.check("libtiff"):
        raise RuntimeError("Pillow was built without libtiff support")
    expected = np.array([[0, 1, 32768, 65534, 65535]], dtype=np.uint16)
    with io.BytesIO() as buffer:
        Image.fromarray(expected).save(buffer, format="TIFF")
        buffer.seek(0)
        with Image.open(buffer) as image:
            actual = np.array(image)
    if actual.dtype.kind != "u" or actual.dtype.itemsize != 2:
        raise RuntimeError(f"TIFF lost its unsigned 16-bit dtype: {actual.dtype}")
    if not np.array_equal(actual, expected):
        raise RuntimeError("16-bit TIFF pixel values changed during the round trip")
    return {"libtiff": True, "dtype": str(actual.dtype), "values": actual.tolist()}


def check_compute(require_cuda=False):
    import torch

    cuda_available = torch.cuda.is_available()
    if require_cuda and not cuda_available:
        raise RuntimeError(f"CUDA is unavailable; torch.version.cuda={torch.version.cuda}")
    device = torch.device("cuda:0" if cuda_available else "cpu")
    values = torch.ones((2, 2), dtype=torch.float32, device=device)
    actual = values @ values
    if cuda_available:
        torch.cuda.synchronize(device)
    if not torch.allclose(actual, torch.full_like(actual, 2.0)):
        raise RuntimeError("Tensor matrix multiplication returned unexpected values")
    details = {
        "torch_version": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "device": str(device),
        "matrix_result": actual.cpu().tolist(),
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(device)
        details.update({
            "name": properties.name,
            "memory_gib": round(properties.total_memory / 1024**3, 2),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        })
    return details


def collect_report(require_cuda=False):
    checks = [run_check("conda_interpreter", check_interpreter)]
    for distribution, module_name in DEPENDENCIES.items():
        checks.append(run_check(
            f"dependency:{distribution}",
            lambda distribution=distribution, module_name=module_name:
                check_dependency(distribution, module_name),
        ))
    checks.append(run_check("uint16_tiff_roundtrip", check_tiff))
    checks.append(run_check(
        "tensor_matrix_multiplication", lambda: check_compute(require_cuda=require_cuda),
    ))
    return {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(check["status"] == "pass" for check in checks) else "fail",
        "require_cuda": require_cuda,
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "platform": platform.platform(),
            "working_directory": str(Path.cwd()),
            "project_root": str(PROJECT_ROOT),
        },
        "checks": checks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda", action="store_true",
        help="Fail unless CUDA is available and a GPU tensor calculation succeeds",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs" / "checks" / "environment.json",
        help="JSON report path; relative paths are resolved from the current directory",
    )
    args = parser.parse_args(argv)
    report = collect_report(require_cuda=args.require_cuda)
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    except OSError as error:
        parser.exit(2, f"Cannot save environment report: {error}\n")
    for check in report["checks"]:
        details = check.get("error", "OK")
        print(f"[{check['status'].upper()}] {check['name']}: {details}")
    print(f"Report: {args.output.resolve()}")
    print(f"Overall: {report['status'].upper()}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
