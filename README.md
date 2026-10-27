# Monocular Endoscopic Depth Estimation

A Python/PyTorch project for monocular depth estimation on the Colonoscopy 3D Video Dataset (C3VD). The planned experiment compares a constant-median-depth baseline with a released general-purpose monocular depth model using paired endoscopic RGB images and ground-truth depth.

The current implementation provides data preparation, a shared PyTorch Dataset, and validation tools. Model inference and metric computation are still under development.

## Project status

| Component | Status |
| --- | --- |
| C3VD archive audit and RGB/depth pairing | Implemented and validated |
| Depth decoding and valid-pixel masks | Implemented and validated |
| Shared Dataset and complete DataLoader traversal | Implemented and validated |
| Synthetic checks | 12 checks available |
| Constant-median-depth baseline | Pending |
| Pretrained monocular depth model | Pending |
| Metric implementation and comparative experiments | Pending |

## Installation

Clone or download this repository and run the following commands from its root directory.

The verified dependency snapshot targets **Windows x64, Python 3.11.16, and the PyTorch CUDA 13.2 build**. Exact direct and transitive package versions are recorded in [requirements.txt](requirements.txt). Other platforms or PyTorch builds require a separately validated environment specification.

```shell
conda create -n c3vd-depth python=3.11.16 pip -y
conda activate c3vd-depth
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132
python -m pip check
```

`c3vd-depth` is an example name; any conda environment name is supported. Skip environment creation if a suitable environment already exists. The extra package index supplies the pinned `torch==2.14.0+cu132` and `torchvision==0.29.0+cu132` builds.

Verify the environment:

```shell
python scripts/check_environment.py
```

The default check uses CUDA when available and otherwise checks CPU tensor execution. To require successful GPU execution:

```shell
python scripts/check_environment.py --require-cuda
```

A successful check prints `Overall: PASS` and writes `outputs/checks/environment.json`.

## Dataset preparation

Download the following registered-video archives from the [official C3VD release](https://durrlab.github.io/C3VD/). These two sequences define this project's benchmark split.

| Split | Sequence | Paired frames | Resolution, width x height |
| --- | --- | --- | --- |
| Development | `trans_t1_a` | 61 | 1350 x 1080 |
| Test | `trans_t2_b` | 103 | 1350 x 1080 |

Keep the original filenames and place the ZIP files as follows. Manual extraction is unnecessary.

```text
data/
└── downloads/
    ├── trans_t1_a.zip
    └── trans_t2_b.zip
```

Run the archive and image audit:

```shell
python scripts/prepare_data.py
```

The script checks ZIP CRCs, pairs RGB/depth files by frame ID, validates image types and dimensions, and preserves the extracted image bytes. It produces:

- Original RGB/depth pairs in `data/raw/<sequence>/`.
- Portable manifests at `data/manifests/dev.csv` and `data/manifests/test.csv`.
- Audit reports at `outputs/checks/data_audit.json` and `outputs/checks/data_frames.csv`.
- Three development-set previews in `outputs/previews/`.

After data preparation passes, validate a complete traversal of both splits:

```shell
python scripts/check_dataloader.py
```

Both scripts should finish with `Overall: PASS`. The traversal report is saved to `outputs/checks/dataloader.json`. Both scripts support `--data-root` and `--output-root`; use matching paths for the two stages. After relocating prepared data, rerun preparation to refresh the local audit paths before running the traversal check.

The dataset, downloaded archives, model weights, and generated run outputs are excluded from version control.

## Using the shared Dataset

Repository scripts locate the source directory automatically. To import the Dataset from your own Python code or notebooks, register the source package after installing the dependencies:

```shell
python -m pip install --no-deps -e .
```

```python
from torch.utils.data import DataLoader
from c3vd_data.dataset import C3VDDataset

dataset = C3VDDataset(data_root="data", split="dev")  # "dev" or "test"
loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
batch = next(iter(loader))

print(batch["sample_id"][0])
print(batch["image"].shape)       # [1, 3, 1080, 1350]
print(batch["depth_mm"].shape)    # [1, 1080, 1350]
print(batch["valid_mask"].dtype)  # torch.bool
```

Each Dataset sample has the following fields. DataLoader collation adds a leading batch dimension to tensors and groups string fields into lists.

| Field | Type and shape | Meaning |
| --- | --- | --- |
| `image` | float32 Tensor, `[3,H,W]` | RGB values in `[0,1]` |
| `depth_mm` | float32 Tensor, `[H,W]` | Camera z-depth in millimeters |
| `valid_mask` | bool Tensor, `[H,W]` | Valid GT pixels for the project protocol |
| `sample_id` | string | Sequence and frame identifier, e.g. `trans_t1_a/0000` |
| `sequence` | string | Source sequence name |

Unsigned 16-bit ground truth is decoded as `depth_mm = raw / 65535 * 100`. The project's mask is `(raw > 0) & (raw < 65535)`, excluding zero and saturated depth values. Invalid depth values remain in `depth_mm`; use the mask when evaluating predictions.

The shared Dataset preserves the original image geometry and performs no augmentation or model-specific normalization. Model preprocessing belongs in the inference code. Default loading uses CPU tensors, batch size 1, zero worker processes, and sequential sampling.

## Experimental protocol

The planned comparison uses one constant median fitted from all valid development-set GT pixels and a released monocular depth model. Model selection and any scale calibration use development data; the primary test protocol freezes these choices before evaluation.

Both methods will be evaluated on all 103 test frames at the original spatial resolution with the same GT mask. The per-frame metrics are:

| Metric | Definition over valid pixels | Unit |
| --- | --- | --- |
| AbsRel | `mean(abs(pred - gt) / gt)` | Dimensionless |
| RMSE | `sqrt(mean((pred - gt) ** 2))` | mm |
| Delta < 1.25 | `mean(max(pred / gt, gt / pred) < 1.25)` | Fraction |

Each sequence score will be the arithmetic mean of its per-frame scores. Predictions must be finite and positive at every valid GT pixel. Model predictions must be restored to the original image grid before evaluation; the GT and mask retain their original resolution. Any supplementary test-GT-based alignment must be reported separately from the primary comparison.

## Model inference

## Evaluation commands

## Results

| Method | AbsRel | RMSE (mm) | Delta < 1.25 |
| --- | --- | --- | --- |
| Constant median | | | |
| Pretrained model | | | |

## Tests

```shell
python -m unittest discover -s tests -p 'test_*.py' -v
```

The 12 synthetic checks cover report generation, uint16 depth boundaries, frame pairing, geometry, sample values, manifest validation, and DataLoader behavior. They use temporary images and simulated environment checks; real-data validation is performed by the preparation and traversal scripts.

## Repository structure

```text
.
├── scripts/
│   ├── check_environment.py
│   ├── prepare_data.py
│   ├── check_dataloader.py
│   └── build_method_figures.py
├── src/c3vd_data/
│   ├── io.py
│   └── dataset.py
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

`pyproject.toml` reads the same dependency list as `requirements.txt` and declares the Python version and package layout. Workflow diagrams can be generated with the standard-library-only helper:

```shell
python scripts/build_method_figures.py --output-dir outputs/figures
```

## Reproducibility

The version pins were checked against the environment used to validate the data pipeline. They cover the project's Python dependency closure on the verified Windows platform. Conda native libraries, GPU drivers, interpreter builds, and wheel hashes are not fully locked by this file, and a fresh-environment reinstall has not yet been validated.

The audit reports record archive and manifest SHA-256 digests, image dimensions, valid-pixel counts, environment versions, and loader settings. Final model experiments should also record the code revision, weights and their hashes, preprocessing, calibration parameters, random seeds, device, and evaluation configuration.

## References

- Bobrow, T. L., et al. *Colonoscopy 3D video dataset with paired depth from 2D-3D registration*. Medical Image Analysis, 2023, 102956. [Paper](https://doi.org/10.1016/j.media.2023.102956).
- [C3VD official dataset and format documentation](https://durrlab.github.io/C3VD/). Dataset use is subject to its CC BY-NC-SA 4.0 license.
