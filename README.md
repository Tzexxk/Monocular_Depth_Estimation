# Monocular Endoscopic Depth Estimation

A Python/PyTorch project comparing a constant-median-depth baseline, calibrated Depth Anything V2 Relative Small, and residual depth refinement on the Colonoscopy 3D Video Dataset (C3VD).

The repository includes data preparation, a shared Dataset, baseline fitting, model inference/export commands, strict evaluation, and refinement training code. Existing returned predictions have been re-evaluated with the integrated evaluator. A one-frame integrated CUDA C1 forward using strict FP32 and a returned relative-depth input passed the predefined tolerance on all valid GT pixels. The pinned official DA forward also completed on one development frame, with and without TF32. Both runs passed the relative-output valid-region comparison, but neither calibrated millimeter output passed the predefined valid-region tolerance. Full-grid and complete-split numerical replay remain unverified. Full pretrained/CNN inference and training have **not** been validated end to end in the current Windows environment.

## Installation

Run commands from the repository root. The verified base environment is Windows x64, Python 3.11.16, PyTorch 2.14.0+cu132 and torchvision 0.29.0+cu132. Keep these builds together. `requirements.txt` pins direct dependencies and their Windows runtime closure; `pyproject.toml` reads that same file.

```powershell
conda create -n c3vd-depth python=3.11.16 pip -y
conda activate c3vd-depth
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python scripts/check_environment.py
```

`c3vd-depth` is only an example environment name. A suitable existing environment can be used. All 26 installed version pins were checked in the integration environment. OpenCV 4.13.0.92 and the other direct dependencies import successfully; the uint16 TIFF round trip and required CUDA tensor check passed on 2026-11-01. A clean reinstall, other platforms and other PyTorch/CUDA builds have not been validated. Requirements do not lock native Conda libraries or GPU drivers.

For GPU execution, run `python scripts/check_environment.py --require-cuda`. This checks a small tensor operation, not model inference. The environment check includes OpenCV and the integrated direct dependencies.

## Data preparation

Obtain `trans_t1_a.zip` and `trans_t2_b.zip` from the [official C3VD Registered Videos release](https://durrlab.github.io/C3VD/) and place them in `data/downloads/`. Keep their original filenames. Dataset use is subject to CC BY-NC-SA 4.0.

```powershell
python scripts/prepare_data.py
python scripts/check_dataloader.py
```

Preparation verifies archive CRCs, pairs RGB PNGs with uint16 depth TIFFs, preserves original bytes, and writes portable manifests, audit reports and development previews. Both commands should print `Overall: PASS`.

| Split | Sequence | Frames | Original height × width |
| --- | --- | ---: | --- |
| Development | trans_t1_a | 61 | 1080 × 1350 |
| Test | trans_t2_b | 103 | 1080 × 1350 |

The shared interface is:

```python
from c3vd_data.dataset import C3VDDataset
sample = C3VDDataset(data_root="data", split="test")[0]
# image: float32 [3,H,W], RGB in [0,1]
# depth_mm: float32 [H,W]; valid_mask: bool [H,W]
# sample_id: "trans_t2_b/0000"; sequence: "trans_t2_b"
```

GT is decoded as `float32(raw) * float32(100 / 65535)` millimeters. The mask is `0 < raw < 65535`; invalid depth values remain in the depth array. The Dataset does not resize, crop, augment or normalize for a model. Evaluation reads the same manifests and decoder without requiring tensor loading.

## Constant-median baseline

Fit one exact median from **all valid pixels pooled across all 61 development frames**, freeze it, then export and evaluate all 103 test frames:

```powershell
python scripts/run_baseline.py
```

The fitting artifact is saved before test GT is read. The returned development constant is 17.259479522705078 mm (uint16 median 11311). To reuse it without fitting again, the following command checks the development manifest and all 61 GT hashes first:

```powershell
python scripts/run_baseline.py --constant-config configs/constant_median.json --output-root outputs/frozen_baseline
```

Both paths write `predictions/constant_median/trans_t2_b/*.npy`, `results/constant_median/{per_frame.csv,summary.json,config.json,figures/}` and a frozen calibration JSON under the chosen output root. Existing output paths are refused. Optional baseline figures can be generated after a completed run:

```powershell
python scripts/build_baseline_materials.py --result-dir outputs/results/constant_median
```

## Pretrained model and weights

Use the [official Depth Anything V2 implementation](https://github.com/DepthAnything/Depth-Anything-V2), pinned to the experiment's source revision:

```powershell
git clone https://github.com/DepthAnything/Depth-Anything-V2.git weights/Depth-Anything-V2
git -C weights/Depth-Anything-V2 checkout a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
New-Item -ItemType Directory -Path weights -Force | Out-Null
Invoke-WebRequest -Uri 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth' -OutFile 'weights/depth_anything_v2_vits.pth'
Get-FileHash weights/depth_anything_v2_vits.pth -Algorithm SHA256
```

Expected SHA-256: `715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378`. The inference command checks the revision, tracked source modifications and checkpoint hash. Keep upstream license files. Small is released under Apache-2.0; see [resource notices](THIRD_PARTY.md).

Inference uses the official `infer_image`: aspect-preserving resize with a 518-pixel lower bound and multiples of 14, OpenCV cubic interpolation, and ImageNet mean/std normalization. Relative output is restored to the full original grid using **bilinear interpolation with `align_corners=True`**. Refinement's internal upsampling separately retains `False`. GT and masks are never resized.

The following are explicit full inference commands; they have not been executed as part of integration:

```powershell
python scripts/infer_pretrained.py --split dev
python scripts/infer_pretrained.py --split test
python scripts/export_predictions.py --method da_v2 --split test
python scripts/evaluate_predictions.py --predictions outputs/predictions/da_v2 --output-dir outputs/results/da_v2 --method da_v2 --metadata outputs/predictions/da_v2/generation.json
```

Inference selects CUDA, MPS or CPU as the official implementation does. The optional `--strict-fp32` flag disables cuDNN/matmul TF32 and records the resulting numerical settings; the default preserves the official path. The optional xFormers fallback completed one-frame CUDA checks without installing xFormers. Disabling TF32 did not resolve the DA calibrated-depth mismatch, so no DA numerical-reproduction claim is made. Use a new `--output-root` if the default contains existing artifacts. For a small inference check, use `--max-frames 1 --output-root outputs/verification_da`; a partial export cannot pass complete-split evaluation.

## Scale calibration and residual refinement

The default frozen mapping is:

```text
depth_mm = 1 / (a * relative + b)
a = 0.014460004298744292
b = 0.007217614873792536
```

`configs/inverse_calibration_clean.json` records fitting on development frames 0000–0039, validation on 0050–0060, 5,000 sampled pixels per training frame and seed 42. Original sample arrays and their generation code were not fully preserved. The provided parameters reproduce the archived relative-to-millimeter conversion, but exact regeneration of the original fit has not been established. They are retained as frozen inputs; refitting with `c3vd_models.calibration.fit_inverse_calibration` creates a new experiment.

Four returned residual architectures are available: Simple CNN, MultiScale, gated RGB/depth, and highlight-aware gated refinement. The last two share a 485,841-parameter architecture. The highlight-aware model uses HSV value >=220 and saturation <=100, followed by a 5×5 elliptical opening and dilation; its control uses a zero highlight channel. Model inputs use depth/100, and output is multiplied by 100. Outputs are not clipped. Refinement export retains cuDNN deterministic mode from the original Notebook; `--strict-fp32` additionally disables cuDNN/matmul TF32 and uses the highest float32 matmul precision, recording the actual settings in generation.json. The flag does not change weights, masks or calibration.

Returned trained refinement weights are not distributed with this public repository and currently have no public release URL. To use a separately obtained checkpoint, record its hash and run:

```powershell
python scripts/export_predictions.py --method highlight_c1 --checkpoint checkpoints/highlight_c1/seed_123_best.pth --device cuda --strict-fp32
python scripts/evaluate_predictions.py --predictions outputs/predictions/highlight_c1 --output-dir outputs/results/highlight_c1 --method highlight_c1 --metadata outputs/predictions/highlight_c1/generation.json
```

To produce new weights from prepared data and development relative predictions:

```powershell
python scripts/train_refinement.py --model highlight_c1 --seed 123 --output-root outputs/new_training --device cuda
```

This invokes the contributed AdamW/masked-L1 training loop: learning rate and weight decay 1e-4, batch size 1, maximum 30 epochs and patience 6. It uses development frames 0000–0039 for training and 0050–0060 for validation, with the original checkpoint-improvement threshold of 1e-5. This entry point has not been trained end to end after integration; do not label newly produced weights as the archived checkpoint. Existing checkpoint/output paths are refused.

## Strict evaluation

```powershell
python scripts/evaluate_predictions.py --predictions outputs/predictions/da_v2 --output-dir outputs/results/da_v2_recheck --method da_v2 --metadata configs/da_v2.json
```

The static metadata files describe the archived runs. For newly generated predictions, use their `generation.json`, which records actual input/output and checkpoint hashes.

The evaluator requires a complete canonical split, with float32 `[1080,1350]` millimeter predictions at `SEQUENCE/FRAME_ID.npy`. Missing/extra predictions, malformed geometry and nonfinite/nonpositive predictions at **any valid GT pixel** fail evaluation. It never drops failed pixels or frames. CSV fields are `sample_id,absrel,rmse_mm,delta1,valid_pixels`.

| Metric | Per-frame definition over the original valid GT mask |
| --- | --- |
| AbsRel | mean(abs(pred−gt)/gt) |
| RMSE (mm) | sqrt(mean((pred−gt)²)) |
| Delta < 1.25 | mean(max(pred/gt,gt/pred) < 1.25), as a fraction |

Sequence scores are arithmetic means of the 103 per-frame scores. Evaluation uses the shared original `depth_mm`, without a `/100*100` round trip. It does not fit scale, clip predictions or align against test GT.

## Results and verification limits

The following values were verified from existing full-resolution predictions using the integrated evaluator on all 103 test frames and 136,655,314 valid pixels. Baseline arrays were reconstructed from the frozen development constant; model arrays were returned artifacts. **These are existing-prediction checks, not results of newly executed integrated model inference.**

| Method | AbsRel ↓ | RMSE (mm) ↓ | Delta < 1.25 ↑ |
| --- | ---: | ---: | ---: |
| Constant median | 0.549902 | 17.691578 | 0.275154 |
| Calibrated DA V2 Relative Small | 0.311751 | 6.011795 | 0.558338 |
| Highlight-aware gated refinement, seed 123 | 0.211726 | 4.823709 | 0.659201 |

Archived model scores used a float32 GT normalization round trip. Original files are preserved separately; the new full-precision scores differ slightly, although the six-decimal table above is unchanged. Exact original calibration sampling and complete historical environment reproduction remain unverified. A one-frame CUDA C1 replay completed with a maximum difference of 0.02624 mm and mean difference of 0.00421 mm against the returned prediction; it failed the predefined rtol=1e-4, atol=0.001 mm check. Aligning the Notebook's cuDNN deterministic setting alone produced an output identical to the first local replay. Disabling TF32 reduced mean/max differences to 0.000242/0.003029 mm. All 1,326,399 valid GT pixels then passed the unchanged tolerance; the 52 full-grid tolerance failures were all in the pre-existing GT-invalid region. Use `--strict-fp32` for these refinement precision settings. This verifies one frame on the valid GT region, not full-grid equivalence or all 103 frames; the scores above remain checks of the returned arrays.

Simple CNN, MultiScale and gated-control test artifacts contain nonpositive predictions in 27, 8 and 44 frames respectively. Their reported floor-based diagnostic metrics and valid-frame-only means are supplementary results, not complete strict scores, and are excluded from this table. Exploratory gradient, boundary and highlight-region analyses have different masks/definitions and remain separate.

Only one development and one test sequence are used. Neighboring video frames are correlated; these scores are not evidence of clinical or cross-device generalization.

## Layout and checks

```text
configs/                    # Public frozen parameters and archived-run metadata
src/c3vd_data/              # Shared manifests, decoding and Dataset
src/c3vd_eval/              # Pooled median and strict evaluation
src/c3vd_models/            # DA adapter, residual models, training and analysis
scripts/                   # Data, baseline, inference, export and evaluation entry points
tests/                     # Synthetic tests; no weights or data download required
data/                      # Ignored archives, raw images and manifests
weights/                   # Ignored pretrained source checkout and weights
checkpoints/               # Ignored trained refinement weights
outputs/predictions/       # Ignored METHOD/SEQUENCE/FRAME_ID.npy
outputs/results/           # Ignored per-frame metrics, summary, config and figures
outputs/intermediate/      # Ignored relative predictions
outputs/history/           # Ignored unmodified returned results and diagnostics
```

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/build_method_figures.py --output-dir outputs/figures
```

The synthetic suite covers data decoding, manifests, evaluation boundaries, full prediction-set validation, original-GT evaluation, checkpoint structure and overwrite protection. OpenCV import and the required CUDA tensor check have passed. The pinned official DA source was acquired and one development-frame forward completed. Cross-environment calibrated numerical reproduction, complete pretrained/CNN inference, training and a fresh installation still require separate execution checks. Private working documents are not required to run the public commands.

## References

- Bobrow et al., C3VD, Medical Image Analysis 2023. [Paper](https://doi.org/10.1016/j.media.2023.102956), [data](https://durrlab.github.io/C3VD/).
- Yang et al., Depth Anything V2, NeurIPS 2024. [Paper](https://arxiv.org/abs/2406.09414), [implementation](https://github.com/DepthAnything/Depth-Anything-V2).
- Ronneberger et al., U-Net, MICCAI 2015. [Paper](https://arxiv.org/abs/1505.04597).
- [Third-party resources and attribution](THIRD_PARTY.md).
