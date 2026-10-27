"""Shared C3VD split definitions and lossless uint16 depth reading."""

import numpy as np
from PIL import Image


SPLITS = {"dev": ("trans_t1_a", 61), "test": ("trans_t2_b", 103)}


def decode_depth(raw):
    """Return float32 millimeter z-depth and the project's valid pixel mask."""
    if raw.ndim != 2 or raw.dtype.kind != "u" or raw.dtype.itemsize != 2:
        raise ValueError(f"Expected a single-channel uint16 depth image; got {raw.shape}, {raw.dtype}")
    depth_mm = raw.astype(np.float32) * np.float32(100.0 / 65535.0)
    valid_mask = (raw > 0) & (raw < 65535)
    return depth_mm, valid_mask


def read_pair(rgb_path, depth_path):
    with Image.open(rgb_path) as image:
        if image.format != "PNG" or image.mode not in {"RGB", "RGBA"}:
            raise ValueError(f"Expected a color PNG: {rgb_path}; got {image.format}, {image.mode}")
        rgb = np.array(image.convert("RGB"))
    with Image.open(depth_path) as image:
        if image.format != "TIFF":
            raise ValueError(f"Expected a TIFF depth image: {depth_path}")
        raw = np.array(image)
    try:
        depth_mm, valid_mask = decode_depth(raw)
    except ValueError as error:
        raise ValueError(f"{depth_path}: {error}") from error
    if rgb.shape[:2] != raw.shape:
        raise ValueError(f"RGB/depth size mismatch: {rgb_path.name} {rgb.shape}, {raw.shape}")
    if not valid_mask.any():
        raise ValueError(f"No valid depth pixels: {depth_path}")
    return rgb, raw, depth_mm, valid_mask
