"""Reuse shared lossless decoding and manifests without requiring PyTorch."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from c3vd_data.io import decode_depth
from c3vd_data.manifest import load_manifest as shared_manifest

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def load_manifest(data_root, split):
    return shared_manifest(data_root, split, complete=True)


def read_depth(root, row):
    with Image.open(Path(root) / row['depth_path']) as im:
        if im.format != 'TIFF':
            raise ValueError('Depth must be TIFF')
        raw = np.array(im)
    depth, mask = decode_depth(raw)
    if raw.shape != (1080, 1350):
        raise ValueError(f'Unexpected original grid: {raw.shape}')
    if not mask.any():
        raise ValueError('Empty valid mask')
    return raw, depth, mask

def histogram_median(hist):
    """Exact median of decoded float32 depths; average both middle values if even."""
    n = int(hist.sum())
    if n == 0:
        raise ValueError('No valid development pixels')
    cdf = hist.cumsum()
    bins = np.searchsorted(cdf, [(n-1)//2 + 1, n//2 + 1])
    values = bins.astype(np.float32) * np.float32(100.0 / 65535.0)
    return float(np.mean(values, dtype=np.float32)), bins.tolist()

def fit_constant(data_root):
    rows = load_manifest(data_root, 'dev')
    hist = np.zeros(65536, dtype=np.int64)
    hashes = {}
    for row in rows:
        raw, _, mask = read_depth(data_root, row)
        hist += np.bincount(raw[mask], minlength=65536)
        hashes[row['sample_id']] = digest(Path(data_root)/row['depth_path'])
    constant, middle = histogram_median(hist)
    return {'method':'constant_median', 'constant_mm':constant,
            'fit_split':'dev', 'fit_sequence':'trans_t1_a', 'fit_frames':len(rows),
            'fit_valid_pixels':int(hist.sum()), 'middle_raw_bins':middle,
            'median_definition':'all valid dev pixels pooled; exact uint16 histogram; decoded float32',
            'dev_manifest_sha256':digest(Path(data_root)/'manifests/dev.csv'),
            'dev_depth_sha256':hashes}, hist
