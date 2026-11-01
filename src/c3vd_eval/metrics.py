"""Original-grid, GT-mask-only depth metrics, with equal weight per frame."""
import numpy as np

METRICS = ('absrel', 'rmse_mm', 'delta1')

def depth_metrics(prediction, depth_mm, valid_mask):
    gt = np.asarray(depth_mm)
    mask = np.asarray(valid_mask)
    pred = np.asarray(prediction)
    if gt.ndim != 2 or mask.shape != gt.shape or mask.dtype != np.bool_:
        raise ValueError('GT must be HxW and mask must be a matching boolean array')
    if pred.shape != gt.shape:
        raise ValueError(f'Prediction shape {pred.shape} differs from original GT {gt.shape}')
    if not mask.any():
        raise ValueError('No valid GT pixels')
    g, p = gt[mask].astype(np.float64), pred[mask].astype(np.float64)
    if not np.isfinite(g).all() or (g <= 0).any():
        raise ValueError('Valid GT must be finite and positive')
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError('Prediction must be finite and positive at ALL valid GT pixels')
    error = p - g
    return {'absrel': float(np.mean(np.abs(error) / g)),
            'rmse_mm': float(np.sqrt(np.mean(error * error))),
            'delta1': float(np.mean(np.maximum(p / g, g / p) < 1.25)),
            'valid_pixels': int(g.size)}

def summarize(rows):
    if not rows:
        raise ValueError('No frames to summarize')
    return {'frames': len(rows), 'valid_pixels': sum(r['valid_pixels'] for r in rows),
            'aggregation': 'arithmetic mean of per-frame metrics',
            **{k: float(np.mean([r[k] for r in rows])) for k in METRICS}}
