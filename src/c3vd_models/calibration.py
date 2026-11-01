import numpy as np
from scipy.optimize import least_squares

def apply_inverse_calibration(relative, a, b):
    denom = a * np.asarray(relative, dtype=np.float64) + b
    if not np.isfinite(denom).all() or np.any(denom <= 0):
        raise ValueError('Invalid inverse calibration denominator')
    return 1.0 / denom

def fit_inverse_calibration(relative_samples, gt_mm_samples):
    r = np.asarray(relative_samples, dtype=np.float64)
    gt = np.asarray(gt_mm_samples, dtype=np.float64)
    def residual(params):
        return 1.0 / (params[0] * r + params[1]) - gt
    fit = least_squares(residual, x0=[0.014,0.008], bounds=([1e-8,1e-8],[1.,1.]), method='trf', max_nfev=300, ftol=1e-10, xtol=1e-10)
    if not fit.success: raise RuntimeError(fit.message)
    return float(fit.x[0]), float(fit.x[1])
