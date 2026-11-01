import numpy as np
import cv2
EDGE_THRESHOLD=1.0
EDGE_TOLERANCE=2
DIAGNOSTIC_FLOOR_MM=0.01

def sobel_gradient(depth):
    depth = np.asarray(depth, dtype=np.float32)
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return gx, gy


def geometry_metrics(pred, gt, valid, highlight):
    # Exclude stencil positions touching invalid GT pixels.
    kernel = np.ones((3, 3), np.uint8)
    gradient_valid = cv2.erode(
        valid.astype(np.uint8), kernel
    ).astype(bool)

    pgx, pgy = sobel_gradient(pred)
    ggx, ggy = sobel_gradient(gt)

    grad_error = np.sqrt(
        (pgx - ggx)**2 + (pgy - ggy)**2
    )

    p_mag = np.hypot(pgx, pgy)
    g_mag = np.hypot(ggx, ggy)

    pred_edge = (p_mag >= EDGE_THRESHOLD) & gradient_valid
    gt_edge = (g_mag >= EDGE_THRESHOLD) & gradient_valid

    match_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * EDGE_TOLERANCE + 1,) * 2
    )

    gt_dilated = cv2.dilate(
        gt_edge.astype(np.uint8), match_kernel
    ).astype(bool)

    pred_dilated = cv2.dilate(
        pred_edge.astype(np.uint8), match_kernel
    ).astype(bool)

    tp_precision = np.count_nonzero(pred_edge & gt_dilated)
    tp_recall = np.count_nonzero(gt_edge & pred_dilated)

    precision = tp_precision / max(np.count_nonzero(pred_edge), 1)
    recall = tp_recall / max(np.count_nonzero(gt_edge), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    near = gt_dilated & gradient_valid
    far = (~gt_dilated) & gradient_valid

    high = highlight & gradient_valid
    nonhigh = (~highlight) & gradient_valid

    def region_mean(values, region):
        return float(values[region].mean()) if region.any() else np.nan

    return {
        "GradientError": region_mean(grad_error, gradient_valid),
        "BoundaryPrecision": float(precision),
        "BoundaryRecall": float(recall),
        "BoundaryF1": float(f1),
        "HighlightGradientError": region_mean(grad_error, high),
        "NonHighlightGradientError": region_mean(grad_error, nonhigh),
        "NearGTEdgeGradientError": region_mean(grad_error, near),
        "FarGTEdgeGradientError": region_mean(grad_error, far),
        "NearGTEdgeRatio": float(near.sum() / max(gradient_valid.sum(), 1)),
        "PredEdgeCountRatio": float(
            pred_edge.sum() / max(gt_edge.sum(), 1)
        )
    }


def depth_metrics_diagnostic(pred, gt, valid, highlight):
    # Diagnostic floor is disclosed and is NOT a strict result.
    p = np.maximum(pred, DIAGNOSTIC_FLOOR_MM)
    d = gt[valid]
    pv = p[valid]

    error = np.abs(pv - d)

    high = valid & highlight
    low = valid & (~highlight)

    return {
        "AbsRel": float(np.mean(error / d)),
        "RMSE_mm": float(np.sqrt(np.mean((pv - d)**2))),
        "Delta1": float(np.mean(np.maximum(pv/d, d/pv) < 1.25)),
        "MAE_mm": float(error.mean()),
        "HighlightMAE": float(np.abs(p[high] - gt[high]).mean())
            if high.any() else np.nan,
        "NonHighlightMAE": float(np.abs(p[low] - gt[low]).mean())
            if low.any() else np.nan
    }

