import numpy as np
import cv2

def depth_gradients_mm(depth):
    d = np.asarray(depth, dtype=np.float32)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.hypot(gx, gy)


def disk_kernel(radius):
    size = 2 * radius + 1
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (size, size)
    )


def highlight_boundary_band(highlight, radius=5):
    h = highlight.astype(np.uint8)
    kernel = disk_kernel(radius)

    outer = cv2.dilate(h, kernel).astype(bool)
    inner = cv2.erode(h, kernel).astype(bool)

    return outer & ~inner


def sobel_xy(depth):
    d = np.asarray(depth, dtype=np.float32)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return gx, gy


def regional_metrics(pred, gt, region):
    n = int(region.sum())
    if n == 0:
        return {
            "pixels": 0,
            "mae_mm": np.nan,
            "gradient_error": np.nan
        }

    abs_error = np.abs(pred - gt)

    px, py = sobel_xy(pred)
    gx, gy = sobel_xy(gt)

    grad_error = np.hypot(px - gx, py - gy)

    return {
        "pixels": n,
        "mae_mm": float(abs_error[region].mean()),
        "gradient_error": float(grad_error[region].mean())
    }

