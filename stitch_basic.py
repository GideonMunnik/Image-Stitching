import cv2
import numpy as np


def stitch_basic(left, right, overlap_fraction: float = 0.2):
    """Stitches two frames with a fixed overlap using simple alpha blending."""
    overlap_fraction = float(np.clip(overlap_fraction, 0.0, 0.9))

    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))

    h, w = left.shape[:2]
    overlap_px = max(1, int(w * overlap_fraction))

    # Split sections
    left_main = left[:, : w - overlap_px]
    right_main = right[:, overlap_px:]
    left_overlap = left[:, w - overlap_px :]
    right_overlap = right[:, :overlap_px]

    # Linear blend across the overlap
    alpha = np.linspace(0, 1, overlap_px, dtype=np.float32)
    alpha = np.tile(alpha, (h, 1))
    alpha = alpha[..., None]
    blend = (left_overlap * (1 - alpha) + right_overlap * alpha).astype(np.uint8)

    stitched = np.hstack((left_main, blend, right_main))
    return stitched
