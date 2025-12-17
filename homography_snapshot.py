import cv2
import numpy as np


def _read_first_frame(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read first frame from: {path}")
    return frame


def _crop_roi(img, fraction: float, side: str):
    """Return a vertical seam band and its offset."""
    h, w = img.shape[:2]
    band_w = max(1, int(w * fraction))
    if side == "right":
        x0 = max(0, w - band_w)
        x1 = w
    else:
        x0 = 0
        x1 = min(w, band_w)
    return img[:, x0:x1], (x0, 0)


def warp_with_homography(left, right, H):
    """Warp right into left reference space using a fixed homography and blend overlap."""
    hA, wA = left.shape[:2]
    hB, wB = right.shape[:2]

    if H is None:
        raise RuntimeError("Homography is not available for warping.")

    cornersB = np.float32([[0, 0], [0, hB], [wB, hB], [wB, 0]]).reshape(-1, 1, 2)
    cornersA = np.float32([[0, 0], [0, hA], [wA, hA], [wA, 0]]).reshape(-1, 1, 2)
    warped_cornersB = cv2.perspectiveTransform(cornersB, H)
    all_corners = np.vstack((warped_cornersB, cornersA))

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translate = [-x_min, -y_min]
    translation_mat = np.array([[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]])

    out_width = x_max - x_min
    out_height = y_max - y_min
    warp_right = cv2.warpPerspective(right, translation_mat @ H, (out_width, out_height))
    mask_right = cv2.warpPerspective(np.ones((hB, wB), dtype=np.uint8), translation_mat @ H, (out_width, out_height))

    result = np.zeros_like(warp_right)
    mask_left = np.zeros((out_height, out_width), dtype=np.uint8)
    y_off, x_off = translate[1], translate[0]
    result[y_off : y_off + hA, x_off : x_off + wA] = left
    mask_left[y_off : y_off + hA, x_off : x_off + wA] = 1

    only_right = (mask_left == 0) & (mask_right == 1)
    overlap = (mask_left == 1) & (mask_right == 1)

    result[only_right] = warp_right[only_right]
    if overlap.any():
        result[overlap] = cv2.addWeighted(result[overlap], 0.5, warp_right[overlap], 0.5, 0)
    return result


def save_initial_homography_sample(
    left_video_path: str,
    right_video_path: str,
    output_path: str = "homography_sample.png",
):
    """
    Grab the first frame from each video, compute a homography, and save the stitched result.
    Returns metadata describing the run.
    """
    left = _read_first_frame(left_video_path)
    right = _read_first_frame(right_video_path)

    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))

    roi_fraction = 0.6
    left_roi, left_offset = _crop_roi(left, roi_fraction, side="right")
    right_roi, right_offset = _crop_roi(right, roi_fraction, side="left")

    gray_left = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)

    detector = cv2.SIFT_create()
    kps_left, des_left = detector.detectAndCompute(gray_left, None)
    kps_right, des_right = detector.detectAndCompute(gray_right, None)

    if des_left is None or des_right is None:
        raise RuntimeError("Could not compute descriptors on the initial frames.")

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(des_right, des_left)
    if len(matches) < 4:
        raise RuntimeError(f"Insufficient matches ({len(matches)}) to compute homography.")

    matches = sorted(matches, key=lambda m: m.distance)
    num_good = max(4, int(len(matches) * 0.2))
    good = matches[:num_good]

    pts_left = np.float32(
        [
            (
                kps_left[m.trainIdx].pt[0] + left_offset[0],
                kps_left[m.trainIdx].pt[1] + left_offset[1],
            )
            for m in good
        ]
    ).reshape(-1, 1, 2)
    pts_right = np.float32(
        [
            (
                kps_right[m.queryIdx].pt[0] + right_offset[0],
                kps_right[m.queryIdx].pt[1] + right_offset[1],
            )
            for m in good
        ]
    ).reshape(-1, 1, 2)

    if len(pts_left) < 4:
        raise RuntimeError("Top matches produced fewer than 4 point pairs.")

    H, mask = cv2.findHomography(
        pts_right,
        pts_left,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.999,
    )
    if H is None:
        raise RuntimeError("RANSAC failed to compute a valid homography.")

    stitched = warp_with_homography(left, right, H)
    if not cv2.imwrite(output_path, stitched):
        raise RuntimeError(f"Could not write sample image to {output_path}")

    inliers = int(mask.sum()) if mask is not None else 0
    return {
        "output_path": output_path,
        "total_matches": len(matches),
        "used_matches": len(good),
        "inliers": inliers,
        "roi_fraction": roi_fraction,
        "homography": H,
    }
