import cv2
import numpy as np

from homography_snapshot import warp_with_balance


class AdvancedStitcher:
    """Feature-based stitcher that refreshes the homography every N frames."""

    def __init__(
        self,
        feature_interval: int = 30,
        max_features: int = 1200,
        good_match_percent: float = 0.15,
        roi_fraction: float = 0.2,
        detect_downscale: float = 0.5,
        detector_type: str = "ORB",
        matcher_type: str = "BF",
        use_optical_flow: bool = False,
        stability: float = 0.0,
        balance_mode: int = 2,
    ):
        self.feature_interval = max(1, int(feature_interval))
        self.max_features = max_features
        self.good_match_percent = good_match_percent
        self.roi_fraction = float(min(1.0, max(0.05, roi_fraction)))
        self.detect_downscale = float(min(1.0, max(0.1, detect_downscale)))
        self.detector_type = detector_type.upper()
        self.matcher_type = matcher_type.upper()
        self.use_optical_flow = bool(use_optical_flow)
        self.stability = float(min(1.0, max(0.0, stability)))
        self.balance_mode = int(np.clip(balance_mode, 0, 2))

        self.frame_count = 0
        self.homography = None
        self.debug_last = {}
        self.prev_left_gray = None
        self.prev_right_gray = None
        self.prev_pts_left = None
        self.prev_pts_right = None
        self._last_scale = None

    def reset_state(self, reset_frame_count: bool = False):
        self.homography = None
        self.debug_last = {}
        self.prev_left_gray = None
        self.prev_right_gray = None
        self.prev_pts_left = None
        self.prev_pts_right = None
        if reset_frame_count:
            self.frame_count = 0

    def stitch(self, left, right):
        """Returns (stitched_frame, debug_info). stitched_frame is None on failure."""
        self.frame_count += 1
        debug_info = dict(self.debug_last)
        recomputed = False
        recompute_failed = False
        flow_used = False
        flow_failed = False
        update_source = "reuse"

        scale = float(min(1.0, max(0.1, self.detect_downscale)))
        if self._last_scale is None or self._last_scale != scale:
            self.prev_left_gray = None
            self.prev_right_gray = None
            self.prev_pts_left = None
            self.prev_pts_right = None
            self._last_scale = scale

        need_recompute = self.homography is None or self.frame_count % self.feature_interval == 0

        left_gray = None
        right_gray = None
        if self.use_optical_flow and not need_recompute:
            left_gray = self._to_gray_scaled(left, scale)
            right_gray = self._to_gray_scaled(right, scale)
            if (
                self.prev_left_gray is not None
                and self.prev_right_gray is not None
                and self.prev_pts_left is not None
                and self.prev_pts_right is not None
            ):
                H_flow, flow_debug, pts_left, pts_right = self._compute_homography_from_flow(
                    self.prev_left_gray,
                    self.prev_right_gray,
                    left_gray,
                    right_gray,
                    self.prev_pts_left,
                    self.prev_pts_right,
                    scale,
                )
                debug_info = dict(flow_debug)
                if H_flow is not None:
                    update_source = "flow"
                    flow_used = True
                    self._update_homography(H_flow)
                    self.prev_left_gray = left_gray
                    self.prev_right_gray = right_gray
                    self.prev_pts_left = pts_left
                    self.prev_pts_right = pts_right
                else:
                    flow_failed = True
                    self.prev_left_gray = None
                    self.prev_right_gray = None
                    self.prev_pts_left = None
                    self.prev_pts_right = None

        if need_recompute or self.homography is None:
            H_feat, feat_debug, pts_left, pts_right = self._compute_homography(left, right)
            debug_info = dict(feat_debug)
            if H_feat is not None:
                update_source = "features"
                recomputed = True
                self._update_homography(H_feat)
                if self.use_optical_flow:
                    if left_gray is None:
                        left_gray = self._to_gray_scaled(left, scale)
                        right_gray = self._to_gray_scaled(right, scale)
                    self.prev_left_gray = left_gray
                    self.prev_right_gray = right_gray
                    self.prev_pts_left = pts_left
                    self.prev_pts_right = pts_right
            else:
                if self.homography is None:
                    info = {"homography_found": False}
                    info.update(feat_debug)
                    return None, info
                recompute_failed = True

        stitched = self._warp_and_blend(left, right, self.homography)
        debug_info.update(
            {
                "homography_found": self.homography is not None,
                "recomputed": recomputed,
                "recompute_failed_kept_previous": recompute_failed,
                "flow_used": flow_used,
                "flow_failed": flow_failed,
                "update_source": update_source,
                "stability": self.stability,
            }
        )
        self.debug_last = debug_info
        return stitched, debug_info

    def _update_homography(self, H_new):
        H_new = self._normalize_homography(H_new)
        if self.homography is None or self.stability <= 0.0:
            self.homography = H_new
            return
        H_prev = self._normalize_homography(self.homography)
        alpha_prev = float(min(1.0, max(0.0, self.stability)))
        alpha_new = 1.0 - alpha_prev
        H = (H_prev * alpha_prev) + (H_new * alpha_new)
        self.homography = self._normalize_homography(H)

    def _normalize_homography(self, H):
        if H is None:
            return None
        if abs(H[2, 2]) < 1e-8:
            return H
        return H / H[2, 2]

    def _to_gray_scaled(self, frame, scale):
        if scale != 1.0:
            new_w = max(2, int(frame.shape[1] * scale))
            new_h = max(2, int(frame.shape[0] * scale))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _compute_homography(self, left, right):
        if left.shape[:2] != right.shape[:2]:
            right = cv2.resize(right, (left.shape[1], left.shape[0]))

        scale = self.detect_downscale
        if scale != 1.0:
            new_w = max(2, int(left.shape[1] * scale))
            new_h = max(2, int(left.shape[0] * scale))
            left_proc = cv2.resize(left, (new_w, new_h), interpolation=cv2.INTER_AREA)
            right_proc = cv2.resize(right, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            left_proc, right_proc = left, right

        left_roi, left_offset = self._crop_seam_band(left_proc, self.roi_fraction, side="right")
        right_roi, right_offset = self._crop_seam_band(right_proc, self.roi_fraction, side="left")

        gray_left = cv2.cvtColor(left_roi, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_roi, cv2.COLOR_BGR2GRAY)

        detector, det_name, desc_type = self._create_detector()
        kps_left, des_left = detector.detectAndCompute(gray_left, None)
        kps_right, des_right = detector.detectAndCompute(gray_right, None)
        debug = {
            "kp_left": len(kps_left) if kps_left is not None else 0,
            "kp_right": len(kps_right) if kps_right is not None else 0,
            "roi_fraction": self.roi_fraction,
            "detect_downscale": self.detect_downscale,
            "detector": det_name,
            "matcher": self.matcher_type,
        }

        if des_left is None or des_right is None:
            return None, debug, None, None

        matches = self._match_descriptors(des_right, des_left, desc_type)
        if len(matches) < 4:
            debug["matches"] = len(matches)
            return None, debug, None, None

        matches = sorted(matches, key=lambda m: m.distance)
        num_good = max(10, int(len(matches) * self.good_match_percent))
        good = matches[:num_good]
        pts_left = np.float32(
            [
                (
                    (kps_left[m.trainIdx].pt[0] + left_offset[0]) / scale,
                    (kps_left[m.trainIdx].pt[1] + left_offset[1]) / scale,
                )
                for m in good
            ]
        ).reshape(-1, 1, 2)
        pts_right = np.float32(
            [
                (
                    (kps_right[m.queryIdx].pt[0] + right_offset[0]) / scale,
                    (kps_right[m.queryIdx].pt[1] + right_offset[1]) / scale,
                )
                for m in good
            ]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_right, pts_left, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0

        debug.update(
            {
                "matches": len(matches),
                "good_matches": len(good),
                "inliers": inliers,
            }
        )
        return H, debug, pts_left, pts_right

    def _compute_homography_from_flow(
        self,
        prev_left_gray,
        prev_right_gray,
        left_gray,
        right_gray,
        prev_pts_left,
        prev_pts_right,
        scale,
    ):
        debug = {}
        if prev_pts_left is None or prev_pts_right is None:
            return None, debug, None, None

        pts_left_scaled = (prev_pts_left * scale).astype(np.float32)
        pts_right_scaled = (prev_pts_right * scale).astype(np.float32)

        lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        next_left, st_left, err_left = cv2.calcOpticalFlowPyrLK(
            prev_left_gray, left_gray, pts_left_scaled, None, **lk_params
        )
        next_right, st_right, err_right = cv2.calcOpticalFlowPyrLK(
            prev_right_gray, right_gray, pts_right_scaled, None, **lk_params
        )

        if next_left is None or next_right is None:
            return None, debug, None, None

        good_mask = (st_left.flatten() == 1) & (st_right.flatten() == 1)
        if not np.any(good_mask):
            debug["flow_tracked"] = 0
            return None, debug, None, None

        next_left = next_left[good_mask] / scale
        next_right = next_right[good_mask] / scale
        next_left = next_left.reshape(-1, 1, 2)
        next_right = next_right.reshape(-1, 1, 2)

        if len(next_left) < 4:
            debug["flow_tracked"] = len(next_left)
            return None, debug, None, None

        H, mask = cv2.findHomography(next_right, next_left, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0
        debug.update(
            {
                "flow_tracked": len(next_left),
                "flow_inliers": inliers,
            }
        )
        return H, debug, next_left, next_right

    def _create_detector(self):
        name = self.detector_type.upper()
        if name == "SIFT":
            if hasattr(cv2, "SIFT_create"):
                return cv2.SIFT_create(nfeatures=self.max_features), "SIFT", "float"
            name = "ORB"
        if name == "AKAZE":
            return cv2.AKAZE_create(), "AKAZE", "binary"
        return cv2.ORB_create(self.max_features), "ORB", "binary"

    def _match_descriptors(self, des_right, des_left, desc_type):
        use_flann = self.matcher_type.upper() == "FLANN"
        if desc_type == "float":
            norm = cv2.NORM_L2
        else:
            norm = cv2.NORM_HAMMING

        if use_flann:
            if desc_type == "float":
                index_params = dict(algorithm=1, trees=5)
                des_right = np.float32(des_right)
                des_left = np.float32(des_left)
            else:
                index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            raw = matcher.knnMatch(des_right, des_left, k=2)
            good = []
            for pair in raw:
                if len(pair) != 2:
                    continue
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            return good

        matcher = cv2.BFMatcher(norm, crossCheck=True)
        return matcher.match(des_right, des_left)

    def _crop_seam_band(self, img, fraction, side="left"):
        """Returns a vertical band at the seam side and its (x_offset, y_offset)."""
        h, w = img.shape[:2]
        band_w = max(16, int(w * fraction))
        if side == "right":
            x0 = max(0, w - band_w)
            x1 = w
        else:
            x0 = 0
            x1 = min(w, band_w)
        return img[:, x0:x1], (x0, 0)

    def _warp_and_blend(self, left, right, H):
        try:
            return warp_with_balance(left, right, H, self.balance_mode)
        except RuntimeError:
            return warp_with_balance(left, right, H, 2)
