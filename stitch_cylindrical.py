import cv2
import numpy as np


class CylindricalStitcher:
    """
    Helper that calibrates two cameras once and keeps the cylindrical warp maps
    so every subsequent frame can be remapped with a single cv2.remap call.
    """

    def __init__(self, downscale: float = 0.5, ratio_thresh: float = 0.75, projection: str = "cylindrical"):
        self.downscale = float(np.clip(downscale, 0.1, 1.0))
        self.ratio_thresh = float(np.clip(ratio_thresh, 0.5, 0.95))
        projection = (projection or "cylindrical").lower()
        if projection not in ("cylindrical", "spherical"):
            raise ValueError("projection must be 'cylindrical' or 'spherical'")
        self.projection = projection
        self.sift = cv2.SIFT_create()
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.initialized = False
        self._calib_info = {}
        self._warp_cache = []
        self._panorama_size = None
        self._panorama_offset = (0, 0)
        self._warper = None

    def calibrate(self, left_frame, right_frame):
        """
        Computes feature matches, estimates homography, refines the camera params and
        precomputes the PyRotationWarper maps.
        """
        if left_frame is None or right_frame is None:
            raise RuntimeError("Calibration frames are missing.")
        self._warp_cache = []
        self._calib_info = {}

        proc_left = self._downscale_frame(left_frame)
        proc_right = self._downscale_frame(right_frame)

        feat_left, kps_left, des_left = self._compute_features(proc_left, 0)
        feat_right, kps_right, des_right = self._compute_features(proc_right, 1)
        if des_left is None or des_right is None or len(kps_left) < 4 or len(kps_right) < 4:
            raise RuntimeError("Could not find enough keypoints for calibration.")

        matches, inlier_mask, H = self._match_and_filter(kps_left, des_left, kps_right, des_right)
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        if H is None or inliers < 4:
            raise RuntimeError("Homography estimation failed during calibration.")

        features = [feat_left, feat_right]
        num_imgs = len(features)
        pairwise_matches = [
            self._empty_matches_info(src_idx=i, dst_idx=j) for i in range(num_imgs) for j in range(num_imgs)
        ]
        pairwise_matches[0 * num_imgs + 1] = self._build_matches_info(matches, inlier_mask, H, src_idx=0, dst_idx=1)

        if H is not None:
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                H_inv = None
            if H_inv is not None:
                reverse_matches = [cv2.DMatch(m.trainIdx, m.queryIdx, m.distance) for m in matches]
                pairwise_matches[1 * num_imgs + 0] = self._build_matches_info(
                    reverse_matches, inlier_mask, H_inv, src_idx=1, dst_idx=0
                )

        estimator = cv2.detail.HomographyBasedEstimator()
        ok, cameras = estimator.apply(features, pairwise_matches, None)
        if not ok:
            raise RuntimeError("HomographyBasedEstimator could not estimate initial cameras.")

        cameras = [self._ensure_camera_float32(cam) for cam in cameras]
        adjuster = cv2.detail.BundleAdjusterRay()
        adjuster.setConfThresh(0.0)
        ok, adjusted = adjuster.apply(features, pairwise_matches, cameras)
        if ok:
            cameras = [self._ensure_camera_float32(cam) for cam in adjusted]
            self._calib_info["bundle_adjusted"] = True
        else:
            print("[warn] BundleAdjusterRay failed; using estimator output.")
            cameras = [self._ensure_camera_float32(cam) for cam in cameras]
            self._calib_info["bundle_adjusted"] = False
        scale_up = 1.0 / self.downscale
        for cam in cameras:
            cam.focal *= scale_up
            cam.ppx *= scale_up
            cam.ppy *= scale_up
            cam.R = np.asarray(cam.R, dtype=np.float32).reshape(3, 3)

        warper_scale = max(cam.focal for cam in cameras)
        self._warper = cv2.PyRotationWarper(self.projection, float(warper_scale))

        self._prepare_warp_maps(cameras, left_frame.shape[1], left_frame.shape[0])

        self.initialized = True
        self._calib_info.update(
            {
                "matches": len(matches),
                "inliers": inliers,
                "warper_scale": warper_scale,
                "downscale": self.downscale,
                "bundle_adjusted": self._calib_info.get("bundle_adjusted", True),
                "projection": self.projection,
            }
        )
        return dict(self._calib_info)

    def stitch(self, left_frame, right_frame):
        """
        Applies the cached maps to both frames and blends them on a shared canvas.
        """
        if not self.initialized or not self._warp_cache or self._panorama_size is None:
            raise RuntimeError("Stitcher has not been calibrated yet.")

        canvas_w, canvas_h = self._panorama_size
        accum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        weights = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        for frame, cache in zip((left_frame, right_frame), self._warp_cache):
            warped = cv2.remap(
                frame,
                cache["xmap"],
                cache["ymap"],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            mask = cache["mask"]
            mask_float = mask.astype(np.float32) / 255.0
            if not np.any(mask_float):
                continue

            x0 = int(round(cache["corner"][0] + self._panorama_offset[0]))
            y0 = int(round(cache["corner"][1] + self._panorama_offset[1]))
            h, w = warped.shape[:2]
            x1 = x0 + w
            y1 = y0 + h

            accum[y0:y1, x0:x1] += warped.astype(np.float32) * mask_float[..., None]
            weights[y0:y1, x0:x1] += mask_float

        valid = weights > 1e-5
        output = np.zeros_like(accum, dtype=np.uint8)
        if np.any(valid):
            weighted = accum[valid] / weights[valid][..., None]
            output[valid] = weighted.clip(0, 255).astype(np.uint8)
        return output

    def info(self):
        return dict(self._calib_info)

    def _downscale_frame(self, frame):
        if self.downscale == 1.0:
            return frame.copy()
        new_w = max(2, int(frame.shape[1] * self.downscale))
        new_h = max(2, int(frame.shape[0] * self.downscale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _compute_features(self, frame, img_idx):
        features = cv2.detail.computeImageFeatures2(self.sift, frame)
        features.img_idx = int(img_idx)
        features.img_size = (frame.shape[1], frame.shape[0])
        keypoints = list(features.getKeypoints()) if hasattr(features, "getKeypoints") else list(features.keypoints)

        descriptors = features.descriptors
        if descriptors is not None and hasattr(descriptors, "get"):
            descriptors = descriptors.get()
        if descriptors is not None:
            descriptors = np.asarray(descriptors, dtype=np.float32, order="C")
        return features, keypoints, descriptors

    def _ensure_camera_float32(self, cam):
        cam.R = np.asarray(cam.R, dtype=np.float32).reshape(3, 3)
        cam.t = np.asarray(getattr(cam, "t", np.zeros((3, 1), dtype=np.float32)), dtype=np.float32).reshape(3, 1)
        cam.focal = float(cam.focal)
        cam.ppx = float(cam.ppx)
        cam.ppy = float(cam.ppy)
        cam.aspect = float(getattr(cam, "aspect", 1.0))
        return cam

    def _match_and_filter(self, kps_left, des_left, kps_right, des_right):
        raw_matches = self.flann.knnMatch(des_left, des_right, k=2)
        good = []
        for pair in raw_matches:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self.ratio_thresh * n.distance:
                good.append(m)
        if len(good) < 4:
            return good, None, None

        pts_left = np.float32([kps_left[m.queryIdx].pt for m in good])
        pts_right = np.float32([kps_right[m.trainIdx].pt for m in good])
        H, mask = cv2.findHomography(pts_right, pts_left, cv2.RANSAC, 4.0)
        if mask is None:
            return good, None, None
        mask = mask.reshape(-1, 1)
        if H is not None:
            H = np.asarray(H, dtype=np.float64).reshape(3, 3)
        return good, mask, H

    def _build_matches_info(self, matches, inlier_mask, H, src_idx=0, dst_idx=1):
        matches_info = cv2.detail.MatchesInfo()
        matches_info.src_img_idx = int(src_idx)
        matches_info.dst_img_idx = int(dst_idx)
        mask_list = [int(v) for v in inlier_mask.flatten().tolist()]
        matches_info.matches = matches
        matches_info.num_inliers = int(sum(mask_list))
        matches_info.inliers_mask = mask_list
        matches_info.H = np.asarray(H, dtype=np.float64).reshape(3, 3)
        conf = float(matches_info.num_inliers) / max(1, len(matches))
        matches_info.confidence = max(conf, 1e-6)
        return matches_info

    def _empty_matches_info(self, src_idx, dst_idx):
        info = cv2.detail.MatchesInfo()
        info.src_img_idx = int(src_idx)
        info.dst_img_idx = int(dst_idx)
        info.matches = []
        info.inliers_mask = []
        info.num_inliers = 0
        info.confidence = 0.0
        info.H = np.eye(3, dtype=np.float64)
        return info

    def _prepare_warp_maps(self, cameras, width, height):
        image_size = (width, height)
        all_corners = []
        max_x = -np.inf
        max_y = -np.inf
        min_x = np.inf
        min_y = np.inf
        self._warp_cache = []

        full_mask = np.ones((height, width), dtype=np.uint8) * 255
        for cam in cameras:
            K = np.asarray(cam.K(), dtype=np.float32)
            R = np.asarray(cam.R, dtype=np.float32).reshape(3, 3)
            corner, xmap, ymap = self._warper.buildMaps(image_size, K, R)
            warped_mask = cv2.remap(
                full_mask, xmap, ymap, interpolation=cv2.INTER_NEAREST, borderValue=0
            )
            h, w = warped_mask.shape[:2]
            cx, cy = corner[0], corner[1]

            min_x = min(min_x, cx)
            min_y = min(min_y, cy)
            max_x = max(max_x, cx + w)
            max_y = max(max_y, cy + h)
            all_corners.append((cx, cy, w, h))

            self._warp_cache.append(
                {
                    "xmap": xmap,
                    "ymap": ymap,
                    "corner": (cx, cy),
                    "size": (w, h),
                    "mask": warped_mask,
                }
            )

        min_x = int(np.floor(min_x))
        min_y = int(np.floor(min_y))
        max_x = int(np.ceil(max_x))
        max_y = int(np.ceil(max_y))
        self._panorama_offset = (-min_x, -min_y)
        self._panorama_size = (max_x - min_x, max_y - min_y)
        self._calib_info["corners"] = all_corners
        self._calib_info["panorama_size"] = self._panorama_size
