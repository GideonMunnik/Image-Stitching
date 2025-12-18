import argparse
import cv2

from stitch_cylindrical import CylindricalStitcher
from video_source import VideoStreamPair

DETECTOR_CHOICES = ["ORB", "AKAZE", "SIFT"]
MATCHER_CHOICES = ["BF", "FLANN"]


def parse_args():
    parser = argparse.ArgumentParser(description="Projection stitching demo.")
    parser.add_argument(
        "projection",
        choices=["cylindrical", "spherical"],
        help="Projection type to use for warping",
    )
    parser.add_argument("left_video", help="Path to the left video file")
    parser.add_argument("right_video", help="Path to the right video file")
    parser.add_argument("--width", type=int, help="Resize width for both videos")
    parser.add_argument("--height", type=int, help="Resize height for both videos")
    parser.add_argument(
        "--downscale",
        type=float,
        default=0.5,
        help="Downscale factor for calibration frames (0.1-1.0, default 0.5)",
    )
    return parser.parse_args()


def overlay_text(frame, lines):
    y = 18
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 240, 30), 1, cv2.LINE_AA)
        y += 16
    return frame


def matrix_to_lines(label, mat):
    segments = []
    for idx in range(min(3, mat.shape[0])):
        row = mat[idx]
        segments.append(" ".join(f"{val:.1f}" for val in row))
    return [f"{label}: [{' | '.join(segments)}]"]


def perform_calibration(stitcher, left_frame, right_frame):
    info = stitcher.calibrate(left_frame, right_frame)
    canvas_w, canvas_h = info.get("panorama_size", (0, 0))
    if not info.get("bundle_adjusted", True):
        print("[warn] Bundle adjustment failed; using estimator output.")
    print(
        "Calibration: matches {matches} | inliers {inliers} | warper scale {scale:.2f} | canvas {cw}x{ch}".format(
            matches=info.get("matches", 0),
            inliers=info.get("inliers", 0),
            scale=info.get("warper_scale", 0.0),
            cw=canvas_w,
            ch=canvas_h,
        )
    )
    return info, canvas_w, canvas_h


def create_trackbars(window, stitcher):
    cv2.createTrackbar("Max features", window, 0, 4000, lambda x: None)
    cv2.createTrackbar("ROI %", window, 0, 100, lambda x: None)
    cv2.createTrackbar("Detect %", window, 0, 100, lambda x: None)
    cv2.createTrackbar("Good match %", window, 0, 50, lambda x: None)
    cv2.createTrackbar("Detector", window, 0, len(DETECTOR_CHOICES) - 1, lambda x: None)
    cv2.createTrackbar("Matcher", window, 0, len(MATCHER_CHOICES) - 1, lambda x: None)

    cv2.setTrackbarPos("Max features", window, 4000)
    cv2.setTrackbarPos("ROI %", window, 100)
    cv2.setTrackbarPos("Detect %", window, 100)
    cv2.setTrackbarPos("Good match %", window, 50)
    cv2.setTrackbarPos("Detector", window, len(DETECTOR_CHOICES) - 1)
    cv2.setTrackbarPos("Matcher", window, len(MATCHER_CHOICES) - 1)


def apply_trackbar_settings(window, stitcher):
    stitcher.max_features = max(200, cv2.getTrackbarPos("Max features", window))
    roi_val = cv2.getTrackbarPos("ROI %", window)
    stitcher.roi_fraction = min(1.0, max(0.05, roi_val / 100.0))
    detect_val = max(10, cv2.getTrackbarPos("Detect %", window))
    stitcher.detect_downscale = min(1.0, max(0.1, detect_val / 100.0))
    good_val = max(5, cv2.getTrackbarPos("Good match %", window))
    stitcher.good_match_percent = min(0.5, max(0.05, good_val / 100.0))
    det_idx = min(len(DETECTOR_CHOICES) - 1, max(0, cv2.getTrackbarPos("Detector", window)))
    mat_idx = min(len(MATCHER_CHOICES) - 1, max(0, cv2.getTrackbarPos("Matcher", window)))
    stitcher.detector_type = DETECTOR_CHOICES[det_idx]
    stitcher.matcher_type = MATCHER_CHOICES[mat_idx]


def main():
    args = parse_args()
    target_size = None
    if args.width and args.height:
        target_size = (args.width, args.height)

    stream = VideoStreamPair(args.left_video, args.right_video, target_size=target_size)
    try:
        current_left, current_right = stream.read()
    except RuntimeError as exc:
        stream.release()
        raise RuntimeError(f"Could not read initial frames: {exc}") from exc

    stitcher = CylindricalStitcher(downscale=args.downscale, projection=args.projection)

    window = "Cylindrical Stitching"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    create_trackbars(window, stitcher)
    apply_trackbar_settings(window, stitcher)
    calib_info, canvas_w, canvas_h = perform_calibration(stitcher, current_left, current_right)
    print("Controls: q/ESC to quit, r to recalibrate.")

    while True:
        try:
            stitched = stitcher.stitch(current_left, current_right)
        except RuntimeError as exc:
            print(f"[stitch] {exc}")
            break

        ratio = stitcher.compute_inlier_ratio(current_left, current_right)

        lines = [
            "Press q or ESC to exit",
            f"{args.projection.capitalize()} stitch",
            f"Matches: {calib_info.get('matches', 0)} | Inliers: {calib_info.get('inliers', 0)}",
            f"Warper scale: {calib_info.get('warper_scale', 0.0):.1f} | Downscale: {calib_info.get('downscale', 0.0):.2f}",
            f"Canvas: {canvas_w} x {canvas_h}",
            f"Detector: {stitcher.detector_type} | Matcher: {stitcher.matcher_type}",
            (
                f"Inlier Ratio: {ratio:.1f}%"
                + (" (press 'r' to recalibrate)" if ratio is not None and ratio < 70.0 else "")
                if ratio is not None
                else "Inlier Ratio: n/a"
            ),
        ]
        for idx, cam in enumerate(stitcher.get_camera_info()):
            lines.append(f"Cam{idx} focal: {cam['focal']:.1f}")
            lines.extend(matrix_to_lines(f"K{idx}", cam["K"]))
            lines.extend(matrix_to_lines(f"R{idx}", cam["R"]))
        display = overlay_text(stitched.copy(), lines)
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            try:
                apply_trackbar_settings(window, stitcher)
                calib_info, canvas_w, canvas_h = perform_calibration(stitcher, current_left, current_right)
            except RuntimeError as exc:
                print(f"[recalc] {exc}")

        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

        try:
            current_left, current_right = stream.read()
        except RuntimeError as exc:
            print(f"[stream] {exc}")
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
