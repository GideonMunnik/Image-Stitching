import argparse
import cv2
import numpy as np

from stitch_advanced import AdvancedStitcher
from stitch_basic import stitch_basic
from video_source import VideoStreamPair


def parse_args():
    parser = argparse.ArgumentParser(description="Image stitching demo (basic vs advanced).")
    parser.add_argument("left_video", help="Path to the left video file")
    parser.add_argument("right_video", help="Path to the right video file")
    parser.add_argument("--width", type=int, help="Resize width for both videos")
    parser.add_argument("--height", type=int, help="Resize height for both videos")
    parser.add_argument("--overlap", type=float, default=0.2, help="Initial overlap fraction for basic mode (0-0.9)")
    parser.add_argument("--advanced", action="store_true", help="Start with homography stitching enabled")
    parser.add_argument(
        "--feature-interval", type=int, default=30, help="Frames between feature recalculation in advanced mode"
    )
    parser.add_argument("--max-features", type=int, default=1200, help="Feature cap (lower to save CPU)")
    parser.add_argument(
        "--roi-fraction",
        type=float,
        default=0.2,
        help="Seam-band width fraction for feature detection (e.g. 0.2 = left/right 20% strips)",
    )
    parser.add_argument(
        "--detect-downscale",
        type=float,
        default=0.5,
        help="Downscale factor for feature detection (1.0 keeps full res, 0.5 halves it)",
    )
    parser.add_argument(
        "--good-match-percent",
        type=float,
        default=0.15,
        help="Top match fraction to keep when estimating homography (0.05-0.5)",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="ORB",
        choices=["ORB", "AKAZE", "SIFT"],
        help="Feature detector to use",
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default="BF",
        choices=["BF", "FLANN"],
        help="Descriptor matcher to use",
    )
    parser.add_argument(
        "--optical-flow",
        action="store_true",
        help="Use optical flow between feature refreshes to update homography",
    )
    parser.add_argument(
        "--stability",
        type=float,
        default=0.0,
        help="Homography smoothing strength (0.0 = no smoothing, 1.0 = very stable)",
    )
    return parser.parse_args()


def overlay_text(frame, lines):
    y = 28
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 20), 2, cv2.LINE_AA)
        y += 26
    return frame


def main():
    args = parse_args()
    target_size = None
    if args.width and args.height:
        target_size = (args.width, args.height)

    stream = VideoStreamPair(args.left_video, args.right_video, target_size=target_size)
    stitcher = AdvancedStitcher(
        feature_interval=args.feature_interval,
        max_features=args.max_features,
        roi_fraction=args.roi_fraction,
        detect_downscale=args.detect_downscale,
        good_match_percent=args.good_match_percent,
        detector_type=args.detector,
        matcher_type=args.matcher,
        use_optical_flow=args.optical_flow,
        stability=args.stability,
    )

    window = "Stitching Demo"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    detector_choices = ["ORB", "AKAZE", "SIFT"]
    matcher_choices = ["BF", "FLANN"]
    detector_index = detector_choices.index(args.detector)
    matcher_index = matcher_choices.index(args.matcher)
    cv2.createTrackbar("Overlap %", window, int(np.clip(args.overlap, 0, 0.9) * 100), 90, lambda x: None)
    cv2.createTrackbar("Homography", window, 1 if args.advanced else 0, 1, lambda x: None)
    cv2.createTrackbar("Optical Flow", window, 1 if args.optical_flow else 0, 1, lambda x: None)
    cv2.createTrackbar("Stability %", window, int(np.clip(args.stability, 0.0, 1.0) * 100), 100, lambda x: None)
    interval_max = max(120, int(args.feature_interval))
    cv2.createTrackbar("Recalc N frames", window, max(1, args.feature_interval), interval_max, lambda x: None)
    cv2.createTrackbar("Max features", window, max(200, int(args.max_features)), 4000, lambda x: None)
    cv2.createTrackbar("ROI %", window, int(np.clip(args.roi_fraction, 0.05, 1.0) * 100), 100, lambda x: None)
    cv2.createTrackbar(
        "Detect %", window, int(np.clip(args.detect_downscale, 0.1, 1.0) * 100), 100, lambda x: None
    )
    cv2.createTrackbar(
        "Good match %", window, int(np.clip(args.good_match_percent, 0.05, 0.5) * 100), 50, lambda x: None
    )
    cv2.createTrackbar("Detector", window, detector_index, len(detector_choices) - 1, lambda x: None)
    cv2.createTrackbar("Matcher", window, matcher_index, len(matcher_choices) - 1, lambda x: None)
    print("Controls: q/ESC to quit, space to toggle homography, r to reset homography.")

    last_detector = stitcher.detector_type
    last_matcher = stitcher.matcher_type

    while True:
        try:
            left, right = stream.read()
        except RuntimeError as exc:
            print(f"[error] {exc}")
            break

        homography_on = cv2.getTrackbarPos("Homography", window) == 1
        overlap = np.clip(cv2.getTrackbarPos("Overlap %", window) / 100.0, 0.0, 0.9)
        stitcher.feature_interval = max(1, cv2.getTrackbarPos("Recalc N frames", window))
        stitcher.max_features = max(200, cv2.getTrackbarPos("Max features", window))
        stitcher.roi_fraction = np.clip(cv2.getTrackbarPos("ROI %", window) / 100.0, 0.05, 1.0)
        stitcher.detect_downscale = np.clip(cv2.getTrackbarPos("Detect %", window) / 100.0, 0.1, 1.0)
        stitcher.good_match_percent = np.clip(cv2.getTrackbarPos("Good match %", window) / 100.0, 0.05, 0.5)
        stitcher.stability = np.clip(cv2.getTrackbarPos("Stability %", window) / 100.0, 0.0, 1.0)
        stitcher.use_optical_flow = cv2.getTrackbarPos("Optical Flow", window) == 1

        detector_choice = detector_choices[cv2.getTrackbarPos("Detector", window)]
        matcher_choice = matcher_choices[cv2.getTrackbarPos("Matcher", window)]
        if detector_choice != last_detector or matcher_choice != last_matcher:
            stitcher.detector_type = detector_choice
            stitcher.matcher_type = matcher_choice
            stitcher.reset_state()
            last_detector = detector_choice
            last_matcher = matcher_choice

        stitched = None
        debug = {}
        if homography_on:
            stitched, debug = stitcher.stitch(left, right)
            if stitched is None:
                stitched = stitch_basic(left, right, overlap)
                debug = {"homography_found": False}
        else:
            stitched = stitch_basic(left, right, overlap)

        lines = [
            f"Mode: {'HOMOGRAPHY' if homography_on else 'BASIC'}",
            f"Overlap: {int(overlap * 100)}%",
            f"Feature interval: {stitcher.feature_interval} frames",
        ]
        if homography_on and debug:
            if "matches" in debug:
                lines.append(f"Matches: {debug.get('matches', 0)} | Inliers: {debug.get('inliers', 0)}")
            if debug.get("flow_used") or debug.get("flow_failed"):
                lines.append(
                    f"Flow tracked: {debug.get('flow_tracked', 0)} | Inliers: {debug.get('flow_inliers', 0)}"
                )
            if not debug.get("homography_found", True):
                lines.append("Homography not found - falling back")
            elif debug.get("recomputed"):
                lines.append("Homography refreshed")
            if debug.get("recompute_failed_kept_previous"):
                lines.append("Recalc failed - using previous homography")
            if debug.get("flow_failed"):
                lines.append("Flow failed - using previous homography")
            lines.append(
                "ROI "
                f"{int(stitcher.roi_fraction*100)}% band | "
                f"detect scale {stitcher.detect_downscale:.2f} | "
                f"cap {stitcher.max_features} | "
                f"good match {int(stitcher.good_match_percent*100)}% | "
                f"stability {int(stitcher.stability*100)}%"
            )
            detector_used = debug.get("detector", stitcher.detector_type)
            matcher_used = debug.get("matcher", stitcher.matcher_type)
            if detector_used != stitcher.detector_type:
                detector_label = f"{stitcher.detector_type} -> {detector_used}"
            else:
                detector_label = stitcher.detector_type
            if matcher_used != stitcher.matcher_type:
                matcher_label = f"{stitcher.matcher_type} -> {matcher_used}"
            else:
                matcher_label = stitcher.matcher_type
            lines.append(
                f"Detector: {detector_label} | Matcher: {matcher_label} | Update: {debug.get('update_source', 'reuse')}"
            )

        display = overlay_text(stitched.copy(), lines)
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            new_val = 0 if homography_on else 1
            cv2.setTrackbarPos("Homography", window, new_val)
        if key == ord("r"):
            stitcher.reset_state(reset_frame_count=True)
            print("Homography reset.")

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
