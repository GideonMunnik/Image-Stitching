import argparse
import cv2

from stitch_cylindrical import CylindricalStitcher
from video_source import VideoStreamPair


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
    y = 28
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 240, 30), 2, cv2.LINE_AA)
        y += 28
    return frame


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
    calib_info = stitcher.calibrate(current_left, current_right)
    canvas_w, canvas_h = calib_info.get("panorama_size", (0, 0))
    if not calib_info.get("bundle_adjusted", True):
        print("[warn] Bundle adjustment failed; using estimator output.")
    print(
        "Calibration: matches {matches} | inliers {inliers} | warper scale {scale:.2f} | canvas {cw}x{ch}".format(
            matches=calib_info.get("matches", 0),
            inliers=calib_info.get("inliers", 0),
            scale=calib_info.get("warper_scale", 0.0),
            cw=canvas_w,
            ch=canvas_h,
        )
    )

    window = "Cylindrical Stitching"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("Controls: q/ESC to quit.")

    while True:
        try:
            stitched = stitcher.stitch(current_left, current_right)
        except RuntimeError as exc:
            print(f"[stitch] {exc}")
            break

        lines = [
            f"{args.projection.capitalize()} stitch",
            f"Matches: {calib_info.get('matches', 0)} | Inliers: {calib_info.get('inliers', 0)}",
            f"Warper scale: {calib_info.get('warper_scale', 0.0):.1f} | Downscale: {calib_info.get('downscale', 0.0):.2f}",
            f"Canvas: {canvas_w} x {canvas_h}",
            "Press q or ESC to exit",
        ]
        display = overlay_text(stitched.copy(), lines)
        cv2.imshow(window, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

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
