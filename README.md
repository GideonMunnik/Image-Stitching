# Image Stitching Demo

Python demo for stitching two looping videos in basic (fixed overlap) or homography (feature-based) mode. The homography mode refreshes the transform every _N_ frames and falls back to basic stitching if matching fails.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run
```bash
python main.py path/to/left.mp4 path/to/right.mp4 --width 1280 --height 720 --overlap 0.2 --feature-interval 30 --advanced
```
Options:
- `--width/--height` resize both streams (recommended when sources differ).
- `--overlap` starting overlap fraction for basic mode (0-0.9).
- `--advanced` start with homography mode enabled.
- `--feature-interval` frames between feature recomputation in homography mode.
- `--max-features` cap keypoints (lower for less CPU).
- `--roi-fraction` width fraction for the seam bands used in feature detection (e.g. 0.2 = left/right 20% strips).
- `--detect-downscale` downscale factor for feature detection (1.0 keeps full res, 0.5 halves it).
- `--good-match-percent` top match fraction to keep for homography (0.05-0.5).
- `--detector` feature detector (`ORB`, `AKAZE`, `SIFT`).
- `--matcher` descriptor matcher (`BF`, `FLANN`).
- `--optical-flow` update homography between feature refreshes using optical flow.
- `--stability` homography smoothing strength (0.0 = none, 1.0 = very stable).

Note: If `SIFT` is unavailable in your OpenCV build, it will fall back to `ORB`. Install `opencv-contrib-python` if you need guaranteed SIFT support.

## Controls
- `q` or `ESC`: quit.
- `space`: toggle homography mode via the trackbar.
- `r`: reset the advanced stitcher (drops cached homography/flow state).
- `w`: capture the first frame of each video, compute a calibration homography, save `homography_sample.png`, and enter **Calibrated** mode.
- `a`: switch to **Calibrated** mode (if a calibration exists).
- `s`: switch to **Basic** mode (homography off, calibration off).
- `d`: switch to **Homography** mode (live feature tracking).
- Trackbars inside the OpenCV window let you adjust overlap, toggle homography/optical flow, tune feature settings, and set the `Balance` slider (0 = warp left, 1 = warp both, 2 = warp right). In Calibrated mode the current balance is respected for the static warp.

## Extra performance tips
- Keep input resolution modest (`--width/--height`) to reduce the warp cost.
- Lower `--max-features`, increase `--feature-interval`, or reduce `--roi-fraction` (seam band width) to cut feature-matching work.
- Leave `--detect-downscale` at 0.5 or smaller for faster feature detection while still warping the full-res frames.
