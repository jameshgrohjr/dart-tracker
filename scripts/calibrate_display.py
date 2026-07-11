"""Interactive calibration for an electronic dartboard's LED score display.

The display is usually a small fraction of the camera frame, which makes
clicking its 4 corners precisely (at native resolution) error-prone -- a
click that's off by a few pixels can crop the wrong region entirely, or
warp a near-point-sized sliver up to fill the output size (producing an
unusably blurry rectified crop). To fix that, this is a two-stage flow:

1. On the frozen frame, click ONCE roughly where the display is.
2. A zoomed-in window opens (a magnified crop centered on that click).
   Click the 4 corners of the display -- in order: top-left, top-right,
   bottom-right, bottom-left -- on THIS zoomed view instead, for far more
   precision. Press R at any point in this stage to re-center if the
   display doesn't fully fit in the zoomed crop.

Saves a homography that rectifies the clicked region into a straightened
crop of fixed size, so every later capture / model inference sees a
consistent frontal view regardless of the camera's mounting angle.

Usage: python scripts/calibrate_display.py
Press SPACE to capture a frame, click roughly on the display, then click
its 4 corners in the zoomed view. Press Q at any time to quit.
"""
import sys
import json
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"
CAM_INDEX = 1             # edit if your webcam isn't device 1
OUTPUT_SIZE = (640, 160)  # rectified crop size (w, h) fed to the display model later
ZOOM_FACTOR = 5           # magnification applied to the region around your rough click
ZOOM_CROP_RADIUS = 130    # pixels around the rough click (in the original frame) to zoom into

FULL_WIN = "Display calibration"
ZOOM_WIN = "Zoomed - click 4 corners"

CORNER_LABELS = [
    "top-left corner of the display",
    "top-right corner of the display",
    "bottom-right corner of the display",
    "bottom-left corner of the display",
]


def zoom_crop_bounds(cx: int, cy: int, frame_w: int, frame_h: int, radius: int = ZOOM_CROP_RADIUS) -> tuple:
    """Top-left corner (x0, y0) and side length of a square crop of the
    frame, centered on (cx, cy) as closely as the frame edges allow."""
    size = 2 * radius
    x0 = int(max(0, min(cx - radius, frame_w - size)))
    y0 = int(max(0, min(cy - radius, frame_h - size)))
    return x0, y0, size


def zoom_to_original(zx: float, zy: float, x0: int, y0: int, zoom_factor: int = ZOOM_FACTOR) -> tuple:
    """Map a click on the zoomed/magnified view back to original-frame pixel coords."""
    return x0 + zx / zoom_factor, y0 + zy / zoom_factor


def main():
    state = {"frame": None, "stage": "live", "zoom_origin": None, "zoom_clean": None, "points": []}

    def redraw_zoom():
        disp = state["zoom_clean"].copy()
        x0, y0 = state["zoom_origin"]
        for i, (px, py) in enumerate(state["points"]):
            zx, zy = int((px - x0) * ZOOM_FACTOR), int((py - y0) * ZOOM_FACTOR)
            cv2.circle(disp, (zx, zy), 6, (0, 255, 0), 2)
            cv2.putText(disp, str(i + 1), (zx + 10, zy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(ZOOM_WIN, disp)

    def start_zoom(cx: int, cy: int):
        frame = state["frame"]
        h, w = frame.shape[:2]
        x0, y0, size = zoom_crop_bounds(cx, cy, w, h)
        crop = frame[y0:y0 + size, x0:x0 + size]
        state["zoom_clean"] = cv2.resize(crop, None, fx=ZOOM_FACTOR, fy=ZOOM_FACTOR, interpolation=cv2.INTER_LINEAR)
        state["zoom_origin"] = (x0, y0)
        state["points"] = []
        state["stage"] = "corners"
        cv2.namedWindow(ZOOM_WIN)
        cv2.setMouseCallback(ZOOM_WIN, on_zoom_click)
        redraw_zoom()
        print(f"Zoomed in. Click point 1: {CORNER_LABELS[0]}  (R = re-center, Q = quit)")

    def on_full_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or state["stage"] != "rough":
            return
        start_zoom(x, y)

    def on_zoom_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or state["stage"] != "corners":
            return
        x0, y0 = state["zoom_origin"]
        state["points"].append(zoom_to_original(x, y, x0, y0))
        redraw_zoom()
        n = len(state["points"])
        if n < 4:
            print(f"Point {n} captured. Click point {n + 1}: {CORNER_LABELS[n]}")
        else:
            print("All 4 corners captured. Computing homography...")
            compute_and_save()

    def compute_and_save():
        w, h = OUTPUT_SIZE
        src = np.array(state["points"], dtype=np.float32)
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        img_h, img_w = state["frame"].shape[:2]
        cal = {
            "homography": H.tolist(),
            "output_size": [w, h],
            "image_size": [img_w, img_h],
            "pixel_points": [list(p) for p in state["points"]],
        }
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out = DATA_DIR / "display_calibration.json"
        with open(out, "w") as f:
            json.dump(cal, f, indent=2)
        print(f"Display calibration saved to {out}")

        preview = cv2.warpPerspective(state["frame"], H, (w, h))
        cv2.imshow("Rectified preview (press any key to close)", preview)
        state["stage"] = "done"

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera (device {CAM_INDEX}).")
        sys.exit(1)

    cv2.namedWindow(FULL_WIN)
    cv2.setMouseCallback(FULL_WIN, on_full_click)

    print("Camera opened. Press SPACE to capture a frame, Q to quit.")

    while True:
        if state["stage"] == "live":
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame.")
                break
            cv2.imshow(FULL_WIN, frame)
        elif state["frame"] is not None:
            cv2.imshow(FULL_WIN, state["frame"])

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key == ord(" ") and state["stage"] == "live":
            ret, frame = cap.read()
            if ret:
                state["frame"] = frame.copy()
                state["stage"] = "rough"
                print("Frame captured. Click roughly on the display to zoom in.")
        if key in (ord("r"), ord("R")) and state["stage"] == "corners":
            cv2.destroyWindow(ZOOM_WIN)
            state["stage"] = "rough"
            state["points"] = []
            print("Re-centering. Click roughly on the display to zoom in again.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
