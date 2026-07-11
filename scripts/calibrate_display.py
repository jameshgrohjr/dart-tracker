"""Interactive calibration for an electronic dartboard's LED score display.

Click the 4 corners of the display region (in order: top-left, top-right,
bottom-right, bottom-left) on a captured frame. Saves a homography that
rectifies that region into a straightened crop of fixed size, so every
later capture / model inference sees a consistent frontal view regardless
of the camera's mounting angle.

Usage: python scripts/calibrate_display.py
Press SPACE to capture a frame, then click the 4 corners in order.
Press Q at any time to quit.
"""
import sys
import json
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"
CAM_INDEX = 0             # edit if your webcam isn't device 0
OUTPUT_SIZE = (640, 160)  # rectified crop size (w, h) fed to the display model later

CORNER_LABELS = [
    "top-left corner of the display",
    "top-right corner of the display",
    "bottom-right corner of the display",
    "bottom-left corner of the display",
]


def main():
    clicked_points = []
    captured_frame = [None]

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if captured_frame[0] is None:
            return
        n = len(clicked_points)
        if n >= 4:
            return
        clicked_points.append([x, y])
        cv2.circle(captured_frame[0], (x, y), 6, (0, 255, 0), 2)
        cv2.putText(captured_frame[0], str(n + 1), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Display calibration", captured_frame[0])
        if n + 1 < 4:
            print(f"Point {n + 1} captured. Click point {n + 2}: {CORNER_LABELS[n + 1]}")
        else:
            print("All 4 corners captured. Computing homography...")
            compute_and_save(captured_frame[0].shape)

    def compute_and_save(shape):
        w, h = OUTPUT_SIZE
        src = np.array(clicked_points, dtype=np.float32)
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        img_h, img_w = shape[:2]
        cal = {
            "homography": H.tolist(),
            "output_size": [w, h],
            "image_size": [img_w, img_h],
            "pixel_points": clicked_points,
        }
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out = DATA_DIR / "display_calibration.json"
        with open(out, "w") as f:
            json.dump(cal, f, indent=2)
        print(f"Display calibration saved to {out}")

        preview = cv2.warpPerspective(captured_frame[0], H, (w, h))
        cv2.imshow("Rectified preview (press any key to close)", preview)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera (device {CAM_INDEX}).")
        sys.exit(1)

    cv2.namedWindow("Display calibration")
    cv2.setMouseCallback("Display calibration", on_mouse)

    print("Camera opened. Press SPACE to capture a frame, Q to quit.")
    print(f"After capture, click point 1: {CORNER_LABELS[0]}")

    while True:
        if captured_frame[0] is None:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame.")
                break
            cv2.imshow("Display calibration", frame)
        else:
            cv2.imshow("Display calibration", captured_frame[0])

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        if key == ord(" ") and captured_frame[0] is None:
            ret, frame = cap.read()
            if ret:
                captured_frame[0] = frame.copy()
                print(f"Frame captured. Click point 1: {CORNER_LABELS[0]}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
