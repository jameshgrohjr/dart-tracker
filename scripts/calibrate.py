"""Interactive dartboard camera calibration tool.

Usage: python scripts/calibrate.py
Press SPACE to capture a frame, then click 5 reference points in order.
Press Q at any time to quit.
"""
import sys
import json
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"

# Board coordinates (mm) for the 5 reference points
BOARD_POINTS = [
    [0.0,    0.0],   # 1: Bull centre
    [0.0,  170.0],   # 2: Top of double ring (12 o'clock)
    [0.0, -170.0],   # 3: Bottom of double ring (6 o'clock)
    [170.0,  0.0],   # 4: Right of double ring (3 o'clock)
    [-170.0, 0.0],   # 5: Left of double ring (9 o'clock)
]

POINT_LABELS = [
    "Bull centre (0, 0)",
    "Top double ring (12 o'clock)",
    "Bottom double ring (6 o'clock)",
    "Right double ring (3 o'clock)",
    "Left double ring (9 o'clock)",
]


def pixel_to_board(px: float, py: float, H: np.ndarray) -> tuple:
    """Convert pixel coordinates to board mm coordinates via homography."""
    pt = np.array([[[px, py]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    x_mm, y_mm = result[0][0]
    return float(x_mm), float(y_mm)


def main():
    clicked_points = []
    captured_frame = [None]

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if captured_frame[0] is None:
            return
        n = len(clicked_points)
        if n >= 5:
            return
        clicked_points.append([x, y])
        cv2.circle(captured_frame[0], (x, y), 8, (0, 255, 0), 2)
        cv2.putText(captured_frame[0], str(n + 1), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Calibration", captured_frame[0])
        if n + 1 < 5:
            print(f"Point {n + 1} captured. Click point {n + 2}: {POINT_LABELS[n + 1]}")
        else:
            print("All 5 points captured. Computing homography...")
            compute_and_save(captured_frame[0].shape)

    def compute_and_save(shape):
        src = np.array(clicked_points, dtype=np.float32)
        dst = np.array(BOARD_POINTS, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        h, w = shape[:2]
        cal = {
            "homography": H.tolist(),
            "image_size": [w, h],
            "pixel_points": clicked_points,
            "board_points": BOARD_POINTS,
        }
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        out = DATA_DIR / "calibration.json"
        with open(out, "w") as f:
            json.dump(cal, f, indent=2)
        print(f"Calibration saved to {out}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera (device 0).")
        sys.exit(1)

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_mouse)

    print("Camera opened. Press SPACE to capture a frame, Q to quit.")
    print(f"After capture, click point 1: {POINT_LABELS[0]}")

    while True:
        if captured_frame[0] is None:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame.")
                break
            cv2.imshow("Calibration", frame)
        else:
            cv2.imshow("Calibration", captured_frame[0])

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        if key == ord(" ") and captured_frame[0] is None:
            ret, frame = cap.read()
            if ret:
                captured_frame[0] = frame.copy()
                print(f"Frame captured. Click point 1: {POINT_LABELS[0]}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
