"""Capture and label real images of your electronic dartboard's LED display.

For each dart thrown, press SPACE to grab the current (rectified) display
crop, then type in the terminal what the display actually shows. Builds a
labeled dataset from your own camera/lighting/board -- this matters far
more than synthetic data once you have a few hundred real examples. Upload
the resulting images/ to Roboflow to draw per-character boxes and merge
with data/synthetic_display/ for training.

Requires display calibration first:  python scripts/calibrate_display.py

Usage: python scripts/capture_display_labels.py
SPACE = capture + label the current frame | Q = quit
"""
import sys
import csv
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.cv.display_calibrator import load_display_calibration, rectify

CAM_INDEX = 0  # edit if your webcam isn't device 0

OUT_DIR    = Path(__file__).parent.parent / "data" / "real_display_captures"
IMAGES_DIR = OUT_DIR / "images"
LABELS_CSV = OUT_DIR / "labels.csv"


def _next_index() -> int:
    existing = list(IMAGES_DIR.glob("capture_*.jpg"))
    if not existing:
        return 0
    nums = [int(p.stem.split("_")[1]) for p in existing]
    return max(nums) + 1


def main():
    cal = load_display_calibration()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not LABELS_CSV.exists()
    csv_file = open(LABELS_CSV, "a", newline="")
    writer = csv.writer(csv_file)
    if new_file:
        writer.writerow(["filename", "label", "timestamp"])

    idx = _next_index()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera (device {CAM_INDEX}).")
        sys.exit(1)

    print(f"Camera opened. {idx} existing captures. SPACE = capture, Q = quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed -- is the camera in use by another app?")
                break

            crop = rectify(frame, cal)
            cv2.imshow("Display capture", crop)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
            if key == ord(" "):
                filename = f"capture_{idx:05d}.jpg"
                cv2.imwrite(str(IMAGES_DIR / filename), crop)
                label = input(
                    f"  [{filename}] What does the display show? (blank to discard): "
                ).strip()
                if label:
                    writer.writerow([filename, label, time.time()])
                    csv_file.flush()
                    print(f"  Saved: {filename} -> {label!r}  ({idx + 1} total)")
                    idx += 1
                else:
                    (IMAGES_DIR / filename).unlink(missing_ok=True)
                    print("  Discarded (empty label).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        csv_file.close()
        print(f"Done. Labels saved to {LABELS_CSV}")


if __name__ == "__main__":
    main()
