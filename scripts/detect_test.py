"""
Standalone dart detection sanity check.

Tests ONLY whether the YOLO model puts boxes on dart tips in a webcam frame.
No calibration, no scoring, no coordinate mapping -- those require the mounted,
calibrated camera. This deliberately bypasses DartDetector (which needs
data/calibration.json) so it can run before any mounting/calibration exists.

Run on the Windows dev laptop with a USB webcam.
    python scripts/detect_test.py
Edit MODEL_PATH and CAM_INDEX below if needed.
"""
import time
import cv2
from ultralytics import YOLO

# --- edit these two lines ---------------------------------------------------
MODEL_PATH = "data/dart_model.pt"  # point at your downloaded Roboflow weights
CAM_INDEX = 0                      # 0 = default USB webcam; try 1 or 2 if wrong cam opens
# ---------------------------------------------------------------------------

CONF = 0.25  # intentionally low -- we want to see marginal detections, not hide them

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError(
        f"Camera index {CAM_INDEX} did not open. "
        f"Edit CAM_INDEX (try 1 or 2) and re-run."
    )

print("Running. SPACE = print detections to terminal | Q = quit")
prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed -- is the camera in use by another app?")
        break

    results = model(frame, conf=CONF, verbose=False)[0]

    overlay = frame.copy()
    n = 0
    for box in results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # estimated tip = bottom-center of the box (matches detector.py convention)
        cv2.circle(overlay, ((x1 + x2) // 2, y2), 5, (0, 0, 255), -1)
        cv2.putText(overlay, f"{conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        n += 1

    now = time.time()
    fps = 1.0 / (now - prev) if now != prev else 0.0
    prev = now
    cv2.putText(overlay, f"darts: {n}  fps: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Dart detection test", overlay)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        print(f"\n--- {n} detection(s) ---")
        for box in results.boxes:
            b = [round(v, 1) for v in box.xyxy[0].tolist()]
            print(f"  conf={float(box.conf[0]):.3f}  box(xyxy)={b}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
