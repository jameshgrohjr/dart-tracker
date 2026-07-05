"""Dart detector — YOLOv8 inference + calibrated homography mapping."""
import json
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

from app.cv.calibrator import coords_to_segment_ring

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_MODEL_PATH = DATA_DIR / "dart_model.pt"
DEFAULT_CALIBRATION_PATH = DATA_DIR / "calibration.json"


class DartDetector:
    def __init__(
        self,
        model_path: str | Path | None = None,
        calibration_path: str | Path | None = None,
        conf_threshold: float = 0.4,
    ):
        model_path = Path(model_path or DEFAULT_MODEL_PATH)
        cal_path = Path(calibration_path or DEFAULT_CALIBRATION_PATH)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run  python scripts/download_model.py  to fetch the dart detection model."
            )

        self.model = YOLO(str(model_path))
        self.conf_threshold = conf_threshold
        self.homography: np.ndarray | None = None

        if cal_path.exists():
            with open(cal_path) as f:
                cal = json.load(f)
            self.homography = np.array(cal["homography"], dtype=np.float64)

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run YOLO on one frame; return a list of dart-hit dicts.

        Each dict contains:
            x_px, y_px   — tip position in pixel space
            conf          — detection confidence
            x_mm, y_mm   — board position in mm (None if uncalibrated)
            segment, ring, score  — dartboard result (None if uncalibrated)
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Centre-x, bottom-y of bounding box is the dart tip
            x_px = (x1 + x2) / 2.0
            y_px = y2

            hit: dict = {
                "x_px": x_px,
                "y_px": y_px,
                "conf": float(box.conf[0]),
                "x_mm": None,
                "y_mm": None,
                "segment": None,
                "ring": None,
                "score": None,
            }

            if self.homography is not None:
                pt = np.array([[[x_px, y_px]]], dtype=np.float32)
                board_pt = cv2.perspectiveTransform(pt, self.homography)[0][0]
                x_mm, y_mm = float(board_pt[0]), float(board_pt[1])
                segment, ring, score = coords_to_segment_ring(x_mm, y_mm)
                hit.update(x_mm=x_mm, y_mm=y_mm, segment=segment, ring=ring, score=score)

            detections.append(hit)

        return detections

    # ------------------------------------------------------------------
    # Live camera helpers
    # ------------------------------------------------------------------

    def stream(self, camera_index: int = 0):
        """Generator yielding (frame, detections) from a live camera feed."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame, self.detect(frame)
        finally:
            cap.release()

    def annotate(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes and labels onto a copy of frame."""
        out = frame.copy()
        for d in detections:
            x, y = int(d["x_px"]), int(d["y_px"])
            if d["segment"] is not None:
                label = f"{d['ring']} {d['segment']}  ({d['score']})"
            else:
                label = f"dart  {d['conf']:.2f}"
            cv2.circle(out, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(out, label, (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return out
