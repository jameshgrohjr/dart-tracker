"""Rectify an electronic dartboard's LED display region from a calibrated homography."""
import json
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_CALIBRATION_PATH = DATA_DIR / "display_calibration.json"


def load_display_calibration(path: str | Path | None = None) -> dict:
    path = Path(path or DEFAULT_CALIBRATION_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Display calibration not found at {path}. "
            "Run  python scripts/calibrate_display.py  first."
        )
    with open(path) as f:
        return json.load(f)


def rectify(frame: np.ndarray, cal: dict) -> np.ndarray:
    """Warp the calibrated display region of `frame` into a straightened crop."""
    H = np.array(cal["homography"], dtype=np.float64)
    w, h = cal["output_size"]
    return cv2.warpPerspective(frame, H, (w, h))
