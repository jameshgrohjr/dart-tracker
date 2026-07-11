"""Train the LED display glyph-detection model.

Trains a YOLOv8 model to detect+classify individual characters on the
electronic dartboard's display crop (see app/cv/display_calibrator.py),
starting from data/synthetic_display/ (see generate_synthetic_display_data.py).
Once you've collected and labeled real captures (scripts/capture_display_labels.py
+ Roboflow), merge them into the same YOLO-format dataset and re-run this to
replace data/display_model.pt with a far more accurate version -- no other
code needs to change, DisplayScoreReader just loads whatever .pt is there.

Usage: python scripts/train_display_model.py
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

DATA_DIR = Path(__file__).parent.parent / "data"

DATA_YAML    = DATA_DIR / "synthetic_display" / "data.yaml"
BASE_WEIGHTS = Path(__file__).parent.parent / "yolov8n.pt"
DEST_WEIGHTS = DATA_DIR / "display_model.pt"

RUNS_DIR = DATA_DIR / "runs"
RUN_NAME = "display_model"

EPOCHS = 40
IMGSZ  = 320
BATCH  = 16


def main():
    if not DATA_YAML.exists():
        raise SystemExit(
            f"{DATA_YAML} not found. Run  python scripts/generate_synthetic_display_data.py  first."
        )

    model = YOLO(str(BASE_WEIGHTS))
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        verbose=False,
    )

    best = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"ERROR: expected weights at {best}, not found.")

    shutil.copy(best, DEST_WEIGHTS)
    print(f"Display model saved to {DEST_WEIGHTS}")


if __name__ == "__main__":
    main()
