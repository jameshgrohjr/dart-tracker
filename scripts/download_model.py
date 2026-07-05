"""Download a pre-trained dart detection model from Roboflow Universe.

Steps to use:
1.  Create a free account at https://roboflow.com and grab your API key
    from https://app.roboflow.com/settings/api
2.  pip install roboflow
3.  Browse https://universe.roboflow.com and search "dart" to find a model
    you want.  Note the workspace slug, project slug, and version number.
4.  Fill in the three constants below and run:
        python scripts/download_model.py
    The weights file is saved to  data/dart_model.pt  ready for the detector.
"""
import sys
import shutil
from pathlib import Path

# ── Configure these ──────────────────────────────────────────────────────────
ROBOFLOW_API_KEY = "O8T1h6HgaOfi0eO9KXpB"
WORKSPACE         = "score-lpfmv"
PROJECT           = "darts-bjj98"
VERSION           = 2
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = Path(__file__).parent.parent / "data"
MODEL_DEST = DATA_DIR / "dart_model.pt"


def main():
    if ROBOFLOW_API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Fill in ROBOFLOW_API_KEY, WORKSPACE, and PROJECT before running.")
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow package not installed.  Run:  pip install roboflow")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to Roboflow workspace '{WORKSPACE}' ...")
    rf      = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)

    print(f"Downloading YOLOv8 weights for {PROJECT} v{VERSION} ...")
    model_dir = Path(version.download("yolov8pytorch").location)

    # Roboflow downloads into a subdirectory; find the .pt file
    pt_files = list(model_dir.rglob("*.pt"))
    if not pt_files:
        print(f"ERROR: No .pt file found under {model_dir}")
        sys.exit(1)

    shutil.copy(pt_files[0], MODEL_DEST)
    print(f"Model saved to {MODEL_DEST}")
    print("Run  python scripts/calibrate.py  next if you haven't calibrated yet.")


if __name__ == "__main__":
    main()
