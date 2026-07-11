"""Merge a labeled Roboflow YOLO export of real display captures into the training set.

After collecting real captures with scripts/capture_display_labels.py and
labeling them in Roboflow (draw a box around each character, export as
"YOLOv8"), point this script at the extracted export folder. It copies
those images + labels into data/synthetic_display/, remapping class
indices if Roboflow's class list differs from ours, so a normal
scripts/train_display_model.py run picks up both synthetic and real data
without any other changes.

Usage: python scripts/merge_display_dataset.py path/to/roboflow_export
(the export folder should contain train/valid/test subfolders, each with
images/ and labels/, plus a data.yaml -- the standard Roboflow YOLOv8 layout)
"""
import shutil
import sys
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).parent.parent / "data"
DEST_DIR = DATA_DIR / "synthetic_display"
DEST_IMAGES = DEST_DIR / "images"
DEST_LABELS = DEST_DIR / "labels"


def _load_classes(path: Path) -> list:
    return [c for c in path.read_text().splitlines() if c.strip()]


def _load_roboflow_classes(export_dir: Path) -> list:
    data_yaml = export_dir / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"No data.yaml found in {export_dir} -- is this a YOLOv8-format Roboflow export?")
    cfg = yaml.safe_load(data_yaml.read_text())
    names = cfg["names"]
    # Class names are often bare digits (e.g. "0", "1", "T", "D", ...) -- YAML
    # parses unquoted numeric scalars as ints, so cast back to str explicitly.
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names, key=int)]
    return [str(n) for n in names]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/merge_display_dataset.py path/to/roboflow_export")
    export_dir = Path(sys.argv[1])
    if not export_dir.exists():
        raise SystemExit(f"{export_dir} does not exist")

    existing_classes_path = DEST_DIR / "classes.txt"
    existing_classes = _load_classes(existing_classes_path) if existing_classes_path.exists() else []
    roboflow_classes = _load_roboflow_classes(export_dir)

    merged_classes = list(existing_classes)
    for c in roboflow_classes:
        if c not in merged_classes:
            merged_classes.append(c)
    remap = {i: merged_classes.index(name) for i, name in enumerate(roboflow_classes)}

    DEST_IMAGES.mkdir(parents=True, exist_ok=True)
    DEST_LABELS.mkdir(parents=True, exist_ok=True)

    copied = 0
    for split_dir in sorted(export_dir.iterdir()):
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue
        for img_path in sorted(images_dir.iterdir()):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            stem = f"real_{split_dir.name}_{img_path.stem}"
            shutil.copy(img_path, DEST_IMAGES / f"{stem}{img_path.suffix}")

            lines = []
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                new_idx = remap[int(parts[0])]
                lines.append(" ".join([str(new_idx)] + parts[1:]))
            (DEST_LABELS / f"{stem}.txt").write_text("\n".join(lines))
            copied += 1

    existing_classes_path.write_text("\n".join(merged_classes))
    (DEST_DIR / "data.yaml").write_text(
        f"path: {DEST_DIR.resolve()}\n"
        f"train: images\n"
        f"val: images\n"
        f"names:\n" + "\n".join(f"  {i}: {c}" for i, c in enumerate(merged_classes)) + "\n"
    )

    print(f"Merged {copied} real-captured images into {DEST_DIR}")
    print(f"Classes ({len(merged_classes)}): {merged_classes}")
    print("Next: python scripts/train_display_model.py")


if __name__ == "__main__":
    main()
