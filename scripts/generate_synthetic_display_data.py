"""Generate synthetic training images of an electronic dartboard's LED score display.

Bootstraps a glyph-detection dataset before you have real captures. Renders
plausible score strings ("T20", "D16", "BULL", "141", ...) in a seven-segment
font, then augments each image (blur, rotation, perspective warp, brightness/
color jitter, noise) to approximate a camera looking at a real LED display.
Outputs YOLO-format detection labels (one box per character) so it drops
straight into an ultralytics training run or a Roboflow upload, and merges
cleanly with real captures later (same class list, same label format).

Steps to use:
1.  A seven-segment font is already vendored at data/fonts/DSEG7Classic-Bold.ttf
    (DSEG by Keshikan, SIL OFL license -- see data/fonts/OFL.txt). Swap
    FONT_PATH below if your board's display looks different (e.g. DSEG14 for
    a 14-segment alphanumeric display, or a different weight).
2.  Edit CANDIDATE_TOKENS to match what your board's display actually shows
    (check whether it displays multiplier prefixes like "T20"/"D16" or just
    plain numbers -- this determines what the display-reader model can tell
    you beyond a raw score_value).
3.  Run:  python scripts/generate_synthetic_display_data.py
    Images + YOLO labels are written to data/synthetic_display/.
"""
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

# ── Configure these ──────────────────────────────────────────────────────────
FONT_PATH   = str(Path(__file__).parent.parent / "data" / "fonts" / "DSEG7Classic-Bold.ttf")
NUM_IMAGES  = 2000
IMG_W_RANGE = (420, 640)
IMG_H_RANGE = (120, 180)
LED_COLORS  = [(255, 40, 40), (60, 255, 90), (255, 170, 30), (60, 160, 255)]  # red/green/amber/blue
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR    = Path(__file__).parent.parent / "data" / "synthetic_display"
IMAGES_DIR = OUT_DIR / "images"
LABELS_DIR = OUT_DIR / "labels"

# What the display might show after a throw. Edit to match your board.
CANDIDATE_TOKENS = (
    [str(n) for n in range(0, 181)]                      # raw scores / running totals
    + [f"T{n}" for n in range(1, 21)]                     # triple prefix
    + [f"D{n}" for n in range(1, 21)]                     # double prefix
    + ["25", "50", "BULL", "MISS", "OUT"]
)

CLASSES = sorted({ch for tok in CANDIDATE_TOKENS for ch in tok})


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        raise SystemExit(
            f"Could not load font at FONT_PATH={FONT_PATH!r}. "
            "Download a seven-segment TTF (e.g. DSEG) and update FONT_PATH."
        )


def _render_token(token: str, canvas_w: int, canvas_h: int, color: tuple) -> tuple:
    """Draw `token` centered on a dark canvas. Returns (PIL image RGB, [ (char, box) ])."""
    bg_shade = random.randint(4, 22)
    img = Image.new("RGB", (canvas_w, canvas_h), (bg_shade, bg_shade, bg_shade))
    draw = ImageDraw.Draw(img)

    font_size = int(canvas_h * random.uniform(0.55, 0.85))
    font = _load_font(font_size)

    total_w = draw.textlength(token, font=font)
    x = (canvas_w - total_w) / 2.0
    bbox_full = draw.textbbox((0, 0), token, font=font)
    y = (canvas_h - (bbox_full[3] - bbox_full[1])) / 2.0 - bbox_full[1]

    boxes = []
    cx = x
    for ch in token:
        ch_bbox = draw.textbbox((cx, y), ch, font=font)
        draw.text((cx, y), ch, font=font, fill=color)
        boxes.append((ch, ch_bbox))
        cx += draw.textlength(ch, font=font)

    # LED glow: blurred bright copy screen-blended under the sharp text
    glow = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(3, 7)))
    img = ImageChops.screen(img, glow)

    return img, boxes


def _augment(img_rgb: np.ndarray, boxes_xyxy: list) -> tuple:
    """Apply rotation + perspective warp to image and boxes together, then
    photometric noise (which doesn't need to touch the boxes)."""
    h, w = img_rgb.shape[:2]

    # --- rotation (affine) ---
    angle = random.uniform(-6, 6)
    rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    bg = tuple(int(v) for v in img_rgb[0, 0])
    img_rgb = cv2.warpAffine(img_rgb, rot_mat, (w, h), borderValue=bg)

    def transform_affine(pts):
        pts = np.array(pts, dtype=np.float32)
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        return (rot_mat @ np.hstack([pts, ones]).T).T

    # --- perspective warp ---
    jitter = 0.05
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.uniform(0, jitter) * w, random.uniform(0, jitter) * h],
        [w - random.uniform(0, jitter) * w, random.uniform(0, jitter) * h],
        [w - random.uniform(0, jitter) * w, h - random.uniform(0, jitter) * h],
        [random.uniform(0, jitter) * w, h - random.uniform(0, jitter) * h],
    ])
    persp_mat = cv2.getPerspectiveTransform(src, dst)
    img_rgb = cv2.warpPerspective(img_rgb, persp_mat, (w, h), borderValue=bg)

    def transform_persp(pts):
        pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, persp_mat).reshape(-1, 2)

    new_boxes = []
    for x1, y1, x2, y2 in boxes_xyxy:
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        corners = transform_affine(corners)
        corners = transform_persp(corners)
        xs, ys = corners[:, 0], corners[:, 1]
        new_boxes.append((float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())))

    # --- photometric jitter (image only) ---
    img_f = img_rgb.astype(np.float32)
    img_f *= random.uniform(0.7, 1.3)                       # brightness
    img_f += np.random.normal(0, random.uniform(2, 10), img_f.shape)  # sensor noise
    img_rgb = np.clip(img_f, 0, 255).astype(np.uint8)

    if random.random() < 0.7:
        k = random.choice([3, 5])
        img_rgb = cv2.GaussianBlur(img_rgb, (k, k), 0)

    return img_rgb, new_boxes


def _to_yolo_line(cls_idx: int, box_xyxy: tuple, img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = box_xyxy
    x1, x2 = np.clip([x1, x2], 0, img_w)
    y1, y2 = np.clip([y1, y2], 0, img_h)
    xc = (x1 + x2) / 2 / img_w
    yc = (y1 + y2) / 2 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return f"{cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    written = 0
    for i in range(NUM_IMAGES):
        token = random.choice(CANDIDATE_TOKENS)
        color = random.choice(LED_COLORS)
        w = random.randint(*IMG_W_RANGE)
        h = random.randint(*IMG_H_RANGE)

        pil_img, char_boxes = _render_token(token, w, h, color)
        img_rgb = np.array(pil_img)
        boxes_xyxy = [(b[0], b[1], b[2], b[3]) for _, b in char_boxes]

        img_rgb, boxes_xyxy = _augment(img_rgb, boxes_xyxy)

        # drop samples where augmentation pushed a char box mostly off-canvas
        kept_boxes = []
        for (ch, _), box in zip(char_boxes, boxes_xyxy):
            x1, y1, x2, y2 = box
            if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
                continue
            kept_boxes.append((ch, box))
        if len(kept_boxes) != len(char_boxes):
            continue  # skip rather than emit a mislabeled sample

        stem = f"synth_{i:05d}"
        cv2.imwrite(str(IMAGES_DIR / f"{stem}.jpg"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        lines = [_to_yolo_line(class_to_idx[ch], box, w, h) for ch, box in kept_boxes]
        (LABELS_DIR / f"{stem}.txt").write_text("\n".join(lines))
        written += 1

    (OUT_DIR / "classes.txt").write_text("\n".join(CLASSES))
    (OUT_DIR / "data.yaml").write_text(
        f"path: {OUT_DIR.resolve()}\n"
        f"train: images\n"
        f"val: images\n"
        f"names:\n" + "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES)) + "\n"
    )

    print(f"Wrote {written}/{NUM_IMAGES} synthetic samples to {OUT_DIR}")
    print(f"Classes ({len(CLASSES)}): {CLASSES}")
    print("Next: python -c \"from ultralytics import YOLO; "
          "YOLO('yolov8n.pt').train(data='data/synthetic_display/data.yaml', epochs=50)\" "
          "-- or upload data/synthetic_display/ to Roboflow to merge with real captures later.")


if __name__ == "__main__":
    main()
