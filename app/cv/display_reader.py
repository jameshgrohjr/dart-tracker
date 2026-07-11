"""Display reader — YOLO glyph detection on a rectified electronic dartboard display crop."""
import re
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from app.cv.display_calibrator import load_display_calibration, rectify

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_MODEL_PATH = DATA_DIR / "display_model.pt"

# Optional T/D multiplier prefix followed by 1-3 digits, e.g. "T20", "D16", "141".
_SCORE_RE = re.compile(r"^(T|D)?(\d{1,3})$")

WORD_SCORES = {
    "BULL": 50,
    "MISS": 0,
}


class DisplayScoreReader:
    def __init__(
        self,
        model_path: str | Path | None = None,
        calibration_path: str | Path | None = None,
        conf_threshold: float = 0.4,
    ):
        model_path = Path(model_path or DEFAULT_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Display model not found at {model_path}. "
                "Run  python scripts/train_display_model.py  to train it."
            )

        self.model = YOLO(str(model_path))
        self.conf_threshold = conf_threshold
        self.calibration = load_display_calibration(calibration_path)

    def read(self, frame: np.ndarray) -> dict:
        """Rectify the display region of `frame` and read its current text.

        Returns a dict with:
            raw_text      — assembled left-to-right glyph string (e.g. "T20", "BULL")
            score_value   — parsed integer score, or None if unparseable
            segment, ring — parsed dartboard result when the string unambiguously
                             encodes one (multiplier prefix, BULL/MISS); otherwise None
            conf          — mean detection confidence across glyphs, 0.0 if none found
        """
        crop = rectify(frame, self.calibration)
        glyphs = self._detect_glyphs(crop)

        glyphs.sort(key=lambda g: g[0])
        raw_text = "".join(g[1] for g in glyphs)
        conf = float(np.mean([g[2] for g in glyphs])) if glyphs else 0.0

        score_value, segment, ring = self._parse(raw_text)

        return {
            "raw_text": raw_text,
            "score_value": score_value,
            "segment": segment,
            "ring": ring,
            "conf": conf,
        }

    def _detect_glyphs(self, crop: np.ndarray) -> list:
        """Run the glyph model; return [(x1, char, conf), ...] unsorted."""
        results = self.model(crop, conf=self.conf_threshold, verbose=False)[0]
        glyphs = []
        for box in results.boxes:
            x1 = float(box.xyxy[0][0])
            cls_idx = int(box.cls[0])
            char = self.model.names[cls_idx]
            glyphs.append((x1, char, float(box.conf[0])))
        return glyphs

    @staticmethod
    def _parse(text: str) -> tuple:
        """Best-effort parse of the assembled glyph string.

        T/D-prefixed strings and BULL/MISS unambiguously map to a single
        dart's segment+ring. A bare number's meaning (the dart just thrown
        vs. a running total) depends on what your board actually displays --
        callers should only trust score_value for bare numbers once they've
        confirmed the board shows a per-dart value rather than a running
        total (see CANDIDATE_TOKENS in generate_synthetic_display_data.py).
        Unrecognized tokens (e.g. board-specific words not in WORD_SCORES)
        return (None, None, None) rather than guessing.
        """
        if text in WORD_SCORES:
            value = WORD_SCORES[text]
            if text == "BULL":
                return value, 25, "bullseye"
            return value, 0, "miss"

        m = _SCORE_RE.match(text)
        if not m:
            return None, None, None

        prefix, digits = m.groups()
        value = int(digits)

        if prefix == "T":
            return value * 3, value, "triple"
        if prefix == "D":
            return value * 2, value, "double"
        if value == 25:
            return 25, 25, "bull"
        if value == 50:
            return 50, 25, "bullseye"
        return value, None, None
