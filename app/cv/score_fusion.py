"""Score fusion — reconciles the camera dart-tip detector with the electronic
board's own display reading into a single agreed-upon throw.

Two separate concerns live here:

  ScoreFusion      — pure reconciliation of one camera hit + one display
                      reading for the *same* throw into a single result.
  ThrowCorrelator   — the event-timing problem: the tip detector and the
                      display reader observe asynchronously, so a display
                      change has to be matched against the nearest recent
                      camera hit before ScoreFusion can run on it.
"""
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MISMATCH_DIR = DATA_DIR / "mismatches"

CORRELATION_WINDOW_S = 2.0  # max time gap to treat a camera hit and a display change as the same throw
CHANGE_THRESHOLD = 8.0      # mean abs pixel diff above which the display crop counts as "changed"


@dataclass
class FusedThrow:
    segment: int | None
    ring: str | None
    score_value: int | None
    x_mm: float | None
    y_mm: float | None
    source: str  # "agree" | "camera_only" | "display_only" | "display_override"
    conf: float


class ScoreFusion:
    def reconcile(self, camera_hit: dict | None, display_reading: dict | None) -> FusedThrow:
        if camera_hit is None and display_reading is None:
            raise ValueError("reconcile() needs at least one of camera_hit / display_reading")

        if display_reading is None or display_reading.get("score_value") is None:
            return FusedThrow(
                segment=camera_hit["segment"], ring=camera_hit["ring"],
                score_value=camera_hit["score"], x_mm=camera_hit["x_mm"], y_mm=camera_hit["y_mm"],
                source="camera_only", conf=camera_hit["conf"],
            )

        if camera_hit is None:
            return FusedThrow(
                segment=display_reading["segment"], ring=display_reading["ring"],
                score_value=display_reading["score_value"], x_mm=None, y_mm=None,
                source="display_only", conf=display_reading["conf"],
            )

        agree = (
            camera_hit["score"] == display_reading["score_value"]
            and (display_reading["segment"] is None or camera_hit["segment"] == display_reading["segment"])
        )
        if agree:
            return FusedThrow(
                segment=camera_hit["segment"], ring=camera_hit["ring"],
                score_value=camera_hit["score"], x_mm=camera_hit["x_mm"], y_mm=camera_hit["y_mm"],
                source="agree", conf=max(camera_hit["conf"], display_reading["conf"]),
            )

        # Disagreement: trust the board's own reading of its sensor grid for
        # score/segment, but keep the camera's spatial position for the
        # heatmap. Every mismatch is logged -- these are exactly the hard
        # examples worth relabeling to fine-tune dart_model.pt later.
        self._log_mismatch(camera_hit, display_reading)
        segment = display_reading["segment"] if display_reading["segment"] is not None else camera_hit["segment"]
        ring = display_reading["ring"] if display_reading["ring"] is not None else camera_hit["ring"]
        return FusedThrow(
            segment=segment, ring=ring,
            score_value=display_reading["score_value"],
            x_mm=camera_hit["x_mm"], y_mm=camera_hit["y_mm"],
            source="display_override", conf=display_reading["conf"],
        )

    @staticmethod
    def _log_mismatch(camera_hit: dict, display_reading: dict):
        MISMATCH_DIR.mkdir(parents=True, exist_ok=True)
        row = (
            f"{time.time()},camera_score={camera_hit['score']},camera_segment={camera_hit['segment']},"
            f"display_text={display_reading['raw_text']!r},display_score={display_reading['score_value']}\n"
        )
        with open(MISMATCH_DIR / "log.csv", "a") as f:
            f.write(row)


@dataclass
class _TimestampedHit:
    hit: dict
    t: float


class ThrowCorrelator:
    """Buffers recent camera hits and matches each display-change event
    against the nearest one by timestamp, then reconciles via ScoreFusion.

    Feed it every camera detection as it happens (`add_camera_hit`) and
    every display reading whenever `display_changed()` says the crop
    actually changed (`on_display_change`).
    """

    MAX_BUFFER = 50  # safety cap against unbounded growth if pop_stale_camera_hits is never called

    def __init__(self, fusion: ScoreFusion | None = None, window_s: float = CORRELATION_WINDOW_S):
        self.fusion = fusion or ScoreFusion()
        self.window_s = window_s
        self._camera_buffer: list[_TimestampedHit] = []

    def add_camera_hit(self, hit: dict, t: float | None = None):
        t = t if t is not None else time.time()
        self._camera_buffer.append(_TimestampedHit(hit, t))
        # Not time-pruned here -- pop_stale_camera_hits() is what retires
        # unmatched hits (as camera_only throws) once they age out. This is
        # just a hard cap so a caller that never polls stale hits can't leak.
        if len(self._camera_buffer) > self.MAX_BUFFER:
            self._camera_buffer = self._camera_buffer[-self.MAX_BUFFER:]

    def pop_stale_camera_hits(self, t: float | None = None) -> list:
        """Camera hits older than the correlation window that never got
        matched to a display change -- e.g. a bounce-out or miss the board's
        own sensors didn't register. Returns and removes them so the caller
        can commit them standalone as camera_only throws."""
        t = t if t is not None else time.time()
        cutoff = t - self.window_s
        stale = [c for c in self._camera_buffer if c.t < cutoff]
        self._camera_buffer = [c for c in self._camera_buffer if c.t >= cutoff]
        return [c.hit for c in stale]

    def on_display_change(self, display_reading: dict, t: float | None = None) -> FusedThrow:
        t = t if t is not None else time.time()
        nearest, best_dt = None, None
        for c in self._camera_buffer:
            dt = abs(c.t - t)
            if dt <= self.window_s and (best_dt is None or dt < best_dt):
                nearest, best_dt = c, dt
        camera_hit = nearest.hit if nearest else None
        if nearest is not None:
            self._camera_buffer.remove(nearest)
        return self.fusion.reconcile(camera_hit, display_reading)


def display_changed(prev_crop: np.ndarray | None, crop: np.ndarray, threshold: float = CHANGE_THRESHOLD) -> bool:
    """Cheap change detector between two rectified display crops -- used to
    decide when it's worth running the (comparatively expensive) glyph
    model at all, rather than every frame."""
    if prev_crop is None or prev_crop.shape != crop.shape:
        return True
    diff = cv2.absdiff(prev_crop, crop)
    return float(diff.mean()) > threshold
