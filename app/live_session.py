"""Live session orchestrator.

Wires the camera dart-tip detector (app.cv.detector.DartDetector), the
electronic board's display reader (app.cv.display_reader.DisplayScoreReader),
and score fusion (app.cv.score_fusion) into a running game, persisting each
fused throw via app.database.record_throw.

The display_reader is optional: with none supplied, every newly-detected
camera hit is committed immediately as a camera_only throw -- i.e. this
degrades to the original webcam-only pipeline. This lets the same
orchestrator run on setups without an electronic board.
"""
import math
import time

import cv2

from app.cv.detector import DartDetector
from app.cv.display_reader import DisplayScoreReader
from app.cv.display_calibrator import rectify
from app.cv.score_fusion import FusedThrow, ScoreFusion, ThrowCorrelator, display_changed
from app.database import record_throw

# Dedup threshold: a hit within this radius of an already-tracked dart is
# treated as the same physical dart, not a new throw. DartDetector re-detects
# every dart still stuck in the board on every frame, so without this a
# single dart would be recorded as a fresh throw on every processed frame.
DART_MATCH_RADIUS_MM = 8.0


def _dist_mm(hit: dict, known_xy: tuple) -> float:
    return math.hypot(hit["x_mm"] - known_xy[0], hit["y_mm"] - known_xy[1])


class LiveSession:
    def __init__(
        self,
        game,                      # a BaseGame subclass instance, already created
        session_id: int,
        detector: DartDetector,
        display_reader: DisplayScoreReader | None = None,
        correlator: ThrowCorrelator | None = None,
    ):
        self.game = game
        self.session_id = session_id
        self.detector = detector
        self.display_reader = display_reader
        self.correlator = correlator or ThrowCorrelator()
        self._prev_display_crop = None
        self._known_positions: list = []  # [(x_mm, y_mm), ...] of darts currently on the board

    def run(self, camera_index: int = 0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}.")
        try:
            while not self.game.finished:
                ret, frame = cap.read()
                if not ret:
                    break
                self.process_frame(frame)
        finally:
            cap.release()

    def process_frame(self, frame, t: float | None = None) -> list:
        """Feed one frame through detection/fusion. Returns any game results
        committed as a result (usually 0 or 1, occasionally more if several
        stale camera-only hits retire on the same frame)."""
        t = t if t is not None else time.time()
        committed = []

        raw_hits = [h for h in self.detector.detect(frame) if h["segment"] is not None]
        new_hits = self._new_hits_since_last_frame(raw_hits)

        if self.display_reader is None:
            for hit in new_hits:
                fused = self.correlator.fusion.reconcile(hit, None)
                committed.append(self._commit_throw(fused))
            return committed

        for hit in new_hits:
            self.correlator.add_camera_hit(hit, t=t)

        crop = rectify(frame, self.display_reader.calibration)
        if display_changed(self._prev_display_crop, crop):
            reading = self.display_reader.read(frame)
            if reading["score_value"] is not None:
                fused = self.correlator.on_display_change(reading, t=t)
                committed.append(self._commit_throw(fused))
        self._prev_display_crop = crop

        for stale_hit in self.correlator.pop_stale_camera_hits(t=t):
            fused = self.correlator.fusion.reconcile(stale_hit, None)
            committed.append(self._commit_throw(fused))

        return committed

    def _new_hits_since_last_frame(self, hits: list) -> list:
        new = [
            hit for hit in hits
            if not any(_dist_mm(hit, known) <= DART_MATCH_RADIUS_MM for known in self._known_positions)
        ]
        self._known_positions = [(h["x_mm"], h["y_mm"]) for h in hits]
        return new

    def _commit_throw(self, fused: FusedThrow) -> dict:
        pid = self.game.current_player["id"]
        round_number = self.game.round_number
        throw_in_round = self.game.throw_in_round + 1  # process_throw() below advances state

        result = self.game.process_throw(fused.segment, fused.ring, fused.score_value)

        record_throw(
            session_id=self.session_id,
            player_id=pid,
            round_number=round_number,
            throw_in_round=throw_in_round,
            x=fused.x_mm, y=fused.y_mm,
            segment=fused.segment, ring=fused.ring, score_value=fused.score_value,
        )
        return result
