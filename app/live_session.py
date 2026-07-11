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


def _dist(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _closest_match(pos: tuple, candidates: list) -> tuple | None:
    return next((c for c in candidates if _dist(pos, c) <= DART_MATCH_RADIUS_MM), None)


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
        self._known_positions: list = []  # [(x_mm, y_mm), ...] confirmed darts currently on the board
        self._pending_positions: list = []  # [(x_mm, y_mm), ...] seen once, awaiting a 2nd-frame confirmation

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
        """A hit only becomes a committed new throw once it's been seen in
        roughly the same spot on two consecutive frames. This filters
        single-frame noise -- detector jitter, or the board physically
        vibrating when a new dart lands -- that would otherwise get
        double-counted as extra throws. Confirmed dart positions are kept at
        their original first-seen location rather than overwritten with each
        frame's raw (noisy) reading, so they don't slowly drift out of match
        range over time either.
        """
        confirmed: list = []
        pending: list = []
        new_hits = []

        for hit in hits:
            pos = (hit["x_mm"], hit["y_mm"])

            match = _closest_match(pos, self._known_positions) or _closest_match(pos, confirmed)
            if match is not None:
                confirmed.append(match)
                continue

            match = _closest_match(pos, self._pending_positions)
            if match is not None:
                new_hits.append(hit)
                confirmed.append(match)
                continue

            if _closest_match(pos, pending) is None:  # not a duplicate box of an already-pending candidate
                pending.append(pos)

        self._known_positions = confirmed
        self._pending_positions = pending
        return new_hits

    def _commit_throw(self, fused: FusedThrow) -> dict:
        pid = self.game.current_player["id"]
        pname = self.game.current_player["name"]
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

        print(
            f"[round {round_number}, throw {throw_in_round}] {pname}: "
            f"{fused.ring} {fused.segment} = {fused.score_value}  "
            f"(source={fused.source}, conf={fused.conf:.2f})"
        )
        return result
