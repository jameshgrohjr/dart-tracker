"""CLI entrypoint for a live dart-tracking session.

Wires the camera dart-tip detector (+ optionally an electronic board's
display reader) into a running game via app.live_session.LiveSession,
persisting every fused throw to the database in real time.

Requires camera calibration (scripts/calibrate.py) and, if using an
electronic board, display calibration + a trained display model
(scripts/calibrate_display.py, scripts/train_display_model.py).

Usage: python scripts/run_live_session.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import init_db, create_player, create_session, end_session
from app.game_engine.cricket import CricketGame
from app.game_engine.game_501 import Game501
from app.cv.detector import DartDetector
from app.cv.display_reader import DisplayScoreReader
from app.live_session import LiveSession

CAM_INDEX = 1
USE_DISPLAY_READER = True  # set False to run camera-only (no electronic board)
CONFIRM_THROWS = True      # prompt Y/n before each detected throw is recorded, to reject phantom detections

GAME_TYPES = ("cricket", "501", "301")


def _make_game(game_type: str, session_id: int, players: list):
    if game_type == "cricket":
        return CricketGame(session_id, players)
    if game_type == "501":
        return Game501(session_id, players, start_score=501)
    if game_type == "301":
        return Game501(session_id, players, start_score=301)
    raise ValueError(f"Unknown game type {game_type!r}")


def main():
    init_db()

    names = [n.strip() for n in input("Player names, comma-separated: ").split(",") if n.strip()]
    if not names:
        print("No players entered.")
        sys.exit(1)

    game_type = input(f"Game type ({'/'.join(GAME_TYPES)}): ").strip().lower()
    if game_type not in GAME_TYPES:
        print(f"Unknown game type {game_type!r}.")
        sys.exit(1)

    player_ids = [create_player(n) for n in names]
    session_id = create_session(game_type, player_ids)
    players = [{"id": pid, "name": name} for pid, name in zip(player_ids, names)]
    game = _make_game(game_type, session_id, players)

    detector = DartDetector()

    display_reader = None
    if USE_DISPLAY_READER:
        try:
            display_reader = DisplayScoreReader()
        except FileNotFoundError as e:
            print(f"Display reader unavailable, running camera-only: {e}")

    session = LiveSession(
        game=game, session_id=session_id, detector=detector, display_reader=display_reader,
        require_confirmation=CONFIRM_THROWS,
    )

    print(f"Session {session_id} ({game_type}) started for: {', '.join(names)}")
    if CONFIRM_THROWS:
        print("Throw darts. You'll be asked to confirm each detected throw -- press Enter to accept, 'n' to reject.")
    else:
        print("Throw darts. Ctrl+C to stop early.")
    try:
        session.run(camera_index=CAM_INDEX)
    except KeyboardInterrupt:
        pass

    end_session(session_id)
    print(f"Session {session_id} ended.")


if __name__ == "__main__":
    main()
