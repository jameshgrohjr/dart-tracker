"""Seed script — generates realistic dummy data for 4 players."""
import sys
import math
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import (
    init_db, create_player, create_session, end_session, record_throw,
)

random.seed(42)

SEGMENT_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

PLAYERS = [
    {"name": "Jay",  "skill": 0.55, "drift": 0.25},
    {"name": "Sam",  "skill": 0.42, "drift": 0.18},
    {"name": "Mike", "skill": 0.35, "drift": 0.30},
    {"name": "Emma", "skill": 0.48, "drift": 0.12},
]

SESSIONS_PER_PLAYER = 16
ROUNDS_PER_SESSION = 12
THROWS_PER_ROUND = 3


def segment_angle(segment: int) -> float:
    """Return the centre angle (degrees) of a dartboard segment (clockwise from top)."""
    idx = SEGMENT_ORDER.index(segment)
    return idx * 18.0


def angle_to_segment(angle_deg: float) -> int:
    """Map an angle (degrees, clockwise from top) to the nearest segment."""
    angle_norm = angle_deg % 360
    idx = int((angle_norm + 9) / 18) % 20
    return SEGMENT_ORDER[idx]


def radius_to_ring(r: float):
    """
    Return (ring, score_multiplier) from radius in mm from board centre.
    Bullseye ≤12, Bull ≤29, miss >157, triple 93-102, double 147-157, else single.
    """
    if r <= 12:
        return "bullseye", 2   # 50 pts (stored as multiplier, applied per segment)
    if r <= 29:
        return "bull", 1       # 25 pts
    if r > 157:
        return "miss", 0
    if 93 <= r <= 102:
        return "triple", 3
    if 147 <= r <= 157:
        return "double", 2
    return "single", 1


def simulate_throw(skill: float, target_segment: int):
    spread = (1 - skill) * 72
    target_ang = segment_angle(target_segment)

    angle = target_ang + random.gauss(0, spread / 2.5)
    radius = max(8, min(163, 93 + random.gauss(0, spread / 2)))

    # Polar → cartesian (angle measured clockwise from top → standard math)
    angle_rad = math.radians(angle - 90)          # rotate so 0° is "up"
    x = radius * math.cos(angle_rad)
    y = radius * math.sin(angle_rad)

    ring, multiplier = radius_to_ring(radius)

    if ring == "bullseye":
        segment = 25
        score = 50
    elif ring == "bull":
        segment = 25
        score = 25
    elif ring == "miss":
        segment = 0
        score = 0
    else:
        segment = angle_to_segment(angle)
        score = segment * multiplier

    return x, y, segment, ring, score


def main():
    init_db()
    print("Database initialised.")

    player_ids = {}
    for p in PLAYERS:
        pid = create_player(p["name"])
        player_ids[p["name"]] = pid
        print(f"  Player '{p['name']}' id={pid}")

    all_pids = list(player_ids.values())

    for p in PLAYERS:
        pid = player_ids[p["name"]]
        base_skill = p["skill"]
        drift = p["drift"]

        for s_idx in range(SESSIONS_PER_PLAYER):
            skill = min(0.92, base_skill + drift * (s_idx / SESSIONS_PER_PLAYER))
            game_type = "cricket" if s_idx % 3 == 2 else "501"

            sid = create_session(game_type, [pid], notes=f"seed session {s_idx + 1}")

            for rnd in range(1, ROUNDS_PER_SESSION + 1):
                target = random.choice(SEGMENT_ORDER)
                for throw_num in range(1, THROWS_PER_ROUND + 1):
                    x, y, seg, ring, score = simulate_throw(skill, target)
                    record_throw(
                        session_id=sid,
                        player_id=pid,
                        round_number=rnd,
                        throw_in_round=throw_num,
                        x=x, y=y,
                        segment=seg,
                        ring=ring,
                        score_value=score,
                        target_segment=target,
                    )

            end_session(sid)

        print(f"  Seeded {SESSIONS_PER_PLAYER} sessions for {p['name']}")

    print("\nSeed complete.")


if __name__ == "__main__":
    main()
