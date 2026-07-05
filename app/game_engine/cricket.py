from app.game_engine.base import BaseGame

CRICKET_NUMBERS = [15, 16, 17, 18, 19, 20, 25]


def throw_to_marks(ring: str) -> int:
    return {"single": 1, "double": 2, "triple": 3, "bull": 1, "bullseye": 2}.get(ring, 0)


class CricketGame(BaseGame):
    def __init__(self, session_id: int, players: list):
        super().__init__(session_id, players)
        pids = [p["id"] for p in players]
        self.marks = {pid: {n: 0 for n in CRICKET_NUMBERS} for pid in pids}
        self.points = {pid: 0 for pid in pids}

    def process_throw(self, segment: int, ring: str, score: int) -> dict:
        pid = self.current_player["id"]
        result = {"scored_marks": 0, "scored_points": 0, "segment": segment}

        if segment in CRICKET_NUMBERS:
            new_marks = throw_to_marks(ring)
            current = self.marks[pid][segment]
            overflow = max(0, (current + new_marks) - 3)
            self.marks[pid][segment] = min(3, current + new_marks)
            result["scored_marks"] = new_marks - overflow

            if self.marks[pid][segment] >= 3 and overflow > 0:
                others_closed = all(
                    self.marks[other_pid][segment] >= 3
                    for other_pid in self.marks
                    if other_pid != pid
                )
                if not others_closed:
                    pts = overflow * (25 if segment == 25 else segment)
                    self.points[pid] += pts
                    result["scored_points"] = pts

        self.next_throw()
        return result

    def get_state(self) -> dict:
        return {
            "round": self.round_number,
            "current_player": self.current_player,
            "marks": self.marks,
            "points": self.points,
            "throw_in_round": self.throw_in_round,
        }

    def check_winner(self) -> "int | None":
        max_points = max(self.points.values())
        for pid in self.marks:
            all_closed = all(self.marks[pid][n] >= 3 for n in CRICKET_NUMBERS)
            if all_closed and self.points[pid] >= max_points:
                return pid
        return None
