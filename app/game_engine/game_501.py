from app.game_engine.base import BaseGame


class Game501(BaseGame):
    def __init__(self, session_id: int, players: list, start_score: int = 501):
        super().__init__(session_id, players)
        self.start_score = start_score
        pids = [p["id"] for p in players]
        self.scores = {pid: start_score for pid in pids}
        self.checkout_attempts = {pid: 0 for pid in pids}
        self.checkout_hits = {pid: 0 for pid in pids}

    def process_throw(self, segment: int, ring: str, score: int) -> dict:
        pid = self.current_player["id"]
        remaining = self.scores[pid]
        result = {"bust": False, "winner": False, "scored": 0}

        if remaining - score < 0:
            result["bust"] = True
            self.next_throw()
            return result

        if remaining - score == 0 and ring == "double":
            self.scores[pid] = 0
            self.checkout_hits[pid] += 1
            self.finished = True
            result["winner"] = True
            result["scored"] = score
            self.next_throw()
            return result

        if remaining - score == 0 and ring != "double":
            result["bust"] = True
            self.next_throw()
            return result

        if remaining <= 40 and ring == "double":
            self.checkout_attempts[pid] += 1
            self.scores[pid] -= score
            result["scored"] = score
        else:
            self.scores[pid] -= score
            result["scored"] = score

        self.next_throw()
        return result

    def get_state(self) -> dict:
        return {
            "round": self.round_number,
            "current_player": self.current_player,
            "scores": self.scores,
            "throw_in_round": self.throw_in_round,
        }

    def check_winner(self) -> "int | None":
        for pid, s in self.scores.items():
            if s == 0:
                return pid
        return None
