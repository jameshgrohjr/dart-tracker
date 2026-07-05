from abc import ABC, abstractmethod


class BaseGame(ABC):
    def __init__(self, session_id: int, players: list):
        self.session_id = session_id
        self.players = players
        self.current_player_idx = 0
        self.round_number = 1
        self.throw_in_round = 0
        self.finished = False

    @property
    def current_player(self):
        return self.players[self.current_player_idx]

    def next_throw(self):
        self.throw_in_round += 1
        if self.throw_in_round >= 3:
            self.throw_in_round = 0
            self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
            if self.current_player_idx == 0:
                self.round_number += 1

    @abstractmethod
    def process_throw(self, segment: int, ring: str, score: int) -> dict:
        ...

    @abstractmethod
    def get_state(self) -> dict:
        ...

    @abstractmethod
    def check_winner(self) -> "int | None":
        ...
