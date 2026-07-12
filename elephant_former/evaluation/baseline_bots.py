"""Baseline opponents for evaluating the board-state ElephantFormer (Phase 0).

All bots share a tiny interface: :meth:`Bot.select_move` returns a legal move
``(fx, fy, tx, ty)`` for the side to move, or ``None`` when there are none
(terminal position). Randomness is seeded per bot for reproducible matches.
"""

from __future__ import annotations

import random
from typing import Optional

from elephant_former.engine.elephant_chess_game import (
    ElephantChessGame,
    Move,
    R_ADVISOR,
    R_CANNON,
    R_CHARIOT,
    R_ELEPHANT,
    R_HORSE,
    R_KING,
    R_SOLDIER,
)

# Xiangqi material values used by the greedy bot. The soldier is scored 1 before
# crossing the river and 2 after (see ``_soldier_crossed_river``). The king value
# is only used to prefer flying-general / king captures; it is effectively "win".
PIECE_VALUES = {
    R_KING: 1000.0,
    R_CHARIOT: 9.0,
    R_CANNON: 4.5,
    R_HORSE: 4.0,
    R_ADVISOR: 2.0,
    R_ELEPHANT: 2.0,
    R_SOLDIER: 1.0,  # promoted to 2.0 once across the river
}


def _soldier_crossed_river(piece: int, y: int) -> bool:
    """True if a soldier at rank ``y`` has crossed the river (into enemy half)."""
    if piece > 0:  # red soldier moves up; enemy half is ranks 5..9
        return y >= 5
    return y <= 4  # black soldier moves down; enemy half is ranks 0..4


def captured_value(game: ElephantChessGame, move: Move) -> float:
    """Material value of the piece captured by ``move`` (0.0 if it is not a capture)."""
    _, _, tx, ty = move
    target = int(game.board[ty, tx])
    if target == 0:
        return 0.0
    value = PIECE_VALUES.get(abs(target), 0.0)
    if abs(target) == R_SOLDIER and _soldier_crossed_river(target, ty):
        value = 2.0
    return value


class Bot:
    """Base bot interface."""

    name: str = "bot"

    def select_move(self, game: ElephantChessGame) -> Optional[Move]:
        raise NotImplementedError


class RandomBot(Bot):
    """Plays a uniformly random legal move (the legacy random opponent)."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.name = "random"
        self._rng = random.Random(seed)

    def select_move(self, game: ElephantChessGame) -> Optional[Move]:
        legal = game.get_all_legal_moves_basic(game.current_player)
        if not legal:
            return None
        return self._rng.choice(legal)


class GreedyMaterialBot(Bot):
    """Captures the highest-value piece available; otherwise plays randomly.

    Ties among best captures, and positions with no capture, are broken by a
    seeded random choice.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.name = "greedy"
        self._rng = random.Random(seed)

    def select_move(self, game: ElephantChessGame) -> Optional[Move]:
        legal = game.get_all_legal_moves_basic(game.current_player)
        if not legal:
            return None

        best_value = 0.0
        best_moves = []
        for move in legal:
            value = captured_value(game, move)
            if value > best_value:
                best_value = value
                best_moves = [move]
            elif value == best_value and value > 0.0:
                best_moves.append(move)

        if best_value > 0.0 and best_moves:
            return self._rng.choice(best_moves)
        return self._rng.choice(legal)


def make_bot(kind: str, seed: Optional[int] = None) -> Bot:
    """Factory: ``"random"`` or ``"greedy"``."""
    if kind == "random":
        return RandomBot(seed=seed)
    if kind == "greedy":
        return GreedyMaterialBot(seed=seed)
    raise ValueError(f"Unknown bot kind: {kind!r}")


if __name__ == "__main__":
    game = ElephantChessGame()
    rb = RandomBot(seed=0)
    gb = GreedyMaterialBot(seed=0)
    print("random move:", rb.select_move(game))
    print("greedy move:", gb.select_move(game))
