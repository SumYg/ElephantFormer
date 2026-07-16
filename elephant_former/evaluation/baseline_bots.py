"""Baseline opponents for evaluating the board-state ElephantFormer.

All bots share a tiny interface: :meth:`Bot.select_move` returns a legal move
``(fx, fy, tx, ty)`` for the side to move, or ``None`` when there are none
(terminal position). Randomness is seeded per bot for reproducible matches.

Includes the Phase 1 reference opponent: :class:`PikafishBot`, Pikafish via UCI
restricted to a fixed node budget (``go nodes N``) — node-limited play is
hardware-fair, so results are comparable across machines.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
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


def default_engine_path() -> str:
    """The bundled Pikafish binary for this platform (Windows/Linux avx2)."""
    sub = "Windows/pikafish-avx2.exe" if sys.platform == "win32" else "Linux/pikafish-avx2"
    return str(Path("tools") / sub)


class PikafishBot(Bot):
    """Pikafish via UCI at a fixed node budget — the Phase 1 ladder opponent.

    Positions are sent as the game's initial FEN plus the full UCI move
    history, so the engine can reason about repetitions (games here always
    start from the standard initial position). If the engine's best move is not
    legal under our rules engine (rule-set edge cases), the bot falls back to
    the first legal move and counts the event in ``fallback_moves`` — a nonzero
    count after a match deserves a look.
    """

    def __init__(
        self,
        engine_path: Optional[str] = None,
        nnue_path: Optional[str] = "tools/pikafish.nnue",
        nodes: int = 1000,
        threads: int = 1,
        hash_mb: int = 64,
    ) -> None:
        # Imported here so the torch-free annotation path stays torch-free and
        # bots that never use Pikafish don't need the engine binary present.
        from elephant_former.data_utils.pikafish_annotator import PikafishEngine

        self.name = f"pikafish@{nodes}"
        self.nodes = nodes
        self.fallback_moves = 0
        self._engine = PikafishEngine(
            engine_path or default_engine_path(),
            nnue_path=nnue_path,
            threads=threads,
            hash_mb=hash_mb,
            multipv=1,
        )

    def select_move(self, game: ElephantChessGame) -> Optional[Move]:
        from elephant_former.data_utils.pikafish_annotator import move_to_uci, uci_to_move
        from elephant_former.engine.elephant_chess_game import INITIAL_BOARD_FEN

        legal = game.get_all_legal_moves_basic(game.current_player)
        if not legal:
            return None
        history = [move_to_uci(m) for m in game.move_history]
        best, _ = self._engine.analyze_fen(INITIAL_BOARD_FEN, nodes=self.nodes, moves=history)
        if best is not None:
            move = uci_to_move(best)
            if move in legal:
                return move
        self.fallback_moves += 1
        print(f"  [pikafish] engine move {best!r} not legal here; falling back (#{self.fallback_moves}).")
        return legal[0]

    def close(self) -> None:
        self._engine.close()


def make_bot(kind: str, seed: Optional[int] = None) -> Bot:
    """Factory: ``"random"`` or ``"greedy"`` (see :class:`PikafishBot` for the engine)."""
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
