"""PikafishBot: live-engine opponent smoke test (needs the bundled binary).

Run: ``uv run python tests/test_pikafish_bot.py`` (exits non-zero on failure).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elephant_former.engine.elephant_chess_game import ElephantChessGame
from elephant_former.evaluation.baseline_bots import PikafishBot, RandomBot, default_engine_path


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_live_engine_plays_legal_moves() -> None:
    _check(Path(default_engine_path()).exists(), f"bundled engine missing: {default_engine_path()}")
    bot = PikafishBot(nodes=64)  # tiny budget: this is a plumbing test, not a strength test
    try:
        game = ElephantChessGame()
        opponent = RandomBot(seed=0)
        for ply in range(8):
            mover = bot if ply % 2 == 0 else opponent
            legal = game.get_all_legal_moves_basic(game.current_player)
            move = mover.select_move(game)
            _check(move is not None, f"no move at ply {ply}")
            _check(move in legal, f"illegal move {move} at ply {ply} from {mover.name}")
            game.apply_move(move)
        _check(bot.fallback_moves == 0, "engine moves should be legal from the start position")
    finally:
        bot.close()
    print("  [ok] live engine plays 4 legal moves vs random")


if __name__ == "__main__":
    print("Running Pikafish bot tests:")
    test_live_engine_plays_legal_moves()
    print("All Pikafish bot tests passed.")
