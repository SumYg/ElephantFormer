"""Tests for the Pikafish annotation pipeline (FEN, move mapping, UCI parsing).

Run: ``uv run python tests/test_pikafish_annotator.py`` (exits non-zero on failure).
The live-engine test is skipped when the Pikafish binary is not present.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elephant_former.data_utils import board_features as bf
from elephant_former.data_utils.pikafish_annotator import (
    PikafishEngine,
    game_to_fen,
    move_to_uci,
    parse_info_lines,
    piece_ids_to_fen,
    uci_to_move,
)
from elephant_former.data_utils.tokenization_utils import parse_iccs_move_to_coords
from elephant_former.engine.elephant_chess_game import ElephantChessGame

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
AFTER_H2E2_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C4/9/RNBAKABNR b - - 0 1"

ENGINE = Path(__file__).resolve().parents[1] / "tools" / "Windows" / "pikafish-bmi2.exe"
NNUE = Path(__file__).resolve().parents[1] / "tools" / "pikafish.nnue"


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_fen() -> None:
    game = ElephantChessGame()
    _check(game_to_fen(game) == START_FEN, f"start FEN wrong: {game_to_fen(game)}")

    game.apply_move(parse_iccs_move_to_coords("H2-E2"))
    _check(game_to_fen(game) == AFTER_H2E2_FEN, f"post-H2-E2 FEN wrong: {game_to_fen(game)}")

    feats = bf.extract_features(ElephantChessGame())
    _check(
        piece_ids_to_fen(feats.piece_ids, feats.side_to_move) == START_FEN,
        "piece_ids_to_fen disagrees with game_to_fen on the start position",
    )
    print("  [ok] FEN serialization")


def test_move_mapping() -> None:
    _check(move_to_uci((7, 2, 4, 2)) == "h2e2", "move_to_uci wrong for (7,2,4,2)")
    _check(uci_to_move("h2e2") == (7, 2, 4, 2), "uci_to_move wrong for h2e2")
    for move in [(0, 0, 8, 9), (4, 9, 4, 8), (1, 2, 1, 9)]:
        _check(uci_to_move(move_to_uci(move)) == move, f"uci round-trip failed for {move}")
    print("  [ok] uci move mapping")


def test_info_parsing() -> None:
    lines = [
        "info depth 8 seldepth 10 multipv 1 score cp 45 nodes 9000 pv h2e2 b9c7",
        "info depth 8 seldepth 10 multipv 2 score cp 12 nodes 9000 pv b2e2 h9g7",
        "info depth 9 seldepth 12 multipv 1 score cp 51 nodes 12000 pv h2e2 h9g7",
        "info depth 9 seldepth 11 multipv 2 score mate 3 nodes 12000 pv b2e2",
        "bestmove h2e2 ponder h9g7",
    ]
    pvs = parse_info_lines(lines)
    _check(len(pvs) == 2, f"expected 2 pv lines, got {len(pvs)}")
    _check(pvs[0].move == "h2e2" and pvs[0].cp == 51 and not pvs[0].is_mate, "pv1 wrong")
    _check(pvs[1].move == "b2e2" and pvs[1].is_mate and pvs[1].cp == 29997, "pv2 mate fold wrong")
    print("  [ok] uci info parsing")


def test_live_engine() -> None:
    if not ENGINE.exists():
        print("  [skip] live engine test (binary not found)")
        return
    game = ElephantChessGame()
    legal = set(game.get_all_legal_moves_basic(game.current_player))
    with PikafishEngine(ENGINE, nnue_path=NNUE, multipv=4) as engine:
        best, pvs = engine.analyze_fen(START_FEN, nodes=5000)
    _check(best is not None, "engine returned no bestmove at start position")
    _check(uci_to_move(best) in legal, f"engine best move {best} not legal per our engine")
    _check(1 <= len(pvs) <= 4, f"expected 1..4 pv lines, got {len(pvs)}")
    _check(all(abs(pv.cp) < 2000 for pv in pvs if not pv.is_mate), "implausible start eval")
    print(f"  [ok] live engine (best {best}, cp {pvs[0].cp}, {len(pvs)} pvs)")


if __name__ == "__main__":
    test_fen()
    test_move_mapping()
    test_info_parsing()
    test_live_engine()
    print("PASS: pikafish annotator tests")
