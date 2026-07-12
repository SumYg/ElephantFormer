"""Assert board feature extraction is exactly right on known positions.

Run: ``uv run python tests/test_board_features.py`` (exits non-zero on failure).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elephant_former.data_utils import board_features as bf
from elephant_former.engine.elephant_chess_game import ElephantChessGame


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_index_helpers() -> None:
    _check(bf.square_index(0, 0) == 0, "square_index(0,0) should be 0")
    _check(bf.square_index(8, 9) == 89, "square_index(8,9) should be 89")
    _check(bf.square_index(4, 2) == 22, "square_index(4,2) should be 22")
    for move in [(7, 2, 4, 2), (0, 0, 8, 9), (4, 9, 4, 8)]:
        idx = bf.move_to_policy_index(move)
        _check(0 <= idx < bf.NUM_POLICY_MOVES, f"policy index out of range for {move}")
        _check(bf.policy_index_to_move(idx) == move, f"policy index round-trip failed for {move}")
    print("  [ok] index helpers")


def test_start_position_piece_planes() -> None:
    feats = bf.extract_features(ElephantChessGame())
    pid = feats.piece_ids

    _check(feats.side_to_move == 0, "start side-to-move should be RED (0)")
    _check(pid.shape == (90,), "piece_ids must be shape (90,)")
    _check(feats.flags.shape == (90, 4), "flags must be shape (90, 4)")

    # Red chariot=5, king=1, cannon=6; black king=8, black soldier=14; empty=0.
    _check(pid[bf.square_index(0, 0)] == 5, "red chariot class at (0,0) should be 5")
    _check(pid[bf.square_index(4, 0)] == 1, "red king class at (4,0) should be 1")
    _check(pid[bf.square_index(1, 2)] == 6, "red cannon class at (1,2) should be 6")
    _check(pid[bf.square_index(4, 9)] == 8, "black king class at (4,9) should be 8")
    _check(pid[bf.square_index(0, 6)] == 14, "black soldier class at (0,6) should be 14")
    _check(pid[bf.square_index(4, 5)] == 0, "square (4,5) should be empty")
    _check(int((pid > 0).sum()) == 32, "start position should have 32 pieces")
    print("  [ok] start position piece planes")


def test_start_position_reachability() -> None:
    feats = bf.extract_features(ElephantChessGame())
    stm = feats.flags[:, bf.FLAG_STM_DEST]
    opp = feats.flags[:, bf.FLAG_OPP_DEST]

    # Red (side-to-move) reachable destinations.
    _check(stm[bf.square_index(0, 4)] == 1, "red soldier (0,3)->(0,4) destination missing")
    _check(stm[bf.square_index(4, 2)] == 1, "red cannon (1,2)->(4,2) destination missing")
    _check(stm[bf.square_index(4, 1)] == 1, "red advisor ->(4,1) destination missing")
    _check(stm[bf.square_index(4, 5)] == 0, "red should not reach (4,5) on move 1")
    _check(stm[bf.square_index(4, 9)] == 0, "red should not reach (4,9) on move 1")

    # Black (opponent) reachable destinations, as if it were Black's turn.
    _check(opp[bf.square_index(0, 5)] == 1, "black soldier (0,6)->(0,5) destination missing")
    _check(opp[bf.square_index(4, 4)] == 0, "black should not reach (4,4) on move 1")
    print("  [ok] start position reachability")


def test_previous_move_flags() -> None:
    fresh = bf.extract_features(ElephantChessGame())
    _check(int(fresh.flags[:, bf.FLAG_PREV_FROM].sum()) == 0, "no prev move => prev_from all zero")
    _check(int(fresh.flags[:, bf.FLAG_PREV_TO].sum()) == 0, "no prev move => prev_to all zero")

    game = ElephantChessGame()
    game.apply_move((7, 2, 4, 2))  # H2-E2
    feats = bf.extract_features(game)

    _check(feats.side_to_move == 1, "after one move side-to-move should be BLACK (1)")
    prev_from = feats.flags[:, bf.FLAG_PREV_FROM]
    prev_to = feats.flags[:, bf.FLAG_PREV_TO]
    _check(int(prev_from.sum()) == 1, "exactly one prev_from square expected")
    _check(int(prev_to.sum()) == 1, "exactly one prev_to square expected")
    _check(prev_from[bf.square_index(7, 2)] == 1, "prev_from should mark (7,2)")
    _check(prev_to[bf.square_index(4, 2)] == 1, "prev_to should mark (4,2)")

    # Board should reflect the move: cannon now at (4,2), origin empty.
    _check(feats.piece_ids[bf.square_index(4, 2)] == 6, "cannon should now be at (4,2)")
    _check(feats.piece_ids[bf.square_index(7, 2)] == 0, "origin (7,2) should be empty")
    print("  [ok] previous move flags")


def main() -> int:
    tests = [
        test_index_helpers,
        test_start_position_piece_planes,
        test_start_position_reachability,
        test_previous_move_flags,
    ]
    try:
        for test in tests:
            test()
    except AssertionError as exc:
        print(f"FAIL: {exc}")
        return 1
    print("PASS: board feature tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
