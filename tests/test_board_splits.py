"""Game-level splits: boundary recovery from flags, leak-free split assignment.

Run: ``uv run python tests/test_board_splits.py`` (exits non-zero on failure).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
from types import SimpleNamespace

import numpy as np

from elephant_former.data_utils.board_dataset import (
    BoardChessDataset,
    game_ids_from_flags,
    positions_from_games,
    split_indices_by_game,
)
from elephant_former.engine.elephant_chess_game import ElephantChessGame, INITIAL_BOARD_FEN

NUM_TEST_GAMES = 25
_RESULTS = ["1-0", "0-1", "1/2-1/2"]


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _iccs(move) -> str:
    fx, fy, tx, ty = move
    return f"{chr(ord('A') + fx)}{fy}-{chr(ord('A') + tx)}{ty}"


def _random_games(num_games: int, seed: int = 0):
    """Seeded random self-play games (legal by construction, varying lengths)."""
    rng = random.Random(seed)
    games = []
    for i in range(num_games):
        engine = ElephantChessGame()
        moves = []
        for _ in range(rng.randint(6, 40)):
            legal = engine.get_all_legal_moves_basic(engine.current_player)
            if not legal:
                break
            move = rng.choice(legal)
            moves.append(_iccs(move))
            engine.apply_move(move)
        games.append(
            SimpleNamespace(
                parsed_moves=moves,
                initial_fen=INITIAL_BOARD_FEN,
                result=_RESULTS[i % len(_RESULTS)],
            )
        )
    return games


def _build_arrays():
    games = _random_games(NUM_TEST_GAMES)
    arrays, skipped = positions_from_games(games)
    _check(len(arrays) > 0, "self-play games produced no positions")
    _check(skipped == 0, f"self-play games should all replay, {skipped} skipped")
    return arrays, len(games) - skipped


def test_game_boundaries_from_flags() -> None:
    arrays, games_kept = _build_arrays()
    ids = game_ids_from_flags(arrays.flags)

    _check(ids.shape == (len(arrays),), "one game id per position expected")
    _check(int(ids[0]) == 0, "first position should belong to game 0")
    _check(int(ids[-1]) + 1 == games_kept, f"expected {games_kept} games, got {int(ids[-1]) + 1}")
    steps = np.diff(ids)
    _check(np.all((steps == 0) | (steps == 1)), "game ids must be contiguous and non-decreasing")
    print(f"  [ok] boundaries recovered ({games_kept} games, {len(arrays)} positions)")


def test_split_is_leak_free_and_complete() -> None:
    arrays, _ = _build_arrays()
    ids = game_ids_from_flags(arrays.flags)
    train, val, test = split_indices_by_game(ids, test_ratio=0.2, val_ratio=0.2, seed=42)

    n = len(arrays)
    all_idx = np.concatenate([train, val, test])
    _check(len(all_idx) == n, "splits must cover every position exactly once")
    _check(len(np.unique(all_idx)) == n, "splits must not overlap")

    train_games = set(ids[train].tolist())
    val_games = set(ids[val].tolist())
    test_games = set(ids[test].tolist())
    _check(not (train_games & val_games), "a game leaked between train and val")
    _check(not (train_games & test_games), "a game leaked between train and test")
    _check(not (val_games & test_games), "a game leaked between val and test")

    # Achieved fractions are within one game of the targets.
    max_game = int(np.bincount(ids).max())
    _check(abs(len(test) - 0.2 * n) <= max_game, "test size too far from target")
    _check(abs(len(val) - 0.2 * (n - len(test))) <= max_game, "val size too far from target")
    print(
        f"  [ok] leak-free split: train {len(train)}, val {len(val)}, test {len(test)} "
        f"({len(train_games)}/{len(val_games)}/{len(test_games)} games)"
    )


def test_split_determinism_and_seed_sensitivity() -> None:
    arrays, _ = _build_arrays()
    ids = game_ids_from_flags(arrays.flags)
    a = split_indices_by_game(ids, 0.2, 0.2, seed=7)
    b = split_indices_by_game(ids, 0.2, 0.2, seed=7)
    c = split_indices_by_game(ids, 0.2, 0.2, seed=8)
    _check(all(np.array_equal(x, y) for x, y in zip(a, b)), "same seed must give the same split")
    _check(
        any(not np.array_equal(x, y) for x, y in zip(a, c)),
        "different seeds should give different splits",
    )
    print("  [ok] deterministic per seed")


def test_zero_ratio_edge_cases() -> None:
    arrays, _ = _build_arrays()
    ids = game_ids_from_flags(arrays.flags)
    train, val, test = split_indices_by_game(ids, test_ratio=0.0, val_ratio=0.0, seed=0)
    _check(len(test) == 0 and len(val) == 0, "zero ratios must give empty val/test")
    _check(len(train) == len(arrays), "all positions must land in train with zero ratios")
    print("  [ok] zero-ratio edge cases")


def test_dataset_game_ids_method() -> None:
    dataset = BoardChessDataset(games=_random_games(NUM_TEST_GAMES), use_cache=False)
    ids = dataset.game_ids()
    _check(len(ids) == len(dataset), "dataset.game_ids() must be per-position")
    print("  [ok] dataset.game_ids()")


if __name__ == "__main__":
    tests = [
        test_game_boundaries_from_flags,
        test_split_is_leak_free_and_complete,
        test_split_determinism_and_seed_sensitivity,
        test_zero_ratio_edge_cases,
        test_dataset_game_ids_method,
    ]
    print("Running game-level split tests:")
    for test in tests:
        test()
    print("All split tests passed.")
