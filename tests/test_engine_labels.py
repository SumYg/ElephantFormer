"""Engine-labeled distillation views: target swap, leakage rule, validation.

Run: ``uv run python tests/test_engine_labels.py`` (exits non-zero on failure).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
from types import SimpleNamespace

import numpy as np
import torch

from elephant_former.data_utils.board_dataset import (
    BoardChessDataset,
    game_ids_from_flags,
    split_indices_by_game,
)
from elephant_former.data_utils.engine_labels import EngineLabeledDataset, load_annotations
from elephant_former.engine.elephant_chess_game import ElephantChessGame, INITIAL_BOARD_FEN

NUM_TEST_GAMES = 20


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _iccs(move) -> str:
    fx, fy, tx, ty = move
    return f"{chr(ord('A') + fx)}{fy}-{chr(ord('A') + tx)}{ty}"


def _random_games(num_games: int, seed: int = 0):
    rng = random.Random(seed)
    games = []
    for i in range(num_games):
        engine = ElephantChessGame()
        moves = []
        for _ in range(rng.randint(6, 30)):
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
                result=["1-0", "0-1", "1/2-1/2"][i % 3],
            )
        )
    return games


def _base_and_annotations():
    base = BoardChessDataset(games=_random_games(NUM_TEST_GAMES), use_cache=False)
    n = len(base)
    human_policy = base._arrays.policy_index.astype(np.int64)
    annotations = {
        # Engine "best" deliberately differs from the human move on every row.
        "eng_best_index": ((human_policy + 1) % 8100).astype(np.int32),
        "annotated": np.ones(n, dtype=bool),
    }
    # A few unusable rows: not annotated, or no best move.
    annotations["annotated"][0] = False
    annotations["eng_best_index"][1] = -1
    return base, annotations, human_policy


def test_policy_swapped_value_kept() -> None:
    base, annotations, human_policy = _base_and_annotations()
    ds = EngineLabeledDataset(base, annotations)

    _check(len(ds) == len(base) - 2, "unusable rows must be excluded")
    for idx in [0, len(ds) // 2, len(ds) - 1]:
        row = int(ds._rows[idx])
        b_piece, b_flags, b_stm, b_policy, b_value = base[row]
        e_piece, e_flags, e_stm, e_policy, e_value = ds[idx]
        _check(torch.equal(b_piece, e_piece) and torch.equal(b_flags, e_flags), "features must match base row")
        _check(int(e_policy) == int(annotations["eng_best_index"][row]), "policy must be the engine best move")
        _check(int(e_policy) != int(human_policy[row]), "engine target should differ from human in this fixture")
        _check(int(e_value) == int(b_value), "value target must stay the human outcome")
    print("  [ok] policy swapped to engine best, value/features unchanged")


def test_train_only_restriction() -> None:
    base, annotations, _ = _base_and_annotations()
    ids = game_ids_from_flags(base._arrays.flags)
    train_idx, val_idx, test_idx = split_indices_by_game(ids, 0.2, 0.2, seed=42)

    ds = EngineLabeledDataset(base, annotations, restrict_rows=train_idx)
    train_games = set(ids[train_idx].tolist())
    engine_games = set(ids[ds._rows].tolist())
    _check(len(ds) > 0, "restricted engine dataset should not be empty")
    _check(
        engine_games <= train_games,
        "engine rows must come only from train-split games",
    )
    held_out = set(ids[val_idx].tolist()) | set(ids[test_idx].tolist())
    _check(not (engine_games & held_out), "engine rows leaked into val/test games")
    print(f"  [ok] train-only restriction ({len(ds)} rows from {len(engine_games)} train games)")


def test_load_annotations_validation() -> None:
    base, annotations, _ = _base_and_annotations()
    scratch = Path(__file__).resolve().parents[1] / "data" / "cache"
    tmp = scratch / "_test_engine_labels.npz"
    np.savez_compressed(tmp, **annotations)
    try:
        loaded = load_annotations(tmp, expected_rows=len(base))
        _check(len(loaded["annotated"]) == len(base), "round-trip row count")
        try:
            load_annotations(tmp, expected_rows=len(base) + 1)
            raise AssertionError("row-count mismatch must raise")
        except ValueError:
            pass
        bad = scratch / "_test_engine_labels_bad.npz"
        np.savez_compressed(bad, foo=np.zeros(3))
        try:
            load_annotations(bad)
            raise AssertionError("missing keys must raise")
        except ValueError:
            pass
        finally:
            bad.unlink(missing_ok=True)
    finally:
        tmp.unlink(missing_ok=True)
    print("  [ok] load_annotations validates rows and keys")


def test_row_count_mismatch_raises() -> None:
    base, annotations, _ = _base_and_annotations()
    short = {k: v[:-5] for k, v in annotations.items()}
    try:
        EngineLabeledDataset(base, short)
        raise AssertionError("mismatched annotation length must raise")
    except ValueError:
        pass
    print("  [ok] dataset rejects misaligned annotations")


if __name__ == "__main__":
    tests = [
        test_policy_swapped_value_kept,
        test_train_only_restriction,
        test_load_annotations_validation,
        test_row_count_mismatch_raises,
    ]
    print("Running engine-label tests:")
    for test in tests:
        test()
    print("All engine-label tests passed.")
