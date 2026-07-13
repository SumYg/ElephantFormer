"""Board-state dataset for the encoder-only ElephantFormer (Phase 0).

Replays each parsed game through the rules engine and emits one training example
per position: ``(features, target_policy_index, value_target)``.

* ``features`` are the per-square board features from
  :mod:`elephant_former.data_utils.board_features`.
* ``target_policy_index`` is the flat ``from_sq * 90 + to_sq`` index of the move
  actually played from that position.
* ``value_target`` is the final game result from the mover's perspective
  (``0`` loss / ``1`` draw / ``2`` win), or :data:`VALUE_IGNORE_INDEX` when the
  PGN result is unknown/other (excluded from the value loss via ``ignore_index``).

Games containing a move the engine rejects (unparseable or not legal from the
current position) are skipped in full; the skipped count is logged.

Replaying in Python is slow, so a built dataset is cached under ``data/cache/``
keyed by a hash of the source PGN path + mtime (+ a feature-format version). The
cache is loaded when present.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from elephant_former.data.elephant_parser import ElephantGame, parse_iccs_pgn_file
from elephant_former.data_utils import board_features as bf
from elephant_former.data_utils.tokenization_utils import parse_iccs_move_to_coords
from elephant_former.engine.elephant_chess_game import ElephantChessGame, Player

# Bump when the on-disk feature/label format changes so stale caches are ignored.
# v2: engine legality fixed (stale-board attack test; mate-in-one heuristic removed
# from rules), which changes both game filtering and the threat-flag features.
FEATURE_VERSION = 2
VALUE_IGNORE_INDEX = -100

BoardExample = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def result_to_value(result: str, side_to_move: Player) -> int:
    """Final result as a class from ``side_to_move``'s perspective.

    Returns ``0`` (loss), ``1`` (draw), ``2`` (win), or :data:`VALUE_IGNORE_INDEX`
    when the result string is unknown/other.
    """
    r = result.strip()
    if r == "1-0":
        red_value = 2
    elif r == "0-1":
        red_value = 0
    elif r in ("1/2-1/2", "1/2", "draw", "Draw"):
        return 1
    else:
        return VALUE_IGNORE_INDEX
    return red_value if side_to_move == Player.RED else 2 - red_value


@dataclass
class _PositionArrays:
    """Column-stacked positions ready for caching / tensor conversion."""

    piece_ids: np.ndarray      # (N, 90) int8
    flags: np.ndarray          # (N, 90, 4) int8
    side_to_move: np.ndarray   # (N,) int8
    policy_index: np.ndarray   # (N,) int32
    value: np.ndarray          # (N,) int16

    def __len__(self) -> int:
        return int(self.piece_ids.shape[0])


def _empty_arrays() -> _PositionArrays:
    return _PositionArrays(
        piece_ids=np.zeros((0, bf.NUM_SQUARES), dtype=np.int8),
        flags=np.zeros((0, bf.NUM_SQUARES, bf.NUM_FLAGS), dtype=np.int8),
        side_to_move=np.zeros((0,), dtype=np.int8),
        policy_index=np.zeros((0,), dtype=np.int32),
        value=np.zeros((0,), dtype=np.int16),
    )


def positions_from_games(
    games: List[ElephantGame], min_game_len_moves: int = 2
) -> Tuple[_PositionArrays, int]:
    """Replay ``games`` and build stacked position arrays.

    Returns the arrays and the number of games skipped (rejected move or too short).
    """
    piece_rows: List[np.ndarray] = []
    flag_rows: List[np.ndarray] = []
    stm_rows: List[int] = []
    policy_rows: List[int] = []
    value_rows: List[int] = []

    skipped = 0
    for game in games:
        iccs_moves = game.parsed_moves
        if len(iccs_moves) < min_game_len_moves:
            skipped += 1
            continue

        engine_game = ElephantChessGame(fen=game.initial_fen)
        game_piece: List[np.ndarray] = []
        game_flags: List[np.ndarray] = []
        game_stm: List[int] = []
        game_policy: List[int] = []
        game_value: List[int] = []
        rejected = False

        for iccs_move in iccs_moves:
            coords = parse_iccs_move_to_coords(iccs_move)
            if coords is None:
                rejected = True
                break

            stm = engine_game.current_player
            opp = engine_game.get_opponent(stm)
            stm_legal = engine_game.get_all_legal_moves_basic(stm)
            if coords not in stm_legal:
                rejected = True
                break
            opp_legal = engine_game.get_all_legal_moves_basic(opp)

            feats = bf.extract_features(
                engine_game, stm_legal_moves=stm_legal, opp_legal_moves=opp_legal
            )
            game_piece.append(feats.piece_ids.astype(np.int8))
            game_flags.append(feats.flags.astype(np.int8))
            game_stm.append(feats.side_to_move)
            game_policy.append(bf.move_to_policy_index(coords))
            game_value.append(result_to_value(game.result, stm))

            engine_game.apply_move(coords)

        if rejected:
            skipped += 1
            continue

        piece_rows.extend(game_piece)
        flag_rows.extend(game_flags)
        stm_rows.extend(game_stm)
        policy_rows.extend(game_policy)
        value_rows.extend(game_value)

    if not piece_rows:
        return _empty_arrays(), skipped

    arrays = _PositionArrays(
        piece_ids=np.stack(piece_rows).astype(np.int8),
        flags=np.stack(flag_rows).astype(np.int8),
        side_to_move=np.asarray(stm_rows, dtype=np.int8),
        policy_index=np.asarray(policy_rows, dtype=np.int32),
        value=np.asarray(value_rows, dtype=np.int16),
    )
    return arrays, skipped


def _positions_worker(args: Tuple[List[ElephantGame], int]) -> Tuple[_PositionArrays, int]:
    games_chunk, min_game_len_moves = args
    return positions_from_games(games_chunk, min_game_len_moves)


def _concat_arrays(parts: List[_PositionArrays]) -> _PositionArrays:
    parts = [p for p in parts if len(p) > 0]
    if not parts:
        return _empty_arrays()
    return _PositionArrays(
        piece_ids=np.concatenate([p.piece_ids for p in parts]),
        flags=np.concatenate([p.flags for p in parts]),
        side_to_move=np.concatenate([p.side_to_move for p in parts]),
        policy_index=np.concatenate([p.policy_index for p in parts]),
        value=np.concatenate([p.value for p in parts]),
    )


def positions_from_games_parallel(
    games: List[ElephantGame],
    min_game_len_moves: int = 2,
    num_workers: int = 1,
    chunk_size: int = 200,
) -> Tuple[_PositionArrays, int]:
    """Order-preserving parallel version of :func:`positions_from_games`.

    Fans game chunks out to a process pool; results are concatenated in input
    order, so the output is identical to the sequential builder. Progress is
    printed after each completed chunk (flushed, visible under redirection).
    ``num_workers``: 1 = sequential, 0 = auto (physical cores - 2).
    """
    if num_workers == 0:
        num_workers = max(1, (os.cpu_count() or 3) - 2)
    if num_workers <= 1 or len(games) <= chunk_size:
        return positions_from_games(games, min_game_len_moves)

    chunks = [games[i : i + chunk_size] for i in range(0, len(games), chunk_size)]
    parts: List[_PositionArrays] = []
    skipped = 0
    games_done = 0
    positions = 0
    start = time.time()

    from multiprocessing import Pool

    print(f"Building with {num_workers} workers over {len(chunks)} chunks ...", flush=True)
    with Pool(processes=num_workers) as pool:
        for i, (arrays, chunk_skipped) in enumerate(
            pool.imap(_positions_worker, [(c, min_game_len_moves) for c in chunks])
        ):
            parts.append(arrays)
            skipped += chunk_skipped
            games_done += len(chunks[i])
            positions += len(arrays)
            elapsed = time.time() - start
            rate = games_done / elapsed if elapsed > 0 else 0.0
            eta_min = (len(games) - games_done) / rate / 60 if rate > 0 else float("inf")
            print(
                f"  [{games_done}/{len(games)} games] {positions} positions, "
                f"{rate:.1f} games/s, ~{eta_min:.1f} min left",
                flush=True,
            )

    return _concat_arrays(parts), skipped


def _file_sha1(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(pgn_path: Path, cache_dir: Path, min_game_len_moves: int) -> Path:
    # Keyed on file *content* (not path/mtime) so caches copied between machines
    # (rented GPU boxes, VPS workers) are found without a rebuild.
    key = f"{_file_sha1(pgn_path)}|min{min_game_len_moves}|v{FEATURE_VERSION}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{pgn_path.stem}.{digest}.npz"


def _save_cache(path: Path, arrays: _PositionArrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        piece_ids=arrays.piece_ids,
        flags=arrays.flags,
        side_to_move=arrays.side_to_move,
        policy_index=arrays.policy_index,
        value=arrays.value,
    )


def _load_cache(path: Path) -> _PositionArrays:
    data = np.load(path)
    return _PositionArrays(
        piece_ids=data["piece_ids"],
        flags=data["flags"],
        side_to_move=data["side_to_move"],
        policy_index=data["policy_index"],
        value=data["value"],
    )


class BoardChessDataset(Dataset):
    """PyTorch dataset of board-state positions with policy + value targets."""

    def __init__(
        self,
        games: Optional[List[ElephantGame]] = None,
        pgn_file_path: Optional[Union[str, Path]] = None,
        min_game_len_moves: int = 2,
        cache_dir: Union[str, Path] = "data/cache",
        use_cache: bool = True,
        num_workers: int = 1,
    ) -> None:
        """Build (or load) the dataset.

        Provide ``pgn_file_path`` to enable disk caching (keyed by path+mtime), or
        pass pre-parsed ``games`` directly (no caching in that case).
        """
        cache_dir = Path(cache_dir)

        if pgn_file_path is not None:
            pgn_file_path = Path(pgn_file_path)
            cache_file = _cache_path(pgn_file_path, cache_dir, min_game_len_moves)
            if use_cache and cache_file.exists():
                print(f"Loading cached board dataset from: {cache_file}")
                self._arrays = _load_cache(cache_file)
            else:
                games = games if games is not None else parse_iccs_pgn_file(pgn_file_path)
                print(f"Building board dataset from {len(games)} games in {pgn_file_path} ...")
                self._arrays, skipped = positions_from_games_parallel(
                    games, min_game_len_moves, num_workers=num_workers
                )
                print(
                    f"Built {len(self._arrays)} positions "
                    f"({skipped} games skipped: rejected move or too short)."
                )
                if use_cache:
                    _save_cache(cache_file, self._arrays)
                    print(f"Cached board dataset to: {cache_file}")
        elif games is not None:
            print(f"Building board dataset from {len(games)} pre-loaded games ...")
            self._arrays, skipped = positions_from_games_parallel(
                    games, min_game_len_moves, num_workers=num_workers
                )
            print(
                f"Built {len(self._arrays)} positions "
                f"({skipped} games skipped: rejected move or too short)."
            )
        else:
            raise ValueError("Provide either `games` or `pgn_file_path`.")

    def __len__(self) -> int:
        return len(self._arrays)

    def game_ids(self) -> np.ndarray:
        """Per-position game ids ``(N,)``, derived from the previous-move flags."""
        return game_ids_from_flags(self._arrays.flags)

    def __getitem__(self, idx: int) -> BoardExample:
        a = self._arrays
        return (
            torch.from_numpy(a.piece_ids[idx].astype(np.int64)),
            torch.from_numpy(a.flags[idx].astype(np.int64)),
            torch.tensor(int(a.side_to_move[idx]), dtype=torch.long),
            torch.tensor(int(a.policy_index[idx]), dtype=torch.long),
            torch.tensor(int(a.value[idx]), dtype=torch.long),
        )


def game_ids_from_flags(flags: np.ndarray) -> np.ndarray:
    """Recover per-position game ids ``(N,)`` from cached feature flags.

    The first position of a game is the only one with no previous-move flag
    (``extract_features`` sets ``FLAG_PREV_FROM`` from the move history, which
    is empty exactly at a game's first position), so game boundaries are
    recoverable from existing caches without a format change or rebuild.
    """
    if flags.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    starts = flags[:, :, bf.FLAG_PREV_FROM].sum(axis=1) == 0
    if not starts[0]:
        raise ValueError(
            "First cached position has a previous-move flag; positions are not "
            "in game order — cannot derive game boundaries."
        )
    return np.cumsum(starts.astype(np.int64)) - 1


def split_indices_by_game(
    game_ids: np.ndarray,
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split position indices so every game lands wholly in one split.

    Ratio semantics match ``train_board.py``'s position-level split:
    ``test_ratio`` is the target fraction of all positions, ``val_ratio`` the
    target fraction of the remaining (non-test) positions. Whole games are
    assigned in a seeded shuffled order until each target is reached, so the
    achieved fractions deviate by at most one game.

    Returns ``(train_indices, val_indices, test_indices)`` int64 arrays.
    """
    n = int(game_ids.shape[0])
    if n == 0:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty.copy(), empty.copy()

    num_games = int(game_ids[-1]) + 1
    counts = np.bincount(game_ids, minlength=num_games)
    order = np.random.default_rng(seed).permutation(num_games)
    cum = np.cumsum(counts[order])

    def _games_to_reach(target: int, start: int) -> int:
        """Smallest k such that games order[start:start+k] hold >= target positions."""
        if target <= 0:
            return 0
        base = cum[start - 1] if start > 0 else 0
        k = int(np.searchsorted(cum, base + target, side="left")) + 1 - start
        return min(k, num_games - start)

    n_test_target = int(n * test_ratio)
    k_test = _games_to_reach(n_test_target, 0)
    n_test = int(cum[k_test - 1]) if k_test > 0 else 0
    n_val_target = int((n - n_test) * val_ratio)
    k_val = _games_to_reach(n_val_target, k_test)

    # 0 = train, 1 = val, 2 = test per game, broadcast to positions.
    labels = np.zeros(num_games, dtype=np.int8)
    labels[order[:k_test]] = 2
    labels[order[k_test : k_test + k_val]] = 1
    row_labels = labels[game_ids]
    return (
        np.flatnonzero(row_labels == 0),
        np.flatnonzero(row_labels == 1),
        np.flatnonzero(row_labels == 2),
    )


def board_collate_fn(batch: List[BoardExample]) -> Tuple[torch.Tensor, ...]:
    """Stack a list of :data:`BoardExample` tuples into batched tensors."""
    piece_ids, flags, side_to_move, policy_index, value = zip(*batch)
    return (
        torch.stack(piece_ids),
        torch.stack(flags),
        torch.stack(side_to_move),
        torch.stack(policy_index),
        torch.stack(value),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build and cache a board-state dataset from a PGN file.")
    parser.add_argument("--pgn_file_path", type=str, required=True, help="Source PGN file.")
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--no_cache", action="store_true", help="Build without reading/writing the cache.")
    parser.add_argument("--min_game_len_moves", type=int, default=2)
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Parallel build processes; 0 = auto (cores - 2), 1 = sequential.",
    )
    args = parser.parse_args()

    dataset = BoardChessDataset(
        pgn_file_path=args.pgn_file_path,
        cache_dir=args.cache_dir,
        min_game_len_moves=args.min_game_len_moves,
        use_cache=not args.no_cache,
        num_workers=args.workers,
    )
    print(f"Dataset length: {len(dataset)} positions.")
    if len(dataset) > 0:
        piece_ids, flags, stm, policy, value = dataset[0]
        print("piece_ids", tuple(piece_ids.shape), "flags", tuple(flags.shape))
        print("side_to_move", int(stm), "policy_index", int(policy), "value", int(value))
        print("decoded first move:", bf.policy_index_to_move(int(policy)))
