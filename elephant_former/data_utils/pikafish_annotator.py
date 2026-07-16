"""Pikafish annotation pipeline (Phase 1).

Labels cached board positions with engine targets for distillation training:
the engine's best move (as a flat policy index), its centipawn evaluation, and
the top-k alternatives (MultiPV). Scores are from the side-to-move's
perspective, as per UCI convention.

The annotator reads the ``.npz`` position caches produced by
:mod:`elephant_former.data_utils.board_dataset` (``piece_ids`` + ``side_to_move``
fully determine the position) and writes a parallel ``.npz`` with engine labels.
Annotation is resumable: output is checkpointed periodically and an existing
output file is continued, not overwritten.

Coordinate conventions (see :mod:`board_features`): square ``(x, y)`` has file
``x`` 0..8 and rank ``y`` 0..9; UCI moves are ``<file letter a-i><rank digit>``
pairs, e.g. ``h2e2`` == (7, 2, 4, 2); FEN ranks run y=9 (Black's home) down to
y=0, side to move is ``w`` for Red.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from elephant_former.data_utils import board_features as bf
from elephant_former.engine.elephant_chess_game import ElephantChessGame, Move, Player

# piece class (1..14, see board_features) -> FEN letter; uppercase = Red.
_CLASS_TO_FEN = {
    1: "K", 2: "A", 3: "B", 4: "N", 5: "R", 6: "C", 7: "P",
    8: "k", 9: "a", 10: "b", 11: "n", 12: "r", 13: "c", 14: "p",
}

MATE_SCORE = 30000
NO_MOVE_INDEX = -1


def piece_ids_to_fen(piece_ids: Sequence[int], side_to_move: int) -> str:
    """Standard xiangqi FEN from a cached 90-long piece-class vector.

    ``side_to_move``: 0 = Red ('w'), 1 = Black ('b').
    """
    rows: List[str] = []
    for y in range(9, -1, -1):
        row = ""
        empties = 0
        for x in range(9):
            cls = int(piece_ids[bf.square_index(x, y)])
            if cls == 0:
                empties += 1
            else:
                if empties:
                    row += str(empties)
                    empties = 0
                row += _CLASS_TO_FEN[cls]
        if empties:
            row += str(empties)
        rows.append(row)
    stm = "w" if side_to_move == 0 else "b"
    return f"{'/'.join(rows)} {stm} - - 0 1"


def game_to_fen(game: ElephantChessGame) -> str:
    """FEN of a live engine position (delegates to the cached-feature encoding)."""
    feats = bf.extract_features(game)
    return piece_ids_to_fen(feats.piece_ids, feats.side_to_move)


def move_to_uci(move: Move) -> str:
    fx, fy, tx, ty = move
    return f"{chr(ord('a') + fx)}{fy}{chr(ord('a') + tx)}{ty}"


def uci_to_move(uci: str) -> Move:
    return (ord(uci[0]) - ord("a"), int(uci[1]), ord(uci[2]) - ord("a"), int(uci[3]))


@dataclass
class PvLine:
    """One MultiPV line: best move of the line + score from side-to-move's view."""

    move: str          # UCI move string, e.g. "h2e2"
    cp: int            # centipawns; mate scores are folded to +/-(MATE_SCORE - plies)
    is_mate: bool
    depth: int


def parse_info_lines(lines: Sequence[str]) -> List[PvLine]:
    """Extract the deepest PV line per multipv index from UCI ``info`` output."""
    best: dict[int, PvLine] = {}
    for line in lines:
        parts = line.split()
        if not parts or parts[0] != "info" or "pv" not in parts or "score" not in parts:
            continue
        try:
            depth = int(parts[parts.index("depth") + 1]) if "depth" in parts else 0
            multipv = int(parts[parts.index("multipv") + 1]) if "multipv" in parts else 1
            s_idx = parts.index("score")
            kind, value = parts[s_idx + 1], int(parts[s_idx + 2])
            move = parts[parts.index("pv") + 1]
        except (ValueError, IndexError):
            continue
        if kind == "mate":
            cp = MATE_SCORE - abs(value) if value > 0 else -(MATE_SCORE - abs(value))
            is_mate = True
        else:
            cp = value
            is_mate = False
        prev = best.get(multipv)
        if prev is None or depth >= prev.depth:
            best[multipv] = PvLine(move=move, cp=cp, is_mate=is_mate, depth=depth)
    return [best[i] for i in sorted(best)]


class PikafishEngine:
    """Thin synchronous UCI wrapper around a Pikafish process."""

    def __init__(
        self,
        engine_path: str | Path,
        nnue_path: Optional[str | Path] = None,
        threads: int = 1,
        hash_mb: int = 64,
        multipv: int = 1,
    ) -> None:
        # Resolve before Popen: with cwd= set, POSIX resolves a relative
        # executable path against the child's cwd, not the caller's.
        engine_path = Path(engine_path).resolve()
        self._proc = subprocess.Popen(
            [str(engine_path)],
            cwd=str(engine_path.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.multipv = multipv
        self._send("uci")
        self._read_until("uciok")
        if nnue_path is not None:
            self._send(f"setoption name EvalFile value {Path(nnue_path).resolve()}")
        self._send(f"setoption name Threads value {threads}")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send(f"setoption name MultiPV value {multipv}")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, cmd: str) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _read_until(self, token: str) -> List[str]:
        assert self._proc.stdout is not None
        lines: List[str] = []
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                raise RuntimeError(f"engine terminated while waiting for '{token}'")
            line = line.strip()
            lines.append(line)
            if line.startswith(token):
                return lines

    def analyze_fen(
        self,
        fen: str,
        nodes: int = 10000,
        moves: Optional[Sequence[str]] = None,
    ) -> Tuple[Optional[str], List[PvLine]]:
        """Search a position; returns (bestmove or None if mated/stalled, PV lines).

        ``moves`` (UCI strings) are appended as ``position fen ... moves ...`` so
        the engine sees the game history — required for it to reason about
        repetitions instead of treating every position as fresh.
        """
        cmd = f"position fen {fen}"
        if moves:
            cmd += " moves " + " ".join(moves)
        self._send(cmd)
        self._send(f"go nodes {nodes}")
        lines = self._read_until("bestmove")
        pvs = parse_info_lines(lines)
        best_token = lines[-1].split()
        best = best_token[1] if len(best_token) > 1 else "(none)"
        if best == "(none)":
            return None, pvs
        return best, pvs

    def close(self) -> None:
        try:
            self._send("quit")
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()

    def __enter__(self) -> "PikafishEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def annotate_cache(
    cache_path: str | Path,
    engine_path: str | Path,
    out_path: str | Path,
    nnue_path: Optional[str | Path] = None,
    nodes: int = 10000,
    multipv: int = 4,
    threads: int = 1,
    limit: Optional[int] = None,
    start: int = 0,
    end: Optional[int] = None,
    checkpoint_every: int = 500,
) -> None:
    """Annotate positions from a board-dataset cache with engine labels.

    Resumable: if ``out_path`` exists, annotation continues after the last
    annotated position. ``start``/``end`` restrict work to a row range so the
    job can be sharded across processes/machines (write each shard to its own
    ``out_path``, then combine with :func:`merge_annotations`). Output arrays (aligned to cache row order):

    - ``eng_best_index``  (N,) int32 — flat policy index of engine best move (-1 if none)
    - ``eng_best_cp``     (N,) int32 — score of best line, side-to-move perspective
    - ``eng_topk_index``  (N, multipv) int32 — MultiPV move indices (-1 padded)
    - ``eng_topk_cp``     (N, multipv) int32 — MultiPV scores (0 padded)
    - ``annotated``       (N,) bool
    """
    cache_path, out_path = Path(cache_path), Path(out_path)
    data = np.load(cache_path)
    piece_ids, side_to_move = data["piece_ids"], data["side_to_move"]
    n = len(piece_ids)

    if out_path.exists():
        prev = np.load(out_path)
        eng_best_index = prev["eng_best_index"].copy()
        eng_best_cp = prev["eng_best_cp"].copy()
        eng_topk_index = prev["eng_topk_index"].copy()
        eng_topk_cp = prev["eng_topk_cp"].copy()
        annotated = prev["annotated"].copy()
        if eng_topk_index.shape[1] != multipv:
            raise ValueError(
                f"existing output has multipv={eng_topk_index.shape[1]}, requested {multipv}"
            )
        print(f"Resuming: {int(annotated.sum())}/{n} already annotated.", flush=True)
    else:
        eng_best_index = np.full(n, NO_MOVE_INDEX, dtype=np.int32)
        eng_best_cp = np.zeros(n, dtype=np.int32)
        eng_topk_index = np.full((n, multipv), NO_MOVE_INDEX, dtype=np.int32)
        eng_topk_cp = np.zeros((n, multipv), dtype=np.int32)
        annotated = np.zeros(n, dtype=bool)

    def save() -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            eng_best_index=eng_best_index,
            eng_best_cp=eng_best_cp,
            eng_topk_index=eng_topk_index,
            eng_topk_cp=eng_topk_cp,
            annotated=annotated,
            nodes=np.int64(nodes),
        )

    todo = np.flatnonzero(~annotated)
    if start or end is not None:
        hi = n if end is None else min(end, n)
        todo = todo[(todo >= start) & (todo < hi)]
    if limit is not None:
        todo = todo[:limit]
    if len(todo) == 0:
        print("Nothing to annotate.", flush=True)
        return

    start = time.time()
    done_now = 0
    with PikafishEngine(
        engine_path, nnue_path=nnue_path, threads=threads, multipv=multipv
    ) as engine:
        for i in todo:
            fen = piece_ids_to_fen(piece_ids[i], int(side_to_move[i]))
            best, pvs = engine.analyze_fen(fen, nodes=nodes)
            if best is not None:
                eng_best_index[i] = bf.move_to_policy_index(uci_to_move(best))
                by_move = {pv.move: pv for pv in pvs}
                if best in by_move:
                    eng_best_cp[i] = by_move[best].cp
                elif pvs:
                    eng_best_cp[i] = pvs[0].cp
                for k, pv in enumerate(pvs[:multipv]):
                    eng_topk_index[i, k] = bf.move_to_policy_index(uci_to_move(pv.move))
                    eng_topk_cp[i, k] = pv.cp
            annotated[i] = True
            done_now += 1
            if done_now % checkpoint_every == 0:
                save()
                rate = done_now / (time.time() - start)
                remaining = (len(todo) - done_now) / rate if rate > 0 else float("inf")
                print(
                    f"  [{done_now}/{len(todo)}] {rate:.1f} pos/s, "
                    f"~{remaining / 3600:.1f} h left (this run)",
                    flush=True,
                )

    save()
    rate = done_now / (time.time() - start)
    print(
        f"Annotated {done_now} positions at {rate:.1f} pos/s. "
        f"Total {int(annotated.sum())}/{n} in {out_path}.",
        flush=True,
    )


def merge_annotations(paths: Sequence[str | Path], out_path: str | Path) -> None:
    """Merge shard annotation files (same cache, disjoint row ranges) into one."""
    base = {k: v.copy() for k, v in np.load(paths[0]).items()}
    for p in paths[1:]:
        shard = np.load(p)
        if shard["annotated"].shape != base["annotated"].shape:
            raise ValueError(f"{p}: shard length differs from {paths[0]}")
        if shard["eng_topk_index"].shape[1] != base["eng_topk_index"].shape[1]:
            raise ValueError(f"{p}: shard multipv differs from {paths[0]}")
        m = shard["annotated"]
        overlap = m & base["annotated"]
        if overlap.any():
            print(f"note: {int(overlap.sum())} rows annotated in multiple shards; later shard wins.")
        for key in ("eng_best_index", "eng_best_cp", "eng_topk_index", "eng_topk_cp"):
            base[key][m] = shard[key][m]
        base["annotated"] |= m
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **base)
    print(f"Merged {len(paths)} shards -> {out_path}: {int(base['annotated'].sum())} annotated rows.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Annotate cached positions with Pikafish labels.")
    parser.add_argument("--cache_file", type=str, default=None, help="Board-dataset .npz cache.")
    parser.add_argument("--engine", type=str, default="tools/Windows/pikafish-bmi2.exe")
    parser.add_argument("--nnue", type=str, default="tools/pikafish.nnue")
    parser.add_argument("--out", type=str, required=True, help="Output .npz for engine labels.")
    parser.add_argument("--nodes", type=int, default=10000)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Annotate at most this many new positions.")
    parser.add_argument("--start", type=int, default=0, help="First cache row of this shard (inclusive).")
    parser.add_argument("--end", type=int, default=None, help="Last cache row of this shard (exclusive).")
    parser.add_argument("--checkpoint_every", type=int, default=500)
    parser.add_argument(
        "--merge", type=str, nargs="+", default=None,
        help="Merge these shard .npz files into --out instead of annotating.",
    )
    args = parser.parse_args()

    if args.merge:
        merge_annotations(args.merge, args.out)
        raise SystemExit(0)
    if not args.cache_file:
        parser.error("--cache_file is required unless --merge is used")

    annotate_cache(
        cache_path=args.cache_file,
        engine_path=args.engine,
        out_path=args.out,
        nnue_path=args.nnue,
        nodes=args.nodes,
        multipv=args.multipv,
        threads=args.threads,
        limit=args.limit,
        start=args.start,
        end=args.end,
        checkpoint_every=args.checkpoint_every,
    )
