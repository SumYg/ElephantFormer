"""Engine-labeled training views for Phase 1 distillation.

Wraps a built :class:`BoardChessDataset` together with a Pikafish annotation
file (see :mod:`pikafish_annotator`; arrays are row-aligned with the dataset's
cache) and yields the same board features with the **policy target swapped to
the engine's best move**. The value target stays the human game outcome so the
distillation effect on the policy is isolated (cp-derived value targets are a
possible later ablation).

Leakage rule: engine-labeled rows are meant for the *train* split only, and
only for rows whose game the game-level split assigned to train — an engine
label for a val/test-game position would otherwise leak that position into
training. ``restrict_rows`` implements the filter; ``train_board.py`` computes
the mask from its game-level split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from elephant_former.data_utils.board_dataset import BoardChessDataset, BoardExample

REQUIRED_KEYS = ("eng_best_index", "annotated")


def load_annotations(path: Union[str, Path], expected_rows: Optional[int] = None) -> dict:
    """Load an annotation ``.npz`` and validate shape/coverage.

    Returns a dict of arrays. Raises if required keys are missing or the row
    count does not match ``expected_rows`` (row alignment with the source cache
    is the whole contract).
    """
    path = Path(path)
    data = np.load(path)
    missing = [k for k in REQUIRED_KEYS if k not in data.files]
    if missing:
        raise ValueError(f"{path} is not an annotation file (missing keys: {missing})")
    out = {k: data[k] for k in data.files}
    n = len(out["annotated"])
    if expected_rows is not None and n != expected_rows:
        raise ValueError(
            f"{path} has {n} rows but the dataset has {expected_rows}; "
            "annotations must be built from the same cache."
        )
    return out


class EngineLabeledDataset(Dataset):
    """View over a :class:`BoardChessDataset` with engine policy targets.

    Rows are limited to annotated positions with a valid best move, optionally
    intersected with ``restrict_rows`` (e.g. rows of train-split games).
    """

    def __init__(
        self,
        base: BoardChessDataset,
        annotations: dict,
        restrict_rows: Optional[np.ndarray] = None,
    ) -> None:
        self._base = base
        self._best = annotations["eng_best_index"]
        if len(self._best) != len(base):
            raise ValueError(
                f"annotation rows ({len(self._best)}) != dataset rows ({len(base)})"
            )
        usable = annotations["annotated"] & (self._best >= 0)
        if restrict_rows is not None:
            mask = np.zeros(len(base), dtype=bool)
            mask[restrict_rows] = True
            usable &= mask
        self._rows = np.flatnonzero(usable)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> BoardExample:
        row = int(self._rows[idx])
        piece_ids, flags, side_to_move, _, value = self._base[row]
        policy = torch.tensor(int(self._best[row]), dtype=torch.long)
        return piece_ids, flags, side_to_move, policy, value


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect an engine-label file against a PGN's cache.")
    parser.add_argument("--pgn_file_path", type=str, required=True)
    parser.add_argument("--annotations", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    args = parser.parse_args()

    base = BoardChessDataset(pgn_file_path=args.pgn_file_path, cache_dir=args.cache_dir)
    ann = load_annotations(args.annotations, expected_rows=len(base))
    ds = EngineLabeledDataset(base, ann)
    print(f"{len(ds)}/{len(base)} rows usable as engine-labeled examples.")
    human_policy = base[0][3]
    piece_ids, flags, stm, policy, value = ds[0]
    print(f"row 0: human policy {int(human_policy)}, engine policy {int(policy)}, value {int(value)}")
