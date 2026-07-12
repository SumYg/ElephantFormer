"""Board-state feature extraction for the encoder-only ElephantFormer (Phase 0).

Turns an :class:`ElephantChessGame` position into per-square features for all 90
squares plus a global side-to-move scalar. This is the "board state in" input
representation that replaces the legacy move-history token stream.

Square indexing
---------------
A square ``(x, y)`` with file ``x`` in ``0..8`` and rank ``y`` in ``0..9``
(engine coordinates, ``board[y, x]``) maps to a flat index::

    sq = y * 9 + x

so ``sq`` runs ``0..89`` in row-major (rank-major) order. Files/ranks are
recovered with ``file_of(sq) = sq % 9`` and ``rank_of(sq) = sq // 9``.

Move indexing (policy target)
-----------------------------
A move ``(fx, fy, tx, ty)`` maps to a flat policy index::

    from_sq = fy * 9 + fx
    to_sq   = ty * 9 + tx
    policy_index = from_sq * 90 + to_sq   # 0..8099

Board orientation is kept absolute (no mirroring/rotation); side-to-move is
encoded globally. Mirroring the board for the side-to-move is a possible future
optimisation but is intentionally *not* done here so that piece coordinates line
up 1:1 with the engine and with legal-move indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from elephant_former.engine.elephant_chess_game import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    ElephantChessGame,
    Move,
    Player,
)

NUM_SQUARES = BOARD_WIDTH * BOARD_HEIGHT  # 90
# 0 = empty, 1..7 = red king..soldier, 8..14 = black king..soldier
NUM_PIECE_CLASSES = 15
# Per-square binary flags, in fixed channel order.
NUM_FLAGS = 4
FLAG_STM_DEST = 0      # square is a destination of some legal move of the side-to-move
FLAG_OPP_DEST = 1      # square is a destination of some legal move of the opponent
FLAG_PREV_FROM = 2     # from-square of the previous move
FLAG_PREV_TO = 3       # to-square of the previous move

NUM_POLICY_MOVES = NUM_SQUARES * NUM_SQUARES  # 8100


def square_index(x: int, y: int) -> int:
    """Flat square index ``sq = y * 9 + x`` for file ``x`` and rank ``y``."""
    return y * BOARD_WIDTH + x


def file_of(sq: int) -> int:
    """File (x) of a flat square index."""
    return sq % BOARD_WIDTH


def rank_of(sq: int) -> int:
    """Rank (y) of a flat square index."""
    return sq // BOARD_WIDTH


def move_to_policy_index(move: Move) -> int:
    """Flat policy index ``from_sq * 90 + to_sq`` for a ``(fx, fy, tx, ty)`` move."""
    fx, fy, tx, ty = move
    return square_index(fx, fy) * NUM_SQUARES + square_index(tx, ty)


def policy_index_to_move(index: int) -> Move:
    """Inverse of :func:`move_to_policy_index`."""
    from_sq, to_sq = divmod(index, NUM_SQUARES)
    return (file_of(from_sq), rank_of(from_sq), file_of(to_sq), rank_of(to_sq))


def piece_to_class(piece: int) -> int:
    """Map a signed engine piece value to a piece class id in ``0..14``.

    Empty -> 0, red pieces -> ``abs(piece)`` (1..7), black pieces ->
    ``7 + abs(piece)`` (8..14).
    """
    if piece == 0:
        return 0
    if piece > 0:
        return piece
    return 7 + (-piece)


@dataclass
class BoardFeatures:
    """Per-square features for a single position.

    Attributes:
        piece_ids: ``(90,)`` int64 array of piece class ids (see :func:`piece_to_class`).
        flags: ``(90, 4)`` int64 array of binary flags in the ``FLAG_*`` channel order.
        side_to_move: ``0`` for RED, ``1`` for BLACK.
    """

    piece_ids: np.ndarray
    flags: np.ndarray
    side_to_move: int


def _destination_mask(moves: List[Move]) -> np.ndarray:
    """Binary ``(90,)`` mask marking every to-square appearing in ``moves``."""
    mask = np.zeros(NUM_SQUARES, dtype=np.int64)
    for _, _, tx, ty in moves:
        mask[square_index(tx, ty)] = 1
    return mask


def extract_features(
    game: ElephantChessGame,
    *,
    stm_legal_moves: Optional[List[Move]] = None,
    opp_legal_moves: Optional[List[Move]] = None,
) -> BoardFeatures:
    """Extract board-state features for ``game`` at its current position.

    Args:
        game: The position to featurise. Its ``current_player`` is the side-to-move.
        stm_legal_moves: Optional precomputed legal moves for the side-to-move.
            When ``None`` they are computed with ``get_all_legal_moves_basic``.
        opp_legal_moves: Optional precomputed legal moves for the opponent.

    Returns:
        A :class:`BoardFeatures` instance.

    Notes:
        Reachability is taken from ``ElephantChessGame.get_all_legal_moves_basic``
        (moves that do not leave the moving side's own king in check). For the
        side-to-move this is exactly the set of playable destinations. For the
        opponent it is the opponent's fully-legal moves *as if it were their
        turn*; this is the pragmatic "both sides" channel from the roadmap.

        Caveats:
        * Basic legality drops moves that would leave the moving side's king in
          check, so a pinned opponent piece's squares are not marked even though
          the piece still "attacks" them. This is a threat/mobility map, not a
          raw attack map.
        * In the rare/hypothetical case where the opponent's own king is in check
          at this position, the opponent's reachable set is restricted to
          check-escaping moves. In legally replayed games only the side-to-move
          can be in check, so this does not arise there.
    """
    stm = game.current_player
    opp = game.get_opponent(stm)

    if stm_legal_moves is None:
        stm_legal_moves = game.get_all_legal_moves_basic(stm)
    if opp_legal_moves is None:
        opp_legal_moves = game.get_all_legal_moves_basic(opp)

    board = game.board
    piece_ids = np.fromiter(
        (piece_to_class(int(board[sq // BOARD_WIDTH, sq % BOARD_WIDTH])) for sq in range(NUM_SQUARES)),
        dtype=np.int64,
        count=NUM_SQUARES,
    )

    flags = np.zeros((NUM_SQUARES, NUM_FLAGS), dtype=np.int64)
    flags[:, FLAG_STM_DEST] = _destination_mask(stm_legal_moves)
    flags[:, FLAG_OPP_DEST] = _destination_mask(opp_legal_moves)

    if game.move_history:
        fx, fy, tx, ty = game.move_history[-1]
        flags[square_index(fx, fy), FLAG_PREV_FROM] = 1
        flags[square_index(tx, ty), FLAG_PREV_TO] = 1

    return BoardFeatures(
        piece_ids=piece_ids,
        flags=flags,
        side_to_move=int(stm.value),
    )


def legal_move_indices(game: ElephantChessGame, player: Optional[Player] = None) -> List[int]:
    """Flat policy indices of every legal move for ``player`` (default: side-to-move)."""
    if player is None:
        player = game.current_player
    return [move_to_policy_index(m) for m in game.get_all_legal_moves_basic(player)]


if __name__ == "__main__":
    g = ElephantChessGame()
    feats = extract_features(g)
    print("side_to_move:", feats.side_to_move)
    print("piece_ids shape:", feats.piece_ids.shape, "flags shape:", feats.flags.shape)
    print("num stm destinations:", int(feats.flags[:, FLAG_STM_DEST].sum()))
    print("num opp destinations:", int(feats.flags[:, FLAG_OPP_DEST].sum()))
    print("red king class at (4,0):", feats.piece_ids[square_index(4, 0)])
    print("black king class at (4,9):", feats.piece_ids[square_index(4, 9)])
    print("policy index of (7,2,4,2):", move_to_policy_index((7, 2, 4, 2)))
    print("round-trip:", policy_index_to_move(move_to_policy_index((7, 2, 4, 2))))
