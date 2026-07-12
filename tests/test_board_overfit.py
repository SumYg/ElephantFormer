"""Overfit a tiny position set to prove the learning plumbing end-to-end.

Targets >90% policy top-1 on ~200 unique positions in well under ~2 minutes on
CPU. ``data/sample_games.pgn`` only yields ~10 positions, so it is topped up with
seeded random-rollout positions; entries are de-duplicated by (pieces, side) so
each input maps to a single target (a well-defined function to memorise).

Run: ``uv run python tests/test_board_overfit.py`` (exits non-zero on failure).
"""

import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn

from elephant_former.data.elephant_parser import parse_iccs_pgn_file
from elephant_former.data_utils import board_features as bf
from elephant_former.data_utils.board_dataset import positions_from_games
from elephant_former.engine.elephant_chess_game import ElephantChessGame
from elephant_former.models.board_transformer import BoardTransformer

TARGET_POSITIONS = 200
TIME_BUDGET_SECONDS = 110.0
ACC_THRESHOLD = 0.90

Position = Tuple[np.ndarray, np.ndarray, int, int]  # piece_ids, flags, side_to_move, policy_index


def _collect_positions(target: int, seed: int) -> List[Position]:
    seen: Dict[Tuple[bytes, int], Position] = {}

    # Real positions from the sample PGN first.
    games = parse_iccs_pgn_file("data/sample_games.pgn")
    arrays, _ = positions_from_games(games)
    for i in range(len(arrays)):
        piece_ids = arrays.piece_ids[i].astype(np.int64)
        key = (piece_ids.tobytes(), int(arrays.side_to_move[i]))
        seen.setdefault(
            key,
            (piece_ids, arrays.flags[i].astype(np.int64), int(arrays.side_to_move[i]), int(arrays.policy_index[i])),
        )

    # Top up with seeded random-rollout positions.
    rng = random.Random(seed)
    guard = 0
    while len(seen) < target and guard < 100:
        guard += 1
        game = ElephantChessGame()
        for _ in range(60):
            stm = game.current_player
            legal = game.get_all_legal_moves_basic(stm)
            if not legal:
                break
            move = rng.choice(legal)
            feats = bf.extract_features(game, stm_legal_moves=legal)
            key = (feats.piece_ids.tobytes(), feats.side_to_move)
            seen.setdefault(
                key, (feats.piece_ids, feats.flags, feats.side_to_move, bf.move_to_policy_index(move))
            )
            game.apply_move(move)
            if len(seen) >= target or game.check_game_over()[0] is not None:
                break

    return list(seen.values())[:target]


def main() -> int:
    torch.manual_seed(0)
    positions = _collect_positions(TARGET_POSITIONS, seed=0)
    n = len(positions)
    print(f"Collected {n} unique positions for the overfit test.")
    if n < 50:
        print("FAIL: could not collect enough positions to make the test meaningful.")
        return 1

    piece_ids = torch.from_numpy(np.stack([p[0] for p in positions])).long()
    flags = torch.from_numpy(np.stack([p[1] for p in positions])).long()
    side = torch.tensor([p[2] for p in positions], dtype=torch.long)
    target = torch.tensor([p[3] for p in positions], dtype=torch.long)

    model = BoardTransformer(
        d_model=128, nhead=4, num_encoder_layers=2, dim_feedforward=256, dropout=0.0, policy_head_dim=64
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    start = time.time()
    best_acc = 0.0
    step = 0
    while time.time() - start < TIME_BUDGET_SECONDS:
        step += 1
        optimizer.zero_grad()
        policy_logits, _ = model(piece_ids, flags, side)
        loss = loss_fn(policy_logits, target)
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            with torch.no_grad():
                acc = (policy_logits.argmax(dim=-1) == target).float().mean().item()
            best_acc = max(best_acc, acc)
            print(f"  step {step:4d}  loss {loss.item():7.4f}  train_acc {acc:.3f}  ({time.time() - start:.1f}s)")
            if acc >= 0.995:
                break

    model.eval()
    with torch.no_grad():
        policy_logits, _ = model(piece_ids, flags, side)
        final_acc = (policy_logits.argmax(dim=-1) == target).float().mean().item()

    elapsed = time.time() - start
    print(f"Final eval policy top-1 accuracy: {final_acc:.3f} over {n} positions in {elapsed:.1f}s ({step} steps).")

    if final_acc >= ACC_THRESHOLD:
        print(f"PASS: overfit reached {final_acc:.1%} >= {ACC_THRESHOLD:.0%}")
        return 0
    print(f"FAIL: overfit only reached {final_acc:.1%} < {ACC_THRESHOLD:.0%}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
