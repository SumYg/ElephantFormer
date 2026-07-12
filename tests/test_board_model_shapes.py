"""Assert board model output shapes and legal-move masking behaviour.

Run: ``uv run python tests/test_board_model_shapes.py`` (exits non-zero on failure).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from elephant_former.data_utils import board_features as bf
from elephant_former.engine.elephant_chess_game import ElephantChessGame
from elephant_former.models.board_transformer import (
    BoardTransformer,
    build_legal_mask,
    mask_policy_logits,
    select_move_index,
)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _small_model() -> BoardTransformer:
    torch.manual_seed(0)
    return BoardTransformer(
        d_model=32, nhead=4, num_encoder_layers=2, dim_feedforward=64, policy_head_dim=16
    )


def test_forward_shapes() -> None:
    model = _small_model()
    batch = 3
    piece_ids = torch.randint(0, bf.NUM_PIECE_CLASSES, (batch, bf.NUM_SQUARES))
    flags = torch.randint(0, 2, (batch, bf.NUM_SQUARES, bf.NUM_FLAGS))
    stm = torch.randint(0, 2, (batch,))

    policy_logits, value_logits = model(piece_ids, flags, stm)
    _check(policy_logits.shape == (batch, bf.NUM_POLICY_MOVES), "policy logits shape wrong")
    _check(value_logits.shape == (batch, 3), "value logits shape wrong")
    _check(torch.isfinite(policy_logits).all(), "raw policy logits should be finite")
    print("  [ok] forward shapes")


def test_legal_mask_values() -> None:
    legal = [0, 100, 8099]
    mask = build_legal_mask(legal)
    _check(mask.shape == (bf.NUM_POLICY_MOVES,), "mask shape wrong")
    _check(int(torch.isfinite(mask).sum()) == len(legal), "mask should keep exactly the legal entries finite")
    for idx in legal:
        _check(mask[idx] == 0.0, f"legal index {idx} should have mask 0")
    _check(mask[1] == float("-inf"), "illegal index should be -inf")
    print("  [ok] legal mask values")


def test_masked_argmax_is_legal() -> None:
    model = _small_model()
    model.eval()

    game = ElephantChessGame()
    game.apply_move((7, 2, 4, 2))  # a non-start position (Black to move)

    legal_moves = game.get_all_legal_moves_basic(game.current_player)
    legal_indices = [bf.move_to_policy_index(m) for m in legal_moves]
    legal_set = set(legal_moves)

    feats = bf.extract_features(game)
    piece_ids = torch.from_numpy(feats.piece_ids).long().unsqueeze(0)
    flags = torch.from_numpy(feats.flags).long().unsqueeze(0)
    stm = torch.tensor([feats.side_to_move], dtype=torch.long)

    with torch.no_grad():
        policy_logits, _ = model(piece_ids, flags, stm)

    masked = mask_policy_logits(policy_logits[0], legal_indices)
    _check(int(torch.isfinite(masked).sum()) == len(legal_indices), "masked finite count must equal #legal moves")

    # Argmax over masked logits must decode to a legal move, for several seeds.
    for seed in range(5):
        torch.manual_seed(seed)
        logits = torch.randn(bf.NUM_POLICY_MOVES)
        idx = select_move_index(logits, legal_indices, temperature=0.0)
        _check(bf.policy_index_to_move(idx) in legal_set, f"argmax move not legal (seed {seed})")

    # Temperature sampling must also stay within the legal set.
    for seed in range(5):
        gen = torch.Generator().manual_seed(seed)
        logits = torch.randn(bf.NUM_POLICY_MOVES)
        idx = select_move_index(logits, legal_indices, temperature=1.0, generator=gen)
        _check(bf.policy_index_to_move(idx) in legal_set, f"sampled move not legal (seed {seed})")
    print("  [ok] masked argmax / sampling is always legal")


def main() -> int:
    tests = [test_forward_shapes, test_legal_mask_values, test_masked_argmax_is_legal]
    try:
        for test in tests:
            test()
    except AssertionError as exc:
        print(f"FAIL: {exc}")
        return 1
    print("PASS: board model shape/masking tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
