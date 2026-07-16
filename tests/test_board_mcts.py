"""MCTS: terminal exactness, backup signs, seeing past one ply, determinism.

Run: ``uv run python tests/test_board_mcts.py`` (exits non-zero on failure).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from elephant_former.data_utils import board_features as bf
from elephant_former.engine.elephant_chess_game import (
    ElephantChessGame,
    Player,
    R_CHARIOT,
    R_KING,
)

B_CHARIOT_ABS = R_CHARIOT  # black pieces are negated on placement
from elephant_former.evaluation.baseline_bots import RandomBot
from elephant_former.evaluation.board_match import MCTSBot, play_game
from elephant_former.inference.mcts import MCTS


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _tiny_module(seed: int = 0):
    from elephant_former.training.board_lightning_module import BoardLightningModule

    torch.manual_seed(seed)
    return BoardLightningModule(
        d_model=32, nhead=2, num_encoder_layers=1, dim_feedforward=64, policy_head_dim=32
    )


def _bot(sims: int, seed: int = 0, module=None) -> MCTSBot:
    return MCTSBot(module=module or _tiny_module(), device="cpu", num_simulations=sims, seed=seed)


def _mate_in_one_game() -> ElephantChessGame:
    """Red to move with forced mate (same fixture as the rerank tests)."""
    game = ElephantChessGame()
    game.board[:] = 0
    game.board[0, 4] = R_KING
    game.board[9, 3] = -R_KING
    game.board[7, 8] = R_CHARIOT
    game.board[8, 8] = R_CHARIOT
    game.current_player = Player.RED
    return game


def _is_mating_move(game: ElephantChessGame, move) -> bool:
    child = game.copy()
    child.apply_move(move)
    return not child.get_all_legal_moves_basic(child.current_player)


def test_plays_mate_in_one() -> None:
    game = _mate_in_one_game()
    move = _bot(sims=32).select_move(game)
    _check(move is not None and _is_mating_move(game, move), f"MCTS should mate, played {move}")
    print("  [ok] plays mate in one (exact terminal beats any net)")


def test_root_q_of_mate_is_one() -> None:
    game = _mate_in_one_game()
    bot = _bot(sims=64)
    move, q_by_index = bot._mcts.search(game)
    mate_indices = [
        bf.move_to_policy_index(m)
        for m in game.get_all_legal_moves_basic(Player.RED)
        if _is_mating_move(game, m)
    ]
    visited_mates = [i for i in mate_indices if i in q_by_index]
    _check(len(visited_mates) >= 1, "search should have visited the mating move")
    for i in visited_mates:
        _check(abs(q_by_index[i] - 1.0) < 1e-9, f"mating move Q should be exactly 1.0, got {q_by_index[i]}")
    for i, q in q_by_index.items():
        _check(-1e-9 <= q <= 1.0 + 1e-9, f"root Q out of range: {q}")
        if i not in mate_indices:
            _check(q < 1.0, "non-mating moves must not reach Q=1.0 in this position")
    print("  [ok] backup signs: mate Q == 1.0 at the root, others below")


def test_avoids_mate_in_two() -> None:
    # Black to move; king (3,9) is boxed (rank 8 swept by a red chariot,
    # (4,9)/(5,9) fall to the rank-9 lift or the flying general vs red K(5,0)).
    # Black's chariot (0,9) is the discriminator: shuffles along rank 9 or down
    # file 0 hang R(8,7)->(8,9)# (the black king blocks its own chariot's
    # defence of rank 9), while R->(4,9) interposes in advance and Rx(0,8)
    # removes the rank-8 sweeper — both genuinely safe. A 1-ply bot cannot
    # tell these apart; search must.
    game = ElephantChessGame()
    game.board[:] = 0
    game.board[0, 5] = R_KING       # red king (5,0)
    game.board[9, 3] = -R_KING      # black king (3,9)
    game.board[8, 0] = R_CHARIOT    # red chariot (0,8) — sweeps rank 8
    game.board[7, 8] = R_CHARIOT    # red chariot (8,7) — the rank-9 lift
    game.board[9, 0] = -B_CHARIOT_ABS  # black chariot (0,9), see below
    game.current_player = Player.BLACK

    legal = game.get_all_legal_moves_basic(Player.BLACK)
    traps = []
    safes = []
    for move in legal:
        child = game.copy()
        child.apply_move(move)
        if any(_is_mating_move(child, m) for m in child.get_all_legal_moves_basic(Player.RED)):
            traps.append(move)
        else:
            safes.append(move)
    _check(len(traps) >= 1, f"fixture broken: no trap moves among {legal}")
    _check(len(safes) >= 1, f"fixture broken: no safe moves among {legal}")

    # Terminal discovery is exact, so with enough budget any seed must land on
    # a safe move once the trap lines are explored to the mate.
    for seed in (0, 1):
        bot = _bot(sims=512, seed=seed, module=_tiny_module(seed))
        move = bot.select_move(game)
        _check(move in safes, f"seed {seed}: MCTS hung mate in one by playing {move}")
    print(f"  [ok] avoids mate in two ({len(traps)} traps, {len(safes)} safe moves)")


def test_deterministic_at_scale_zero() -> None:
    module = _tiny_module(seed=3)
    game = ElephantChessGame()
    move_a = _bot(sims=48, seed=0, module=module).select_move(game)
    move_b = _bot(sims=48, seed=7, module=module).select_move(game)
    _check(move_a == move_b, "gumbel_scale=0 search must be seed-independent")
    print("  [ok] deterministic with gumbel_scale=0")


def test_smoke_game_vs_random() -> None:
    bot = _bot(sims=24)
    status, winner = play_game(bot, RandomBot(seed=0), max_moves=30)
    _check(status in {"move_cap", "checkmate", "stalemate", "perpetual_check",
                      "perpetual_chase", "mutual_perpetual_check", "check_vs_chase",
                      "draw_by_repetition"},
           f"unexpected game status: {status}")
    print(f"  [ok] smoke game vs random completed ({status})")


if __name__ == "__main__":
    tests = [
        test_plays_mate_in_one,
        test_root_q_of_mate_is_one,
        test_avoids_mate_in_two,
        test_deterministic_at_scale_zero,
        test_smoke_game_vs_random,
    ]
    print("Running MCTS tests:")
    for test in tests:
        test()
    print("All MCTS tests passed.")
