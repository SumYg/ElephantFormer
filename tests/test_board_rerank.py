"""Value-head 1-ply rerank bot: game copy, terminal scoring, move selection.

Run: ``uv run python tests/test_board_rerank.py`` (exits non-zero on failure).
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
from elephant_former.evaluation.baseline_bots import RandomBot
from elephant_former.evaluation.board_match import ModelBot, ValueRerankBot, play_game


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _tiny_module(seed: int = 0):
    from elephant_former.training.board_lightning_module import BoardLightningModule

    torch.manual_seed(seed)
    return BoardLightningModule(
        d_model=32, nhead=2, num_encoder_layers=1, dim_feedforward=64, policy_head_dim=32
    )


def _mate_in_one_game() -> ElephantChessGame:
    """Red to move with forced mate: chariots on (8,7)/(8,8), black king on (3,9)."""
    game = ElephantChessGame()
    game.board[:] = 0
    game.board[0, 4] = R_KING       # red king (4,0)
    game.board[9, 3] = -R_KING      # black king (3,9)
    game.board[7, 8] = R_CHARIOT    # red chariot (8,7)
    game.board[8, 8] = R_CHARIOT    # red chariot (8,8)
    game.current_player = Player.RED
    return game


def _is_mating_move(game: ElephantChessGame, move) -> bool:
    child = game.copy()
    child.apply_move(move)
    return not child.get_all_legal_moves_basic(child.current_player) and child.is_king_in_check(
        child.current_player
    )


def _is_terminal_win_move(game: ElephantChessGame, move) -> bool:
    """Move after which the opponent has no reply — mate or stalemate (困毙),
    both of which win under xiangqi rules."""
    child = game.copy()
    child.apply_move(move)
    return not child.get_all_legal_moves_basic(child.current_player)


def test_game_copy_is_independent() -> None:
    game = ElephantChessGame()
    clone = game.copy()
    move = clone.get_all_legal_moves_basic(clone.current_player)[0]
    clone.apply_move(move)

    _check(game.current_player == Player.RED, "original player should be unchanged")
    _check(game.move_history == [], "original move history should be unchanged")
    _check(int((game.board != clone.board).sum()) > 0, "clone board should have diverged")
    _check(
        len(clone.position_sequence) == len(game.position_sequence) + 1,
        "clone position tracking should extend the original's, not share it",
    )
    print("  [ok] game copy is independent")


def test_child_features_reflect_move() -> None:
    game = ElephantChessGame()
    move = (1, 2, 4, 2)  # red cannon to the central file
    child = game.copy()
    child.apply_move(move)
    feats = bf.extract_features(child)

    _check(feats.side_to_move == 1, "child side-to-move should be BLACK")
    _check(
        int(feats.flags[bf.square_index(1, 2), bf.FLAG_PREV_FROM]) == 1,
        "child prev-from flag should mark the candidate move's from-square",
    )
    _check(
        int(feats.flags[bf.square_index(4, 2), bf.FLAG_PREV_TO]) == 1,
        "child prev-to flag should mark the candidate move's to-square",
    )
    print("  [ok] child features reflect the applied move")


def test_terminal_win_scores_one() -> None:
    # Mate and stalemate (困毙) both leave the opponent without a reply and
    # both win, so both must score exactly 1.0.
    bot = ValueRerankBot(module=_tiny_module(), device="cpu")
    game = _mate_in_one_game()
    legal = game.get_all_legal_moves_basic(Player.RED)
    mates = [m for m in legal if _is_mating_move(game, m)]
    _check(len(mates) >= 1, "test position should contain at least one mate in one")

    scores = bot.score_candidates(game, legal)
    for move, score in zip(legal, scores):
        _check(0.0 <= score <= 1.0, f"score out of range for {move}: {score}")
        if _is_terminal_win_move(game, move):
            _check(score == 1.0, f"terminal winning move {move} should score exactly 1.0")
        else:
            _check(score < 1.0, f"non-terminal move {move} should score below 1.0")
    print(f"  [ok] terminal wins score 1.0 ({len(mates)} mating moves found)")


def test_rerank_plays_terminal_win() -> None:
    # An untrained value head cannot beat the exact terminal bonus, so the bot
    # must play an immediately winning move (mate or stalemate) regardless of
    # the weights.
    bot = ValueRerankBot(module=_tiny_module(), device="cpu")
    game = _mate_in_one_game()
    move = bot.select_move(game)
    _check(move is not None, "bot should produce a move")
    _check(
        _is_terminal_win_move(game, move),
        f"rerank bot should play an immediately winning move, played {move}",
    )
    print("  [ok] rerank bot plays an immediate win")


def test_stalemate_is_a_win() -> None:
    # Black king boxed in without being in check: 困毙 — Red wins.
    game = _mate_in_one_game()
    game.current_player = Player.BLACK
    if game.get_all_legal_moves(Player.BLACK):
        raise AssertionError("test position should leave Black without a legal move")
    _check(not game.is_king_in_check(Player.BLACK), "Black should not be in check (stalemate, not mate)")
    status, winner = game.check_game_over()
    _check(status == "stalemate", f"expected stalemate status, got {status}")
    _check(winner == Player.RED, f"stalemate should be a win for RED, got {winner}")
    print("  [ok] stalemate adjudicated as a win for the opponent")


def test_repetition_penalty_caps_repeats() -> None:
    # Shuffle horses out and back so the start position has occurred twice;
    # replaying the same horse move then recreates a position seen once before.
    module = _tiny_module()
    game = ElephantChessGame()
    for move in [(1, 0, 2, 2), (1, 9, 2, 7), (2, 2, 1, 0), (2, 7, 1, 9)]:
        game.apply_move(move)

    repeat_move = (1, 0, 2, 2)   # recreates the position after the first move
    fresh_move = (7, 0, 6, 2)    # other horse: never-seen position

    plain = ValueRerankBot(module=module, device="cpu", repetition_penalty=0.0)
    penal = ValueRerankBot(module=module, device="cpu", repetition_penalty=0.1)

    plain_scores = plain.score_candidates(game, [repeat_move, fresh_move])
    penal_scores = penal.score_candidates(game, [repeat_move, fresh_move])

    _check(
        abs(plain_scores[1] - penal_scores[1]) < 1e-9,
        "fresh move's score must be unaffected by the repetition penalty",
    )
    expected = min(plain_scores[0], 0.5) - 0.1
    _check(
        abs(penal_scores[0] - expected) < 1e-6,
        f"repeat should score min(raw, 0.5) - penalty; got {penal_scores[0]}, expected {expected}",
    )
    print("  [ok] repetition penalty caps repeated positions")


def test_top_k_one_matches_policy_argmax() -> None:
    module = _tiny_module(seed=1)
    policy_bot = ModelBot(module=module, device="cpu", temperature=0.0)
    rerank_bot = ValueRerankBot(module=module, device="cpu", top_k=1)
    game = ElephantChessGame()
    _check(
        policy_bot.select_move(game) == rerank_bot.select_move(game),
        "top_k=1 rerank should reduce to the policy argmax move",
    )
    print("  [ok] top_k=1 reduces to policy argmax")


def test_smoke_game_vs_random() -> None:
    bot = ValueRerankBot(module=_tiny_module(), device="cpu", top_k=8)
    status, winner = play_game(bot, RandomBot(seed=0), max_moves=30)
    _check(status in {"move_cap", "checkmate", "stalemate", "perpetual_check",
                      "perpetual_chase", "mutual_perpetual_check", "check_vs_chase",
                      "draw_by_repetition"},
           f"unexpected game status: {status}")
    print(f"  [ok] smoke game vs random completed ({status})")


if __name__ == "__main__":
    tests = [
        test_game_copy_is_independent,
        test_child_features_reflect_move,
        test_terminal_win_scores_one,
        test_rerank_plays_terminal_win,
        test_stalemate_is_a_win,
        test_repetition_penalty_caps_repeats,
        test_top_k_one_matches_policy_argmax,
        test_smoke_game_vs_random,
    ]
    print("Running value-rerank tests:")
    for test in tests:
        test()
    print("All rerank tests passed.")
