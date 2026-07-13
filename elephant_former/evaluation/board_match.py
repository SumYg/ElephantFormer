"""Play the board-state ElephantFormer against the baseline bots (Phase 0).

Provides a :class:`ModelBot` (forward pass -> mask illegal moves -> argmax or
temperature sample) and a small match runner, plus a CLI for the three sanity
matchups: ``model-vs-random``, ``model-vs-greedy``, ``greedy-vs-random``.

Colours are alternated across games so neither side has a fixed first-move edge;
results are always reported from the first bot's perspective.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from elephant_former.data_utils import board_features as bf
from elephant_former.evaluation.baseline_bots import Bot, make_bot
from elephant_former.engine.elephant_chess_game import ElephantChessGame, Move, Player
from elephant_former.models.board_transformer import select_move_index
from elephant_former.training.board_lightning_module import BoardLightningModule


class ModelBot(Bot):
    """Bot backed by a trained :class:`BoardLightningModule` checkpoint."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.name = "model"
        self.device = torch.device(device)
        self.temperature = temperature
        self.model = BoardLightningModule.load_from_checkpoint(
            checkpoint_path=model_path, map_location=self.device
        )
        self.model.to(self.device)
        self.model.eval()
        self._generator = torch.Generator(device="cpu")
        if seed is not None:
            self._generator.manual_seed(seed)

    def select_move(self, game: ElephantChessGame) -> Optional[Move]:
        legal = game.get_all_legal_moves_basic(game.current_player)
        if not legal:
            return None

        feats = bf.extract_features(game, stm_legal_moves=legal)
        piece_ids = torch.from_numpy(feats.piece_ids).long().unsqueeze(0).to(self.device)
        flags = torch.from_numpy(feats.flags).long().unsqueeze(0).to(self.device)
        side = torch.tensor([feats.side_to_move], dtype=torch.long, device=self.device)

        with torch.no_grad():
            policy_logits, _ = self.model(piece_ids, flags, side)

        legal_indices = [bf.move_to_policy_index(m) for m in legal]
        idx = select_move_index(
            policy_logits[0].cpu(),
            legal_indices,
            temperature=self.temperature,
            generator=self._generator,
        )
        return bf.policy_index_to_move(idx)


@dataclass
class MatchResult:
    """Aggregate result of a match from the first bot's perspective."""

    bot_a: str
    bot_b: str
    games: int
    wins_a: int
    wins_b: int
    draws: int

    @property
    def win_rate_a(self) -> float:
        return 100.0 * self.wins_a / self.games if self.games else 0.0


def play_game(
    red_bot: Bot, black_bot: Bot, max_moves: int = 200
) -> Tuple[Optional[str], Optional[Player]]:
    """Play a single game; return ``(status, winner)`` (winner ``None`` on draw)."""
    game = ElephantChessGame()
    for _ in range(max_moves):
        mover = game.current_player
        bot = red_bot if mover == Player.RED else black_bot
        move = bot.select_move(game)
        if move is None:
            if game.is_king_in_check(mover):
                return "checkmate", game.get_opponent(mover)
            # 困毙: no legal moves loses even without check.
            return "stalemate", game.get_opponent(mover)

        game.apply_move(move)
        status, winner = game.check_game_over()
        if status:
            return status, winner
    return "move_cap", None


def play_match(
    bot_a: Bot, bot_b: Bot, num_games: int = 10, max_moves: int = 200, verbose: bool = False
) -> MatchResult:
    """Play ``num_games`` games alternating colours; report from ``bot_a``'s view."""
    wins_a = wins_b = draws = 0
    for i in range(num_games):
        a_is_red = i % 2 == 0
        red_bot, black_bot = (bot_a, bot_b) if a_is_red else (bot_b, bot_a)
        status, winner = play_game(red_bot, black_bot, max_moves=max_moves)

        if winner is None:
            draws += 1
            outcome = "draw"
        else:
            a_won = (winner == Player.RED) == a_is_red
            if a_won:
                wins_a += 1
                outcome = f"{bot_a.name} wins"
            else:
                wins_b += 1
                outcome = f"{bot_b.name} wins"
        if verbose:
            colour = "Red" if a_is_red else "Black"
            print(f"  game {i + 1}: {bot_a.name} as {colour} -> {status} ({outcome})")

    return MatchResult(
        bot_a=bot_a.name,
        bot_b=bot_b.name,
        games=num_games,
        wins_a=wins_a,
        wins_b=wins_b,
        draws=draws,
    )


def _build_bots(args: argparse.Namespace) -> Tuple[Bot, Bot]:
    if args.mode == "model-vs-random":
        model = ModelBot(args.model_path, device=args.device, temperature=args.temperature, seed=args.seed)
        return model, make_bot("random", seed=args.seed)
    if args.mode == "model-vs-greedy":
        model = ModelBot(args.model_path, device=args.device, temperature=args.temperature, seed=args.seed)
        return model, make_bot("greedy", seed=args.seed)
    if args.mode == "greedy-vs-random":
        return make_bot("greedy", seed=args.seed), make_bot("random", seed=args.seed)
    raise ValueError(f"Unknown mode: {args.mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play board-state ElephantFormer vs baseline bots.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["model-vs-random", "model-vs-greedy", "greedy-vs-random"],
    )
    parser.add_argument("--model_path", type=str, default=None, help="Checkpoint path (required for model modes).")
    parser.add_argument("--num_games", type=int, default=10)
    parser.add_argument("--max_moves", type=int, default=200, help="Move (ply) cap per game.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = argmax; >0 samples legal moves.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.mode.startswith("model") and not args.model_path:
        parser.error("--model_path is required for model modes.")

    bot_a, bot_b = _build_bots(args)
    print(f"Playing {args.num_games} games: {bot_a.name} vs {bot_b.name} (max {args.max_moves} moves/game)")
    result = play_match(bot_a, bot_b, num_games=args.num_games, max_moves=args.max_moves, verbose=args.verbose)

    print("\n--- Match Result ---")
    print(f"{result.bot_a} vs {result.bot_b} over {result.games} games")
    print(f"  {result.bot_a} wins: {result.wins_a} ({result.win_rate_a:.1f}%)")
    print(f"  {result.bot_b} wins: {result.wins_b}")
    print(f"  draws: {result.draws}")


if __name__ == "__main__":
    main()
