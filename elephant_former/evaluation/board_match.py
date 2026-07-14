"""Play the board-state ElephantFormer against the baseline bots (Phase 0).

Provides a :class:`ModelBot` (forward pass -> mask illegal moves -> argmax or
temperature sample), a :class:`ValueRerankBot` (value-head 1-ply rerank of the
legal moves) and a small match runner, plus a CLI for the three sanity
matchups: ``model-vs-random``, ``model-vs-greedy``, ``greedy-vs-random``
(``--rerank`` upgrades the model side to the rerank bot).

Colours are alternated across games so neither side has a fixed first-move edge;
results are always reported from the first bot's perspective.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
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
        model_path: Optional[str] = None,
        device: str = "cpu",
        temperature: float = 0.0,
        seed: Optional[int] = None,
        module: Optional[BoardLightningModule] = None,
    ) -> None:
        self.name = "model"
        self.device = torch.device(device)
        self.temperature = temperature
        if module is None:
            if model_path is None:
                raise ValueError("Either model_path or module is required.")
            module = BoardLightningModule.load_from_checkpoint(
                checkpoint_path=model_path, map_location=self.device
            )
        self.model = module
        self.model.to(self.device)
        self.model.eval()
        self._generator = torch.Generator(device="cpu")
        if seed is not None:
            self._generator.manual_seed(seed)

    def _forward_features(
        self, feats_list: Sequence[bf.BoardFeatures]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch a list of positions through the model; returns (policy, value) logits."""
        piece_ids = torch.from_numpy(np.stack([f.piece_ids for f in feats_list])).long()
        flags = torch.from_numpy(np.stack([f.flags for f in feats_list])).long()
        side = torch.tensor([f.side_to_move for f in feats_list], dtype=torch.long)
        with torch.no_grad():
            return self.model(
                piece_ids.to(self.device), flags.to(self.device), side.to(self.device)
            )

    def select_move(self, game: ElephantChessGame) -> Optional[Move]:
        legal = game.get_all_legal_moves_basic(game.current_player)
        if not legal:
            return None

        feats = bf.extract_features(game, stm_legal_moves=legal)
        policy_logits, _ = self._forward_features([feats])

        legal_indices = [bf.move_to_policy_index(m) for m in legal]
        idx = select_move_index(
            policy_logits[0].cpu(),
            legal_indices,
            temperature=self.temperature,
            generator=self._generator,
        )
        return bf.policy_index_to_move(idx)


class ValueRerankBot(ModelBot):
    """Value-head 1-ply rerank: play the move whose resulting position the
    value head likes best for the mover.

    Every candidate move is applied to a copy of the game and the child position
    is scored ``P(loss) + 0.5 * P(draw)`` from the value head — the child's
    side-to-move is the opponent, so that is the mover's expected score. A move
    that leaves the opponent without a legal reply scores 1.0, whether by
    checkmate or stalemate (困毙: the stalemated player loses under xiangqi
    rules). A move recreating a position already seen in the game is capped at
    0.5 and penalised ``repetition_penalty`` per prior occurrence — repeats are
    at best drawish, and perpetual-check/chase adjudication makes the repeating
    side *lose*, so a deterministic bot must be steered off cycles. Ties break
    on the policy logit of the move; ``top_k`` restricts reranking to the
    ``top_k`` policy moves (0 = rerank all legal).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        top_k: int = 0,
        repetition_penalty: float = 0.1,
        seed: Optional[int] = None,
        module: Optional[BoardLightningModule] = None,
    ) -> None:
        super().__init__(
            model_path=model_path, device=device, temperature=0.0, seed=seed, module=module
        )
        self.name = "model-rerank"
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

    def score_candidates(self, game: ElephantChessGame, moves: Sequence[Move]) -> List[float]:
        """Expected score in ``[0, 1]`` for the mover after each candidate move."""
        scores = [0.0] * len(moves)
        pending: List[int] = []
        pending_feats: List[bf.BoardFeatures] = []
        pending_repeats: List[int] = []

        for i, move in enumerate(moves):
            child = game.copy()
            child.apply_move(move)
            replies = child.get_all_legal_moves_basic(child.current_player)
            if not replies:
                # No reply wins outright: checkmate, or stalemate (困毙 — the
                # stalemated player loses).
                scores[i] = 1.0
                continue
            pending.append(i)
            pending_feats.append(bf.extract_features(child, stm_legal_moves=replies))
            # Prior occurrences of the child position in this game (the count
            # includes the occurrence apply_move just recorded).
            pending_repeats.append(
                child.position_history[child.position_sequence[-1]] - 1
            )

        if pending:
            _, value_logits = self._forward_features(pending_feats)
            probs = torch.softmax(value_logits.float().cpu(), dim=-1)
            # Value classes are (loss, draw, win) for the child's side-to-move —
            # the opponent — so the mover's expected score is P(loss) + 0.5 P(draw).
            child_scores = probs[:, 0] + 0.5 * probs[:, 1]
            for i, score, repeats in zip(pending, child_scores.tolist(), pending_repeats):
                if repeats > 0 and self.repetition_penalty > 0.0:
                    score = min(score, 0.5) - self.repetition_penalty * repeats
                scores[i] = float(score)
        return scores

    def select_move(self, game: ElephantChessGame) -> Optional[Move]:
        legal = game.get_all_legal_moves_basic(game.current_player)
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]

        feats = bf.extract_features(game, stm_legal_moves=legal)
        policy_logits, _ = self._forward_features([feats])
        policy_logits = policy_logits[0].cpu()
        move_logits = [float(policy_logits[bf.move_to_policy_index(m)]) for m in legal]

        candidates = sorted(range(len(legal)), key=lambda i: move_logits[i], reverse=True)
        if 0 < self.top_k < len(candidates):
            candidates = candidates[: self.top_k]

        scores = self.score_candidates(game, [legal[i] for i in candidates])
        best = max(
            range(len(candidates)),
            key=lambda j: (scores[j], move_logits[candidates[j]]),
        )
        return legal[candidates[best]]


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


def _make_model_bot(args: argparse.Namespace) -> Bot:
    if args.rerank:
        return ValueRerankBot(
            args.model_path,
            device=args.device,
            top_k=args.rerank_top_k,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )
    return ModelBot(args.model_path, device=args.device, temperature=args.temperature, seed=args.seed)


def _build_bots(args: argparse.Namespace) -> Tuple[Bot, Bot]:
    if args.mode == "model-vs-random":
        return _make_model_bot(args), make_bot("random", seed=args.seed)
    if args.mode == "model-vs-greedy":
        return _make_model_bot(args), make_bot("greedy", seed=args.seed)
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
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Use the value-head 1-ply rerank bot for the model side (deterministic; ignores --temperature).",
    )
    parser.add_argument(
        "--rerank_top_k",
        type=int,
        default=0,
        help="Rerank only the top-k moves by policy logit (0 = rerank every legal move).",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=0.1,
        help="Rerank score penalty per prior occurrence of the resulting position "
        "(repeats are also capped at a draw score; 0 disables).",
    )
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
