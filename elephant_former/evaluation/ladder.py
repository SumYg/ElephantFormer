"""Pikafish node-ladder match runner with paired openings and SPRT (Phase 1).

Plays the model against Pikafish at fixed node budgets (the roadmap's headline
metric: Elo vs Pikafish@N). Each opening from the book is played twice — model
as Red, then as Black — so first-move and opening bias cancel per pair. Results
are reported as W/D/L, score, an Elo estimate with a 95% CI, and (optionally) a
GSPRT log-likelihood ratio for a regression gate.

SPRT follows the fishtest-style trinomial GSPRT approximation: with empirical
per-game mean ``m`` and variance ``v`` of the score,
``LLR ~= (s1 - s0) * (2m - s0 - s1) * N / (2v)`` where ``s0``/``s1`` are the
logistic scores of ``elo0``/``elo1``. Accept H1 when LLR >= log((1-beta)/alpha),
accept H0 when LLR <= log(beta/(1-alpha)).

Each rung appends a JSON line to ``--out_jsonl`` for the Elo-vs-N chart.

Example:
    uv run python -m elephant_former.evaluation.ladder \
        --model_path checkpoints/board-combined/<best>.ckpt --rerank \
        --nodes 256 1024 --num_games 40 --book data/openings_wxf_top100.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from elephant_former.data_utils.tokenization_utils import parse_iccs_move_to_coords
from elephant_former.engine.elephant_chess_game import Move, Player
from elephant_former.evaluation.baseline_bots import Bot, PikafishBot
from elephant_former.evaluation.board_match import MCTSBot, ModelBot, ValueRerankBot, play_game


def load_opening_book(path: str | Path) -> List[List[Move]]:
    """Load a JSON opening book (lists of ICCS move strings) as coordinate moves."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    book: List[List[Move]] = []
    for i, opening in enumerate(data["openings"]):
        moves = [parse_iccs_move_to_coords(m) for m in opening]
        if any(m is None for m in moves):
            raise ValueError(f"Opening {i} has an unparseable ICCS move: {opening}")
        book.append(moves)  # type: ignore[arg-type]
    if not book:
        raise ValueError(f"Opening book {path} is empty.")
    return book


def logistic_score(elo: float) -> float:
    """Expected score of a player rated ``elo`` above the opponent."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def sprt_llr(wins: int, draws: int, losses: int, elo0: float = 0.0, elo1: float = 20.0) -> float:
    """Trinomial GSPRT log-likelihood ratio (fishtest-style approximation)."""
    n = wins + draws + losses
    if n == 0:
        return 0.0
    mean = (wins + 0.5 * draws) / n
    var = (
        wins * (1.0 - mean) ** 2 + draws * (0.5 - mean) ** 2 + losses * (0.0 - mean) ** 2
    ) / n
    if var <= 0.0:
        # Degenerate (all results identical): decide by which hypothesis fits.
        midpoint = 0.5 * (logistic_score(elo0) + logistic_score(elo1))
        return math.inf if mean > midpoint else -math.inf
    s0, s1 = logistic_score(elo0), logistic_score(elo1)
    return (s1 - s0) * (2.0 * mean - s0 - s1) * n / (2.0 * var)


def sprt_bounds(alpha: float = 0.05, beta: float = 0.05) -> Tuple[float, float]:
    """(lower, upper) LLR bounds: cross lower -> accept H0, upper -> accept H1."""
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


def elo_estimate(wins: int, draws: int, losses: int) -> Tuple[float, float, float]:
    """(elo, lo95, hi95) from W/D/L via the logistic model; clamped when degenerate."""
    n = wins + draws + losses
    if n == 0:
        return 0.0, 0.0, 0.0
    eps = 0.5 / n  # continuity clamp so all-win/all-loss stays finite
    mean = min(max((wins + 0.5 * draws) / n, eps), 1.0 - eps)
    var = (
        wins * (1.0 - mean) ** 2 + draws * (0.5 - mean) ** 2 + losses * (0.0 - mean) ** 2
    ) / n
    se = math.sqrt(var / n)

    def to_elo(score: float) -> float:
        score = min(max(score, eps), 1.0 - eps)
        return 400.0 * math.log10(score / (1.0 - score))

    return to_elo(mean), to_elo(mean - 1.96 * se), to_elo(mean + 1.96 * se)


def run_rung(
    model_bot: Bot,
    nodes: int,
    book: Sequence[Sequence[Move]],
    num_games: int,
    max_moves: int,
    engine_path: Optional[str],
    nnue_path: Optional[str],
    threads: int,
    sprt: Optional[Tuple[float, float]] = None,
    verbose: bool = False,
) -> dict:
    """Play one ladder rung (paired games vs Pikafish@nodes); returns a record dict."""
    wins = draws = losses = 0
    lower, upper = sprt_bounds()
    llr = 0.0
    sprt_status = None
    pika = PikafishBot(engine_path=engine_path, nnue_path=nnue_path, nodes=nodes, threads=threads)
    games_played = 0
    try:
        for pair in range(num_games // 2):
            opening = book[pair % len(book)]
            for model_is_red in (True, False):
                red, black = (model_bot, pika) if model_is_red else (pika, model_bot)
                status, winner = play_game(red, black, max_moves=max_moves, opening=opening)
                games_played += 1
                if winner is None:
                    draws += 1
                    outcome = "draw"
                elif (winner == Player.RED) == model_is_red:
                    wins += 1
                    outcome = "model wins"
                else:
                    losses += 1
                    outcome = "model loses"
                if verbose:
                    colour = "Red" if model_is_red else "Black"
                    print(
                        f"  [n={nodes}] game {games_played}: opening {pair % len(book)}, "
                        f"model as {colour} -> {status} ({outcome})",
                        flush=True,
                    )
            if sprt is not None:
                llr = sprt_llr(wins, draws, losses, elo0=sprt[0], elo1=sprt[1])
                if llr >= upper:
                    sprt_status = "H1"
                    break
                if llr <= lower:
                    sprt_status = "H0"
                    break
    finally:
        pika.close()

    elo, lo, hi = elo_estimate(wins, draws, losses)
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "nodes": nodes,
        "games": games_played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": round((wins + 0.5 * draws) / games_played, 4) if games_played else 0.0,
        "elo": round(elo, 1),
        "elo_lo95": round(lo, 1),
        "elo_hi95": round(hi, 1),
        "engine_fallback_moves": pika.fallback_moves,
    }
    if sprt is not None:
        record["sprt"] = {"elo0": sprt[0], "elo1": sprt[1], "llr": round(llr, 3), "status": sprt_status or "inconclusive"}
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Model vs Pikafish node-ladder with paired openings.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--rerank", action="store_true", help="Use the value-head 1-ply rerank bot.")
    parser.add_argument("--rerank_top_k", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=0.1)
    parser.add_argument("--mcts_sims", type=int, default=0, help="Use MCTS with this budget per move (0 = off).")
    parser.add_argument("--mcts_top_m", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--nodes", type=int, nargs="+", default=[256], help="Ladder rungs (one match per value).")
    parser.add_argument("--num_games", type=int, default=40, help="Games per rung (rounded down to even).")
    parser.add_argument("--max_moves", type=int, default=200)
    parser.add_argument("--book", type=str, default="data/openings_wxf_top100.json")
    parser.add_argument("--sprt", action="store_true", help="Stop a rung early on a GSPRT decision.")
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=20.0)
    parser.add_argument("--pikafish_engine", type=str, default=None)
    parser.add_argument("--pikafish_nnue", type=str, default="tools/pikafish.nnue")
    parser.add_argument("--pikafish_threads", type=int, default=1)
    parser.add_argument("--out_jsonl", type=str, default="logs/ladder.jsonl", help="Append one JSON record per rung ('' disables).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    book = load_opening_book(args.book)
    print(f"Loaded {len(book)} openings from {args.book}.")

    if args.mcts_sims > 0 and args.rerank:
        raise SystemExit("--mcts_sims and --rerank are mutually exclusive.")
    if args.mcts_sims > 0:
        model_bot: Bot = MCTSBot(
            args.model_path,
            device=args.device,
            num_simulations=args.mcts_sims,
            root_top_m=args.mcts_top_m,
        )
    elif args.rerank:
        model_bot = ValueRerankBot(
            args.model_path,
            device=args.device,
            top_k=args.rerank_top_k,
            repetition_penalty=args.repetition_penalty,
        )
    else:
        model_bot = ModelBot(args.model_path, device=args.device)

    sprt = (args.elo0, args.elo1) if args.sprt else None
    for nodes in args.nodes:
        num_games = args.num_games - (args.num_games % 2)
        print(f"\n=== Rung: Pikafish@{nodes} nodes, up to {num_games} paired games ===", flush=True)
        record = run_rung(
            model_bot,
            nodes=nodes,
            book=book,
            num_games=num_games,
            max_moves=args.max_moves,
            engine_path=args.pikafish_engine,
            nnue_path=args.pikafish_nnue,
            threads=args.pikafish_threads,
            sprt=sprt,
            verbose=args.verbose,
        )
        record["model_path"] = args.model_path
        record["bot"] = (
            f"mcts{args.mcts_sims}" if args.mcts_sims > 0 else ("rerank" if args.rerank else "policy")
        )
        print(
            f"nodes={record['nodes']}: {record['wins']}W-{record['losses']}L-{record['draws']}D "
            f"over {record['games']} games | score {record['score']:.1%} | "
            f"Elo {record['elo']:+.0f} [{record['elo_lo95']:+.0f}, {record['elo_hi95']:+.0f}]"
            + (f" | SPRT llr={record['sprt']['llr']} -> {record['sprt']['status']}" if sprt else "")
            + (f" | engine fallbacks: {record['engine_fallback_moves']}" if record["engine_fallback_moves"] else "")
        )
        if args.out_jsonl:
            out = Path(args.out_jsonl)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
