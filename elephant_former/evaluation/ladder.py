"""Pikafish node-ladder match runner with paired openings and SPRT (Phase 1).

Plays the model against Pikafish at fixed node budgets (the roadmap's headline
metric: Elo vs Pikafish@N). Each opening from the book is played twice — model
as Red, then as Black — so first-move and opening bias cancel per pair. Results
are reported as W/D/L, score, an Elo estimate with a 95% CI, and (optionally) a
GSPRT log-likelihood ratio for a regression gate.

Every finished game is refereed by a second Pikafish process (see
:func:`audit_game`): the final position's legal moves are cross-checked against
Pikafish's own move generator (``go perft 1`` — after a checkmate/stalemate both
lists must be empty; any mismatch is reported as a rules disagreement), and a
move-capped game is adjudicated from the referee's evaluation instead of being
scored as a draw. W/D/L, Elo and SPRT use the adjudicated results; the raw
tally is kept alongside. With ``--games_out``, each game is logged as one JSON
line (moves, raw and adjudicated result, referee audit) for later inspection.

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
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from elephant_former.data_utils.pikafish_annotator import PikafishEngine, move_to_uci
from elephant_former.data_utils.tokenization_utils import parse_iccs_move_to_coords
from elephant_former.engine.elephant_chess_game import INITIAL_BOARD_FEN, Move, Player
from elephant_former.evaluation.baseline_bots import Bot, PikafishBot, default_engine_path
from elephant_former.evaluation.board_match import (
    MCTSBot,
    ModelBot,
    PlayedGame,
    ValueRerankBot,
    play_recorded_game,
)

RESULT_STRINGS = {Player.RED: "1-0", Player.BLACK: "0-1", None: "1/2-1/2"}
OUTCOME_WORDS = {"win": "model wins", "draw": "draw", "loss": "model loses"}


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


def adjudicate_cap(cp: int, is_mate: bool, side_to_move: Player, threshold_cp: int) -> Optional[Player]:
    """Winner of a move-capped game from the referee's eval, or ``None`` (stays a draw).

    ``cp`` is from the side to move's view (UCI convention). A mate score or
    ``|cp| >= threshold_cp`` awards the game to the side the engine favours.
    """
    if cp == 0 or (not is_mate and abs(cp) < threshold_cp):
        return None
    if cp > 0:
        return side_to_move
    return Player.BLACK if side_to_move == Player.RED else Player.RED


def audit_game(
    referee: PikafishEngine,
    played: PlayedGame,
    adjudicate_nodes: int = 1_000_000,
    adjudicate_cp: int = 300,
) -> Tuple[Optional[Player], dict]:
    """Referee a finished game; returns ``(winner after adjudication, audit dict)``.

    The final position's legal moves (our rules engine) are compared with
    Pikafish's own (``go perft 1``); after a checkmate/stalemate both must be
    empty, so any mismatch flags a rules disagreement. A final position that
    still has moves is evaluated at ``adjudicate_nodes`` (0 skips it), and only
    a ``move_cap`` game is re-scored from that eval (see :func:`adjudicate_cap`).
    Games always start from the standard initial position, so the referee gets
    the initial FEN plus the full move history (repetition-aware).
    """
    game = played.game
    history = [move_to_uci(m) for m in game.move_history]
    ours = {move_to_uci(m) for m in game.get_all_legal_moves_basic(game.current_player)}
    theirs = set(referee.legal_moves(INITIAL_BOARD_FEN, history))
    audit: dict = {"legal_ours": len(ours), "legal_pikafish": len(theirs), "legal_agree": ours == theirs}
    if ours != theirs:
        audit["only_ours"] = sorted(ours - theirs)
        audit["only_pikafish"] = sorted(theirs - ours)

    winner = played.winner
    if theirs and adjudicate_nodes > 0:
        _, pvs = referee.analyze_fen(INITIAL_BOARD_FEN, nodes=adjudicate_nodes, moves=history)
        if pvs:
            pv = pvs[0]
            red_cp = pv.cp if game.current_player == Player.RED else -pv.cp
            audit["eval"] = {"red_cp": red_cp, "is_mate": pv.is_mate, "depth": pv.depth, "nodes": adjudicate_nodes}
            if played.status == "move_cap":
                winner = adjudicate_cap(pv.cp, pv.is_mate, game.current_player, adjudicate_cp)
    audit["adjudicated"] = winner != played.winner
    return winner, audit


def eval_label(ev: dict) -> str:
    """Human-readable referee eval from :func:`audit_game`, e.g. ``+354cp for Red``."""
    if ev["is_mate"]:
        return "mate for " + ("Red" if ev["red_cp"] > 0 else "Black")
    return f"{ev['red_cp']:+d}cp for Red"


def _model_outcome(winner: Optional[Player], model_is_red: bool) -> str:
    """``"win"`` / ``"draw"`` / ``"loss"`` from the model's side."""
    if winner is None:
        return "draw"
    return "win" if (winner == Player.RED) == model_is_red else "loss"


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
    adjudicate_nodes: int = 1_000_000,
    adjudicate_cp: int = 300,
    games_out: Optional[str] = None,
    meta: Optional[dict] = None,
    wsl_distro: Optional[str] = None,
) -> dict:
    """Play one ladder rung (paired games vs Pikafish@nodes); returns a record dict.

    Results are refereed per :func:`audit_game`. With ``games_out``, one JSON
    line per game is appended there; ``meta`` (e.g. model path and bot) is merged
    into every game line and into the returned record. With ``wsl_distro``, the
    opponent and the referee run as the Linux build inside that WSL distro.
    """
    meta = meta or {}
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    counts: Counter = Counter()
    raw_counts: Counter = Counter()
    adjudicated_games = rules_disagreements = 0
    lower, upper = sprt_bounds()
    llr = 0.0
    sprt_status = None
    if games_out:
        Path(games_out).parent.mkdir(parents=True, exist_ok=True)
    pika = PikafishBot(
        engine_path=engine_path, nnue_path=nnue_path, nodes=nodes, threads=threads, wsl_distro=wsl_distro
    )
    # A separate process, so refereeing never touches the opponent's hash: the
    # opponent plays exactly as it would in an unrefereed run.
    referee = PikafishEngine(
        engine_path or default_engine_path(wsl=bool(wsl_distro)),
        nnue_path=nnue_path,
        threads=threads,
        wsl_distro=wsl_distro,
    )
    games_played = 0
    try:
        for pair in range(num_games // 2):
            opening_index = pair % len(book)
            for model_is_red in (True, False):
                red, black = (model_bot, pika) if model_is_red else (pika, model_bot)
                played = play_recorded_game(red, black, max_moves=max_moves, opening=book[opening_index])
                games_played += 1
                winner, audit = audit_game(referee, played, adjudicate_nodes, adjudicate_cp)
                outcome = _model_outcome(winner, model_is_red)
                counts[outcome] += 1
                raw_counts[_model_outcome(played.winner, model_is_red)] += 1
                if audit["adjudicated"]:
                    adjudicated_games += 1
                if verbose:
                    colour = "Red" if model_is_red else "Black"
                    note = f"; adjudicated, referee {eval_label(audit['eval'])}" if audit["adjudicated"] else ""
                    print(
                        f"  [n={nodes}] game {games_played}: opening {opening_index}, "
                        f"model as {colour} -> {played.status} ({OUTCOME_WORDS[outcome]}{note})",
                        flush=True,
                    )
                if not audit["legal_agree"]:
                    rules_disagreements += 1
                    print(
                        f"  [referee] game {games_played}: final-position legal moves differ "
                        f"(ours {audit['legal_ours']}, Pikafish {audit['legal_pikafish']}; "
                        f"only ours {audit['only_ours'][:5]}, only Pikafish {audit['only_pikafish'][:5]})",
                        flush=True,
                    )
                if games_out:
                    line = {
                        "rung_started": started,
                        "nodes": nodes,
                        "game": games_played,
                        "opening": opening_index,
                        "model_color": "red" if model_is_red else "black",
                        "status": played.status,
                        "raw_result": RESULT_STRINGS[played.winner],
                        "result": RESULT_STRINGS[winner],
                        "model_outcome": outcome,
                        "plies": len(played.game.move_history),
                        "moves": [move_to_uci(m) for m in played.game.move_history],
                        "referee": audit,
                        **meta,
                    }
                    with open(games_out, "a", encoding="utf-8") as f:
                        f.write(json.dumps(line) + "\n")
            if sprt is not None:
                llr = sprt_llr(counts["win"], counts["draw"], counts["loss"], elo0=sprt[0], elo1=sprt[1])
                if llr >= upper:
                    sprt_status = "H1"
                    break
                if llr <= lower:
                    sprt_status = "H0"
                    break
    finally:
        pika.close()
        referee.close()

    wins, draws, losses = counts["win"], counts["draw"], counts["loss"]
    elo, lo, hi = elo_estimate(wins, draws, losses)
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "started": started,
        "nodes": nodes,
        "games": games_played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": round((wins + 0.5 * draws) / games_played, 4) if games_played else 0.0,
        "elo": round(elo, 1),
        "elo_lo95": round(lo, 1),
        "elo_hi95": round(hi, 1),
        "raw": {"wins": raw_counts["win"], "draws": raw_counts["draw"], "losses": raw_counts["loss"]},
        "adjudicated_games": adjudicated_games,
        "adjudication": {"nodes": adjudicate_nodes, "cp": adjudicate_cp},
        "rules_disagreements": rules_disagreements,
        "engine_fallback_moves": pika.fallback_moves,
    }
    if games_out:
        record["games_out"] = games_out
    if sprt is not None:
        record["sprt"] = {"elo0": sprt[0], "elo1": sprt[1], "llr": round(llr, 3), "status": sprt_status or "inconclusive"}
    record.update(meta)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Model vs Pikafish node-ladder with paired openings.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Exported .safetensors weights, or a .ckpt you trust (it is unpickled).",
    )
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
    parser.add_argument(
        "--pikafish_wsl",
        type=str,
        default=None,
        help="Run the Linux Pikafish build inside this WSL distro (e.g. Ubuntu), for Windows hosts whose "
        "app-control policy blocks the .exe; --pikafish_engine/--pikafish_nnue stay Windows paths.",
    )
    parser.add_argument(
        "--adjudicate_nodes",
        type=int,
        default=1_000_000,
        help="Referee search budget for final positions that still have moves; move-capped games "
        "are adjudicated from it (0 = no evaluation, move caps stay draws).",
    )
    parser.add_argument(
        "--adjudicate_cp",
        type=int,
        default=300,
        help="A move-capped game goes to the side the referee favours by at least this many "
        "centipawns (or a mate score); otherwise it stays a draw.",
    )
    parser.add_argument(
        "--games_out",
        type=str,
        default="logs/ladder_games.jsonl",
        help="Append one JSON line per game (moves, raw/adjudicated result, referee audit); '' disables.",
    )
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

    meta = {
        "model_path": args.model_path,
        "bot": f"mcts{args.mcts_sims}" if args.mcts_sims > 0 else ("rerank" if args.rerank else "policy"),
    }
    if args.pikafish_wsl:
        meta["pikafish_wsl"] = args.pikafish_wsl
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
            adjudicate_nodes=args.adjudicate_nodes,
            adjudicate_cp=args.adjudicate_cp,
            games_out=args.games_out or None,
            meta=meta,
            wsl_distro=args.pikafish_wsl,
        )
        raw = record["raw"]
        print(
            f"nodes={record['nodes']}: {record['wins']}W-{record['losses']}L-{record['draws']}D "
            f"over {record['games']} games | score {record['score']:.1%} | "
            f"Elo {record['elo']:+.0f} [{record['elo_lo95']:+.0f}, {record['elo_hi95']:+.0f}]"
            + (
                f" | raw {raw['wins']}W-{raw['losses']}L-{raw['draws']}D, "
                f"{record['adjudicated_games']} move-cap game(s) adjudicated"
                if record["adjudicated_games"]
                else ""
            )
            + (f" | SPRT llr={record['sprt']['llr']} -> {record['sprt']['status']}" if sprt else "")
            + (f" | RULES DISAGREEMENTS: {record['rules_disagreements']}" if record["rules_disagreements"] else "")
            + (f" | engine fallbacks: {record['engine_fallback_moves']}" if record["engine_fallback_moves"] else "")
        )
        if args.out_jsonl:
            out = Path(args.out_jsonl)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
