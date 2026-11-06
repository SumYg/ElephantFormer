"""Ladder harness: SPRT math, Elo estimates, opening book, referee, live rung smoke.

Run: ``uv run python tests/test_ladder.py`` (exits non-zero on failure). The live
rung uses the bundled Pikafish under ``tools/``; with ``PIKAFISH_WSL=<distro>`` it
runs the Linux build inside that WSL distro.
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from elephant_former.data_utils.pikafish_annotator import PvLine, move_to_uci
from elephant_former.engine.elephant_chess_game import ElephantChessGame, Player
from elephant_former.evaluation.baseline_bots import RandomBot
from elephant_former.evaluation.board_match import PlayedGame, play_game, play_recorded_game
from elephant_former.evaluation.ladder import (
    adjudicate_cap,
    audit_game,
    elo_estimate,
    eval_label,
    load_opening_book,
    run_rung,
    sprt_bounds,
    sprt_llr,
)

BOOK_PATH = Path(__file__).resolve().parents[1] / "data" / "openings_wxf_top100.json"
# Black to move, in check from the d-file rook, with no legal reply (e9 faces the red king).
MATED_FEN = "3k5/9/9/9/9/9/9/9/3R5/4K4 b - - 0 1"


class _StubReferee:
    """Stands in for the referee engine: canned legal moves and evaluation."""

    def __init__(self, legal, cp=0, is_mate=False):
        self._legal = list(legal)
        self._cp = cp
        self._is_mate = is_mate

    def legal_moves(self, fen, moves=None):
        return list(self._legal)

    def analyze_fen(self, fen, nodes=10000, moves=None):
        best = self._legal[0] if self._legal else None
        return best, [PvLine(move=best or "", cp=self._cp, is_mate=self._is_mate, depth=20)]


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_sprt_llr() -> None:
    _check(sprt_llr(0, 0, 0) == 0.0, "no games -> llr 0")
    _check(sprt_llr(50, 0, 0) == math.inf, "all wins -> +inf")
    _check(sprt_llr(0, 0, 50) == -math.inf, "all losses -> -inf")
    _check(
        sprt_llr(60, 0, 40) > sprt_llr(50, 0, 50) > sprt_llr(40, 0, 60),
        "llr must increase with win count",
    )
    _check(sprt_llr(50, 0, 50, elo0=0, elo1=20) < 0, "exact 50% is evidence against elo1=20")
    lower, upper = sprt_bounds(0.05, 0.05)
    _check(lower < 0 < upper and abs(lower + upper) < 1e-9, "alpha=beta bounds are symmetric")
    print("  [ok] SPRT llr and bounds")


def test_elo_estimate() -> None:
    elo, lo, hi = elo_estimate(50, 0, 50)
    _check(abs(elo) < 1e-9, "50% score -> 0 Elo")
    _check(lo < elo < hi, "CI must bracket the estimate")
    elo, _, _ = elo_estimate(75, 0, 25)
    _check(abs(elo - 400 * math.log10(3)) < 1e-6, "75% -> ~+190.8 Elo")
    elo, lo, hi = elo_estimate(10, 0, 0)
    _check(math.isfinite(elo) and math.isfinite(hi), "all-win estimate stays finite")
    print("  [ok] Elo estimates")


def test_opening_book_and_paired_application() -> None:
    book = load_opening_book(BOOK_PATH)
    _check(len(book) == 100, f"expected 100 openings, got {len(book)}")
    _check(all(len(o) == 6 for o in book), "every opening should be 6 plies")
    for opening in book[:5]:
        status, winner = play_game(RandomBot(seed=1), RandomBot(seed=2), max_moves=20, opening=opening)
        _check(status is not None or winner is None, "game after opening must complete")
    try:
        play_game(RandomBot(seed=1), RandomBot(seed=2), opening=[(0, 0, 0, 9)])
        raise AssertionError("illegal opening move must raise")
    except ValueError:
        pass
    print("  [ok] opening book loads and applies")


def test_adjudicate_cap() -> None:
    _check(adjudicate_cap(350, False, Player.RED, 300) == Player.RED, "+350 for Red to move -> Red")
    _check(adjudicate_cap(350, False, Player.BLACK, 300) == Player.BLACK, "+350 for Black to move -> Black")
    _check(adjudicate_cap(-350, False, Player.RED, 300) == Player.BLACK, "-350 for Red to move -> Black")
    _check(adjudicate_cap(299, False, Player.RED, 300) is None, "below threshold stays a draw")
    _check(adjudicate_cap(-29990, True, Player.RED, 30000) == Player.BLACK, "mate score ignores threshold")
    _check(adjudicate_cap(0, False, Player.RED, 0) is None, "a dead-level eval is never a win")
    _check(eval_label({"red_cp": -354, "is_mate": False}) == "-354cp for Red", "cp label")
    _check(eval_label({"red_cp": -29937, "is_mate": True}) == "mate for Black", "mate label must name the mating side")
    _check(eval_label({"red_cp": 29990, "is_mate": True}) == "mate for Red", "mate label for Red")
    print("  [ok] move-cap adjudication rule and eval labels")


def test_audit_game_with_stub_referee() -> None:
    capped = play_recorded_game(RandomBot(seed=1), RandomBot(seed=2), max_moves=12)
    _check(capped.status == "move_cap" and capped.winner is None, "12 random plies should hit the cap")
    game = capped.game
    ours = sorted(move_to_uci(m) for m in game.get_all_legal_moves_basic(game.current_player))
    stm = game.current_player
    other = Player.BLACK if stm == Player.RED else Player.RED

    winner, audit = audit_game(_StubReferee(ours, cp=500), capped)
    _check(audit["legal_agree"] and winner == stm, "+500 for the side to move wins a capped game")
    _check(audit["adjudicated"], "a decided cap must be marked adjudicated")
    _check(audit["eval"]["red_cp"] == (500 if stm == Player.RED else -500), "eval stored from Red's view")

    winner, audit = audit_game(_StubReferee(ours, cp=-120), capped)
    _check(winner is None and not audit["adjudicated"], "small edge leaves the cap a draw")
    _check("eval" in audit, "the referee eval is recorded even when not decisive")

    winner, audit = audit_game(_StubReferee(ours, cp=-900), capped, adjudicate_nodes=0)
    _check(winner is None and "eval" not in audit, "adjudicate_nodes=0 skips evaluation")

    winner, audit = audit_game(_StubReferee(ours[1:], cp=-900), capped)
    _check(not audit["legal_agree"] and audit["only_ours"] == ours[:1], "missing move must be reported")
    _check(winner == other, "a disagreement is flagged but does not block adjudication")

    mated = ElephantChessGame(fen=MATED_FEN)
    _check(not mated.get_all_legal_moves_basic(mated.current_player), "MATED_FEN must leave no legal moves")
    played = PlayedGame("checkmate", Player.RED, mated)
    winner, audit = audit_game(_StubReferee([]), played)
    _check(audit["legal_agree"] and winner == Player.RED and "eval" not in audit, "agreed mate stands")
    winner, audit = audit_game(_StubReferee(["d9d8"]), played)
    _check(not audit["legal_agree"] and audit["only_pikafish"] == ["d9d8"], "engine escape must be flagged")
    _check(winner == Player.RED, "terminal results are flagged, never re-scored")
    print("  [ok] referee audit: agreement, adjudication, disagreements")


def test_live_rung_smoke() -> None:
    # Tiny module, tiny node budget, 2 games: plumbing only.
    from elephant_former.evaluation.board_match import ModelBot
    from elephant_former.training.board_lightning_module import BoardLightningModule

    torch.manual_seed(0)
    module = BoardLightningModule(
        d_model=32, nhead=2, num_encoder_layers=1, dim_feedforward=64, policy_head_dim=32
    )
    bot = ModelBot(module=module, device="cpu")
    book = load_opening_book(BOOK_PATH)
    with tempfile.TemporaryDirectory() as tmp:
        games_out = str(Path(tmp) / "games.jsonl")
        record = run_rung(
            bot, nodes=64, book=book[:1], num_games=2, max_moves=40,
            engine_path=None, nnue_path="tools/pikafish.nnue", threads=1,
            adjudicate_nodes=20000, games_out=games_out, meta={"bot": "smoke"},
            wsl_distro=os.environ.get("PIKAFISH_WSL"),
        )
        with open(games_out, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
    _check(record["games"] == 2, "rung should play both paired games")
    _check(record["wins"] + record["draws"] + record["losses"] == 2, "results must sum")
    _check(sum(record["raw"].values()) == 2, "raw results must sum")
    _check("elo" in record and "score" in record, "record must carry elo and score")
    _check(record["rules_disagreements"] == 0, "rules engine and Pikafish disagree on a final position")
    _check(record["bot"] == "smoke", "meta must be merged into the record")
    _check(len(lines) == 2 and all(g["bot"] == "smoke" for g in lines), "one game line per game, with meta")
    for g in lines:
        _check(g["plies"] == len(g["moves"]) and g["moves"][:6] == [move_to_uci(m) for m in book[0]],
               "game line must hold the full move list, opening included")
        _check(g["referee"]["legal_agree"], f"final-position disagreement in game {g['game']}")
        _check(g["raw_result"] == g["result"] or g["status"] == "move_cap", "only move caps get re-scored")
    print(
        f"  [ok] live rung smoke ({record['wins']}W-{record['losses']}L-{record['draws']}D vs pikafish@64, "
        f"{record['adjudicated_games']} adjudicated)"
    )


if __name__ == "__main__":
    tests = [
        test_sprt_llr,
        test_elo_estimate,
        test_opening_book_and_paired_application,
        test_adjudicate_cap,
        test_audit_game_with_stub_referee,
        test_live_rung_smoke,
    ]
    print("Running ladder harness tests:")
    for test in tests:
        test()
    print("All ladder tests passed.")
