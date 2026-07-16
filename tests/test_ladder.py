"""Ladder harness: SPRT math, Elo estimates, opening book, live rung smoke.

Run: ``uv run python tests/test_ladder.py`` (exits non-zero on failure).
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from elephant_former.evaluation.baseline_bots import RandomBot
from elephant_former.evaluation.board_match import play_game
from elephant_former.evaluation.ladder import (
    elo_estimate,
    load_opening_book,
    run_rung,
    sprt_bounds,
    sprt_llr,
)

BOOK_PATH = Path(__file__).resolve().parents[1] / "data" / "openings_wxf_top100.json"


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
    record = run_rung(
        bot, nodes=64, book=book[:1], num_games=2, max_moves=40,
        engine_path=None, nnue_path="tools/pikafish.nnue", threads=1,
    )
    _check(record["games"] == 2, "rung should play both paired games")
    _check(record["wins"] + record["draws"] + record["losses"] == 2, "results must sum")
    _check("elo" in record and "score" in record, "record must carry elo and score")
    print(f"  [ok] live rung smoke ({record['wins']}W-{record['losses']}L-{record['draws']}D vs pikafish@64)")


if __name__ == "__main__":
    tests = [
        test_sprt_llr,
        test_elo_estimate,
        test_opening_book_and_paired_application,
        test_live_rung_smoke,
    ]
    print("Running ladder harness tests:")
    for test in tests:
        test()
    print("All ladder tests passed.")
