"""Build a paired-openings book for the ladder from a PGN corpus.

Takes the most common opening prefixes (default 6 plies) across the corpus,
replays each through the rules engine to guarantee legality, and writes a JSON
book of ICCS move lists for :mod:`elephant_former.evaluation.ladder`.

Run:
    uv run python scripts/build_opening_book.py \
        --pgn_file_path data/WXF-41743games.pgns \
        --out data/openings_wxf_top100.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elephant_former.data.elephant_parser import parse_iccs_pgn_file
from elephant_former.data_utils.tokenization_utils import parse_iccs_move_to_coords
from elephant_former.engine.elephant_chess_game import ElephantChessGame


def _replays_legally(opening: tuple) -> bool:
    game = ElephantChessGame()
    for iccs in opening:
        coords = parse_iccs_move_to_coords(iccs)
        if coords is None or coords not in game.get_all_legal_moves_basic(game.current_player):
            return False
        game.apply_move(coords)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an opening book from a PGN corpus.")
    parser.add_argument("--pgn_file_path", type=str, required=True)
    parser.add_argument("--plies", type=int, default=6, help="Opening length in plies (half-moves).")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    games = parse_iccs_pgn_file(args.pgn_file_path)
    print(f"Parsed {len(games)} games from {args.pgn_file_path}.")

    prefixes = Counter(
        tuple(g.parsed_moves[: args.plies]) for g in games if len(g.parsed_moves) >= args.plies
    )
    print(f"{len(prefixes)} distinct {args.plies}-ply openings; validating the most common ...")

    book = []
    for opening, count in prefixes.most_common():
        if len(book) >= args.top_k:
            break
        if _replays_legally(opening):
            book.append({"moves": list(opening), "count": count})
        else:
            print(f"  skipping illegal/unparseable opening (count {count}): {opening}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": str(args.pgn_file_path),
                "plies": args.plies,
                "openings": [entry["moves"] for entry in book],
                "counts": [entry["count"] for entry in book],
            },
            f,
            indent=1,
        )
    coverage = sum(e["count"] for e in book) / max(1, sum(prefixes.values()))
    print(f"Wrote {len(book)} openings to {out} (covers {coverage:.1%} of games).")


if __name__ == "__main__":
    main()
