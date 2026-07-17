"""Render the Elo-vs-nodes ladder chart from logs/ladder.jsonl.

The roadmap's headline artifact: Pikafish node budget N on a log x-axis, our
Elo relative to Pikafish@N on the y-axis, one line per model generation + bot
configuration. Series colors follow a fixed categorical order (assigned by
first appearance in the log, never re-cycled), 95% CIs as error bars, and a
dashed parity line at 0 Elo.

Run:
    uv run python scripts/plot_ladder.py                        # logs/ladder.jsonl -> logs/ladder_elo_vs_nodes.png
    uv run python scripts/plot_ladder.py --jsonl <file> --out <png>
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fixed categorical order (validated palette; order is the CVD-safety mechanism).
SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def series_label(record: dict) -> str:
    stem = Path(record.get("model_path", "?")).stem
    stem = stem.replace("board_former-", "").replace("-val_loss", " vl")
    parent = Path(record.get("model_path", "?")).parent.name or "?"
    return f"{parent}/{stem} · {record.get('bot', '?')}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Elo vs Pikafish nodes from ladder JSONL records.")
    parser.add_argument("--jsonl", type=str, default="logs/ladder.jsonl")
    parser.add_argument("--out", type=str, default="logs/ladder_elo_vs_nodes.png")
    args = parser.parse_args()

    records = []
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise SystemExit(f"No records in {args.jsonl}.")

    # Group by series, keep the latest record per (series, nodes).
    series: "OrderedDict[str, dict]" = OrderedDict()
    for rec in records:
        key = series_label(rec)
        series.setdefault(key, {})[rec["nodes"]] = rec

    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, (label, by_nodes) in enumerate(series.items()):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        nodes = sorted(by_nodes)
        elo = [by_nodes[n]["elo"] for n in nodes]
        lo = [by_nodes[n]["elo"] - by_nodes[n]["elo_lo95"] for n in nodes]
        hi = [by_nodes[n]["elo_hi95"] - by_nodes[n]["elo"] for n in nodes]
        ax.errorbar(
            nodes, elo, yerr=[lo, hi], color=color, linewidth=2, marker="o",
            markersize=6, capsize=3, elinewidth=1, label=label, zorder=3,
        )
        # Selective direct label at the line's end; floor datapoints flagged.
        end_note = " (floor)" if by_nodes[nodes[-1]]["score"] == 0.0 else ""
        ax.annotate(
            f"{label}{end_note}", (nodes[-1], elo[-1]), xytext=(8, 0),
            textcoords="offset points", fontsize=8, color=INK_SECONDARY, va="center",
        )

    ax.axhline(0, color=BASELINE, linewidth=1, linestyle="--", zorder=1)
    ax.annotate("parity with Pikafish@N", xy=(0.01, 0), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK_MUTED)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Pikafish node budget N (log scale)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Elo relative to Pikafish@N", color=INK_SECONDARY, fontsize=10)
    ax.set_title("ElephantFormer vs Pikafish node ladder", color=INK, fontsize=12, loc="left", pad=12)

    ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    if len(series) >= 2:
        legend = ax.legend(loc="upper right", fontsize=8, frameon=False)
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {out} ({len(series)} series, {len(records)} records).")


if __name__ == "__main__":
    main()
