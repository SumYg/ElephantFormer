"""One-shot status report for detached training/build/annotation jobs.

Usage:
    uv run python scripts/training_status.py            # scan logs/ + checkpoints/
    uv run python scripts/training_status.py --log logs/train_combined.log

Reads the tqdm-style logs the detached jobs write (progress uses carriage
returns, so plain `tail` shows a wall of text — this parses it properly),
reports how recently each log was written (the liveness signal), the newest
checkpoints, and GPU utilisation via nvidia-smi.
"""

import argparse
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

EPOCH_RE = re.compile(
    r"Epoch (\d+):\s+(\d+)%.*?(\d+)/(\d+)(?: \[([0-9:]+)<([0-9:?]+))?"
)


def last_progress_line(log_path: Path) -> str | None:
    """Last carriage-return-delimited progress line mentioning an epoch or chunk."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = [seg for chunk in text.splitlines() for seg in chunk.split("\r") if seg.strip()]
    for line in reversed(lines):
        if "Epoch" in line or "games]" in line or "pos/s" in line or "finished" in line:
            return line.strip()
    return lines[-1].strip() if lines else None


def age_str(path: Path) -> str:
    secs = time.time() - path.stat().st_mtime
    if secs < 120:
        return f"{secs:.0f}s ago (ACTIVE)"
    if secs < 7200:
        return f"{secs / 60:.0f} min ago" + (" (likely stopped)" if secs > 300 else "")
    return f"{secs / 3600:.1f} h ago (stopped)"


def gpu_line() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return out or "n/a"
    except Exception:
        return "nvidia-smi not available"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default=None, help="Report on one specific log file.")
    args = parser.parse_args()

    logs = [Path(args.log)] if args.log else sorted(
        (REPO / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True
    )

    print(f"GPU: {gpu_line()}\n")
    for log in logs:
        if not log.exists():
            print(f"{log}: not found")
            continue
        line = last_progress_line(log) or "(empty)"
        if len(line) > 140:
            line = line[:140] + "…"
        print(f"[{log.name}]  written {age_str(log)}")
        print(f"  {line}\n")

    for ckpt_dir in sorted((REPO / "checkpoints").glob("board*")):
        ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if ckpts:
            newest = ckpts[0]
            print(f"[{ckpt_dir.name}]  newest checkpoint: {newest.name}  ({age_str(newest)})")

    ann_dir = REPO / "data" / "annotations"
    if ann_dir.exists():
        for f in sorted(ann_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)[:3]:
            print(f"[annotations]  {f.name}  ({age_str(f)})")


if __name__ == "__main__":
    main()
