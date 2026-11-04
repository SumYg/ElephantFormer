"""Pickle-free weight files for the board model: safe to share and to load.

A Lightning ``.ckpt`` is a pickle, so loading one runs whatever code is inside it:
Lightning's loader calls ``torch.load(weights_only=False)``, and its
``_instantiator`` hook is exploitable even with ``weights_only=True``
(CVE-2026-31221, CVE-2026-58659). Only load ``.ckpt`` files you made yourself.

For anything shared (published models, files from other machines) use the
``.safetensors`` export instead: raw tensors, plus the architecture config as
JSON in the file header. Loading one parses JSON and copies bytes; nothing is
executed.

Export a trained checkpoint (the default output sits next to it)::

    uv run python -m elephant_former.inference.board_weights \\
        --ckpt checkpoints/board-combined/<best>.ckpt [--out <name>.safetensors]

Every ``--model_path`` that :class:`~elephant_former.evaluation.board_match.ModelBot`
loads (``board_match``, ``ladder``) accepts either format. Paths ending in
``.safetensors`` go through :func:`load_weights`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from elephant_former.training.board_lightning_module import BoardLightningModule

FORMAT_NAME = "elephantformer-board"
FORMAT_VERSION = "1"
WEIGHTS_SUFFIX = ".safetensors"

# BoardTransformer constructor args: everything needed to rebuild the network.
ARCH_KEYS: Dict[str, type] = {
    "d_model": int,
    "nhead": int,
    "num_encoder_layers": int,
    "dim_feedforward": int,
    "dropout": float,
    "policy_head_dim": int,
    "num_value_classes": int,
}

_LAYER_KEY = re.compile(r"^encoder\.layers\.(\d+)\.")


def export_weights(module: BoardLightningModule, out_path: str | Path) -> Path:
    """Write ``module``'s network weights + architecture config as safetensors."""
    out_path = Path(out_path)
    config = {key: typ(module.hparams[key]) for key, typ in ARCH_KEYS.items()}
    state = {k: v.detach().cpu().contiguous() for k, v in module.model.state_dict().items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state,
        str(out_path),
        metadata={
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "config": json.dumps(config, sort_keys=True),
        },
    )
    return out_path


def export_checkpoint(ckpt_path: str | Path, out_path: Optional[str | Path] = None) -> Path:
    """Convert a (trusted) Lightning ``.ckpt`` into a shareable ``.safetensors`` file."""
    ckpt_path = Path(ckpt_path)
    if out_path is None:
        out_path = ckpt_path.with_suffix(WEIGHTS_SUFFIX)
    module = BoardLightningModule.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    return export_weights(module, out_path)


def _read_config(metadata: Optional[Dict[str, str]], path: Path) -> Dict[str, int | float]:
    """Validate the header metadata and return constructor kwargs."""
    if not metadata or metadata.get("format") != FORMAT_NAME:
        raise ValueError(f"{path}: not an ElephantFormer board weights file")
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{path}: unsupported format_version {metadata.get('format_version')!r} "
            f"(expected {FORMAT_VERSION!r})"
        )
    try:
        raw = json.loads(metadata["config"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: missing or malformed config") from exc
    if not isinstance(raw, dict) or set(raw) != set(ARCH_KEYS):
        raise ValueError(f"{path}: config must have exactly the keys {sorted(ARCH_KEYS)}")

    config: Dict[str, int | float] = {}
    for key, typ in ARCH_KEYS.items():
        value = raw[key]
        allowed = (int, float) if typ is float else (int,)
        if isinstance(value, bool) or not isinstance(value, allowed):
            raise ValueError(f"{path}: config[{key!r}] must be {typ.__name__}, got {value!r}")
        if (typ is int and value < 1) or (typ is float and not 0.0 <= value < 1.0):
            raise ValueError(f"{path}: config[{key!r}] out of range: {value!r}")
        config[key] = typ(value)
    return config


def load_weights(path: str | Path, device: str | torch.device = "cpu") -> BoardLightningModule:
    """Load an exported ``.safetensors`` file into a fresh module (eval mode).

    The file is treated as untrusted. The config must match the tensors in the
    file before any real memory is allocated, so a crafted header cannot make
    the loader build an oversized model.
    """
    path = Path(path)
    with safe_open(str(path), framework="pt") as f:
        config = _read_config(f.metadata(), path)
        file_shapes = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}

    # Layer count first: it bounds the cost of building even the meta probe.
    # Distinct indices with count n and max n-1 are exactly 0..n-1.
    layers_in_file = {int(m.group(1)) for k in file_shapes if (m := _LAYER_KEY.match(k))}
    n_layers = config["num_encoder_layers"]
    if len(layers_in_file) != n_layers or max(layers_in_file, default=-1) != n_layers - 1:
        raise ValueError(
            f"{path}: config says {n_layers} encoder layers, file has {len(layers_in_file)}"
        )

    # Build on the meta device (no storage) to get the expected tensor shapes.
    try:
        with torch.device("meta"):
            probe = BoardLightningModule(**config)
    except (AssertionError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{path}: config does not describe a valid model: {exc}") from exc
    expected = {k: tuple(v.shape) for k, v in probe.model.state_dict().items()}
    if file_shapes != expected:
        missing = sorted(set(expected) - set(file_shapes))
        unexpected = sorted(set(file_shapes) - set(expected))
        wrong = sorted(k for k in set(expected) & set(file_shapes) if expected[k] != file_shapes[k])
        raise ValueError(
            f"{path}: tensors do not match config (missing={missing[:3]}, "
            f"unexpected={unexpected[:3]}, wrong_shape={wrong[:3]})"
        )

    module = BoardLightningModule(**config)
    module.model.load_state_dict(load_file(str(path)), strict=True)
    return module.to(device).eval()


def load_board_module(path: str | Path, device: str | torch.device = "cpu") -> BoardLightningModule:
    """Load a board model from ``.safetensors`` (safe) or a Lightning ``.ckpt`` (trusted only)."""
    if str(path).endswith(WEIGHTS_SUFFIX):
        return load_weights(path, device=device)
    return BoardLightningModule.load_from_checkpoint(str(path), map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a board-model .ckpt to a pickle-free .safetensors file for sharing."
    )
    parser.add_argument("--ckpt", required=True, help="Lightning checkpoint you trust (it is unpickled).")
    parser.add_argument("--out", default=None, help="Output path (default: <ckpt>.safetensors).")
    args = parser.parse_args()

    out = export_checkpoint(args.ckpt, args.out)
    # Round-trip through the safe loader so a bad export fails here, not later.
    module = load_weights(out)
    n_params = sum(p.numel() for p in module.parameters())
    config = {k: module.hparams[k] for k in ARCH_KEYS}
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {n_params / 1e6:.2f}M params)")
    print(f"  config: {config}")


if __name__ == "__main__":
    main()
