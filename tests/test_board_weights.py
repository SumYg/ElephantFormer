"""Safetensors weight export/load: round-trip fidelity and rejection of bad files.

Run: ``uv run python tests/test_board_weights.py`` (exits non-zero on failure).
"""

import json
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from safetensors.torch import save_file

from elephant_former.data_utils import board_features as bf
from elephant_former.engine.elephant_chess_game import ElephantChessGame
from elephant_former.evaluation.board_match import ModelBot
from elephant_former.inference.board_weights import (
    FORMAT_NAME,
    FORMAT_VERSION,
    export_checkpoint,
    export_weights,
    load_board_module,
    load_weights,
)
from elephant_former.training.board_lightning_module import BoardLightningModule

TINY = dict(d_model=32, nhead=2, num_encoder_layers=2, dim_feedforward=64, policy_head_dim=16)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _tiny_module(seed: int = 0) -> BoardLightningModule:
    torch.manual_seed(seed)
    return BoardLightningModule(**TINY).eval()


def _random_batch(batch: int = 4):
    g = torch.Generator().manual_seed(1)
    piece_ids = torch.randint(0, bf.NUM_PIECE_CLASSES, (batch, bf.NUM_SQUARES), generator=g)
    flags = torch.randint(0, 2, (batch, bf.NUM_SQUARES, bf.NUM_FLAGS), generator=g)
    stm = torch.randint(0, 2, (batch,), generator=g)
    return piece_ids, flags, stm


def _expect_rejected(path: Path, needle: str) -> None:
    try:
        load_weights(path)
    except ValueError as exc:
        _check(needle in str(exc), f"expected error mentioning {needle!r}, got: {exc}")
        return
    raise AssertionError(f"{path.name} should have been rejected ({needle})")


def _write_with_config(module: BoardLightningModule, path: Path, config, **meta) -> None:
    """Write real tensors with a hand-made header, as an attacker could."""
    state = {k: v.contiguous() for k, v in module.model.state_dict().items()}
    metadata = {"format": FORMAT_NAME, "format_version": FORMAT_VERSION, "config": json.dumps(config)}
    metadata.update(meta)
    save_file(state, str(path), metadata=metadata)


def _good_config() -> dict:
    return {**TINY, "dropout": 0.1, "num_value_classes": 3}


def test_round_trip_matches_outputs(tmp: Path) -> None:
    module = _tiny_module()
    path = export_weights(module, tmp / "w.safetensors")
    loaded = load_weights(path)

    _check(not loaded.training, "loaded module should be in eval mode")
    for key in TINY:
        _check(loaded.hparams[key] == module.hparams[key], f"hparam {key} not restored")
    batch = _random_batch()
    with torch.no_grad():
        p0, v0 = module(*batch)
        p1, v1 = loaded(*batch)
    _check(torch.equal(p0, p1) and torch.equal(v0, v1), "outputs differ after round-trip")
    print("  [ok] round-trip reproduces policy + value outputs exactly")


def test_file_is_plain_safetensors(tmp: Path) -> None:
    path = export_weights(_tiny_module(), tmp / "w.safetensors")
    raw = path.read_bytes()
    (header_len,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_len])
    meta = header.pop("__metadata__")
    _check(meta["format"] == FORMAT_NAME, "format tag missing from header")
    _check(set(json.loads(meta["config"])) >= set(TINY), "config missing from header")
    # Everything after the header is the tensors' raw bytes, nothing else.
    data_len = max(t["data_offsets"][1] for t in header.values())
    _check(len(raw) == 8 + header_len + data_len, "unexpected bytes beyond tensor data")
    print("  [ok] export is a JSON header + raw tensor bytes (no pickle)")


def test_checkpoint_export_and_bot_loading(tmp: Path) -> None:
    import pytorch_lightning as pl

    module = _tiny_module(seed=3)
    trainer = pl.Trainer(
        accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.strategy.connect(module)
    ckpt = tmp / "model.ckpt"
    trainer.save_checkpoint(str(ckpt))

    out = export_checkpoint(ckpt)
    _check(out == tmp / "model.safetensors", f"default output path wrong: {out}")
    from_ckpt = load_board_module(ckpt).eval()
    from_safe = load_board_module(out)
    batch = _random_batch()
    with torch.no_grad():
        _check(torch.equal(from_ckpt(*batch)[0], from_safe(*batch)[0]), "ckpt and export disagree")

    bot = ModelBot(model_path=str(out), device="cpu")
    game = ElephantChessGame()
    move = bot.select_move(game)
    _check(move in game.get_all_legal_moves_basic(game.current_player), "bot played an illegal move")
    print("  [ok] .ckpt -> .safetensors export matches, and ModelBot loads the export")


def test_rejects_foreign_or_malformed_headers(tmp: Path) -> None:
    module = _tiny_module()
    state = {k: v.contiguous() for k, v in module.model.state_dict().items()}

    save_file(state, str(tmp / "nometa.safetensors"))
    _expect_rejected(tmp / "nometa.safetensors", "not an ElephantFormer")

    _write_with_config(module, tmp / "v2.safetensors", _good_config(), format_version="2")
    _expect_rejected(tmp / "v2.safetensors", "format_version")

    save_file(state, str(tmp / "badjson.safetensors"),
              metadata={"format": FORMAT_NAME, "format_version": FORMAT_VERSION, "config": "{nope"})
    _expect_rejected(tmp / "badjson.safetensors", "malformed config")

    extra = {**_good_config(), "_instantiator": "os.system"}
    _write_with_config(module, tmp / "extra.safetensors", extra)
    _expect_rejected(tmp / "extra.safetensors", "exactly the keys")

    wrong_type = {**_good_config(), "d_model": "32"}
    _write_with_config(module, tmp / "type.safetensors", wrong_type)
    _expect_rejected(tmp / "type.safetensors", "must be int")

    negative = {**_good_config(), "nhead": -2}
    _write_with_config(module, tmp / "neg.safetensors", negative)
    _expect_rejected(tmp / "neg.safetensors", "out of range")
    print("  [ok] foreign, versioned, malformed and extra-key headers rejected")


def test_rejects_config_tensor_mismatch(tmp: Path) -> None:
    module = _tiny_module()

    # Huge declared width: must fail on the meta probe, before allocating.
    huge = {**_good_config(), "d_model": 1 << 20, "nhead": 1}
    _write_with_config(module, tmp / "huge.safetensors", huge)
    _expect_rejected(tmp / "huge.safetensors", "do not match config")

    many_layers = {**_good_config(), "num_encoder_layers": 10_000_000}
    _write_with_config(module, tmp / "layers.safetensors", many_layers)
    _expect_rejected(tmp / "layers.safetensors", "encoder layers")

    indivisible = {**_good_config(), "nhead": 3}
    _write_with_config(module, tmp / "heads.safetensors", indivisible)
    _expect_rejected(tmp / "heads.safetensors", "valid model")

    state = {k: v.contiguous() for k, v in module.model.state_dict().items()}
    state.pop("policy_q.weight")
    save_file(state, str(tmp / "missing.safetensors"),
              metadata={"format": FORMAT_NAME, "format_version": FORMAT_VERSION,
                        "config": json.dumps(_good_config())})
    _expect_rejected(tmp / "missing.safetensors", "do not match config")
    print("  [ok] config/tensor mismatches rejected before building the model")


if __name__ == "__main__":
    tests = [
        test_round_trip_matches_outputs,
        test_file_is_plain_safetensors,
        test_checkpoint_export_and_bot_loading,
        test_rejects_foreign_or_malformed_headers,
        test_rejects_config_tensor_mismatch,
    ]
    print("Running board weight export/load tests:")
    with tempfile.TemporaryDirectory() as d:
        for test in tests:
            test(Path(d))
    print("All board weight tests passed.")
