"""Train the board-state ElephantFormer (Phase 0).

Mirrors ``train.py`` conventions (train/val/test splits, ModelCheckpoint on
``val_loss``, EarlyStopping, CLI) but uses the encoder-only board model, the
from->to policy head, and the value head.

The expensive per-position featurisation is cached to disk (see
``BoardChessDataset``); the full dataset is built once from ``--pgn_file_path``
and then split with a fixed seed. The default ``--split_by game`` keeps every
game wholly inside one split (no same-game leakage between train and val/test);
``--split_by position`` reproduces the legacy leaky split for resuming runs
started before game-level splitting existed.

Phase 1 distillation: ``--engine_annotations PGN=NPZ`` mixes engine-best-move
policy targets (from :mod:`pikafish_annotator` label files) into the train set,
restricted to train-split games; ``--engine_repeat`` oversamples them to tune
the human:engine ratio. Val/test stay purely human-labeled.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

from elephant_former.data.elephant_parser import parse_iccs_pgn_file
from elephant_former.data_utils.board_dataset import (
    BoardChessDataset,
    board_collate_fn,
    split_indices_by_game,
)
from elephant_former.data_utils.engine_labels import EngineLabeledDataset, load_annotations
from elephant_former.training.board_lightning_module import BoardLightningModule


def _parse_engine_annotations(args: argparse.Namespace) -> dict:
    """``--engine_annotations PGN=NPZ`` pairs -> {pgn_path: npz_path}."""
    if not args.engine_annotations:
        return {}
    pairs = {}
    for spec in args.engine_annotations:
        pgn, sep, npz = spec.partition("=")
        if not sep:
            raise SystemExit(f"--engine_annotations expects PGN=NPZ, got: {spec}")
        if pgn not in args.pgn_file_path:
            raise SystemExit(f"--engine_annotations: {pgn} is not among --pgn_file_path")
        pairs[pgn] = npz
    if args.split_by != "game":
        raise SystemExit("--engine_annotations requires --split_by game (train-only leakage rule).")
    if 0.0 < args.subset_ratio < 1.0:
        raise SystemExit("--engine_annotations is incompatible with --subset_ratio (row alignment).")
    return pairs


def _make_loader(dataset, args, shuffle: bool):
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=board_collate_fn,
        shuffle=shuffle,
        num_workers=args.num_workers,
    )


def _resolve_resume(args: argparse.Namespace):
    """Returns (resume_ckpt_path or None, previous run manifest or None)."""
    if not args.resume_from:
        return None, None
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt = ckpt_dir / "last.ckpt" if args.resume_from == "auto" else Path(args.resume_from)
    if not ckpt.exists():
        raise SystemExit(f"--resume_from: checkpoint not found: {ckpt}")
    manifest_path = ckpt_dir / "board_training_args.json"
    manifest = None
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    print(f"Resuming from checkpoint: {ckpt}")
    return ckpt, manifest


def main(args: argparse.Namespace) -> None:
    random.seed(args.split_seed)
    engine_pairs = _parse_engine_annotations(args)
    resume_ckpt, prev_manifest = _resolve_resume(args)

    use_subset = 0.0 < args.subset_ratio < 1.0
    if use_subset:
        all_games = []
        for p in args.pgn_file_path:
            games_p = parse_iccs_pgn_file(p)
            print(f"Loaded {len(games_p)} games from {p}.")
            all_games.extend(games_p)
        random.shuffle(all_games)
        all_games = all_games[: int(len(all_games) * args.subset_ratio)]
        print(f"Using a subset of {len(all_games)} games ({args.subset_ratio * 100:.2f}%). Caching disabled for subsets.")
        dataset = BoardChessDataset(games=all_games, use_cache=False)
        parts = [dataset]
    else:
        # One (cached) dataset per source PGN, concatenated. Content-keyed caches
        # mean each part loads without a rebuild wherever it was built.
        parts = [
            BoardChessDataset(
                pgn_file_path=p,
                cache_dir=args.cache_dir,
                use_cache=not args.no_cache,
            )
            for p in args.pgn_file_path
        ]
        dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)

    if len(dataset) == 0:
        print("Dataset has 0 positions (all games rejected or too short). Exiting.")
        return

    n = len(dataset)
    if args.split_by == "game":
        # Whole games per split; ids are offset per part so they stay unique
        # across concatenated PGN sources.
        id_parts, offset = [], 0
        for part in parts:
            ids = part.game_ids()
            id_parts.append(ids + offset)
            if len(ids):
                offset += int(ids[-1]) + 1
        game_ids = np.concatenate(id_parts) if id_parts else np.zeros((0,), dtype=np.int64)
        train_idx, val_idx, test_idx = split_indices_by_game(
            game_ids, args.test_split_ratio, args.val_split_ratio, args.split_seed
        )
        val_ds = Subset(dataset, val_idx)
        test_ds = Subset(dataset, test_idx)
        num_games = int(game_ids[-1]) + 1 if len(game_ids) else 0
        print(
            f"Game-level split of {num_games} games / {n} positions -> "
            f"train {len(train_idx)}, val {len(val_ds)}, test {len(test_ds)}."
        )

        # Engine-labeled rows join the train mix only, and only for rows whose
        # game the split assigned to train (an engine label for a val/test-game
        # position would leak that position into training).
        train_sets = [Subset(dataset, train_idx)]
        if engine_pairs:
            row_offsets = np.cumsum([0] + [len(p) for p in parts])
            train_mask = np.zeros(n, dtype=bool)
            train_mask[train_idx] = True
            for pgn, npz in engine_pairs.items():
                pi = args.pgn_file_path.index(pgn)
                part = parts[pi]
                local_train_rows = np.flatnonzero(
                    train_mask[row_offsets[pi] : row_offsets[pi + 1]]
                )
                annotations = load_annotations(npz, expected_rows=len(part))
                engine_ds = EngineLabeledDataset(part, annotations, restrict_rows=local_train_rows)
                print(
                    f"Engine labels {npz}: {len(engine_ds)} train rows "
                    f"x{args.engine_repeat} (from {pgn})."
                )
                train_sets.extend([engine_ds] * args.engine_repeat)
        train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
        if engine_pairs:
            print(f"Train mix: {len(train_ds)} examples ({len(train_idx)} human-labeled).")
    else:
        n_test = int(n * args.test_split_ratio)
        n_val = int((n - n_test) * args.val_split_ratio)
        n_train = n - n_test - n_val
        generator = torch.Generator().manual_seed(args.split_seed)
        train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=generator)
        print(f"Position-level split of {n} positions -> train {n_train}, val {n_val}, test {n_test}.")

    # Resuming with different data or split parameters silently shifts the
    # train/val/test membership (optimizer state would meet leaked val data).
    if resume_ckpt is not None and prev_manifest is not None:
        prev_pgns = prev_manifest.get("pgn_file_path")
        if isinstance(prev_pgns, str):  # manifests from the single-file era
            prev_pgns = [prev_pgns]
        same_data = (
            prev_pgns == args.pgn_file_path
            and prev_manifest.get("split_seed") == args.split_seed
            and prev_manifest.get("test_split_ratio") == args.test_split_ratio
            and prev_manifest.get("val_split_ratio") == args.val_split_ratio
            # Manifests predating game-level splits used the position scheme.
            and prev_manifest.get("split_by", "position") == args.split_by
            and prev_manifest.get("engine_annotations") == args.engine_annotations
            and prev_manifest.get("engine_repeat", 1) == args.engine_repeat
            and prev_manifest.get("n_positions") in (None, n)
        )
        if not same_data and not args.allow_dataset_change:
            raise SystemExit(
                "Resume refused: dataset/split differs from the original run "
                f"(was pgn={prev_manifest.get('pgn_file_path')}, "
                f"n={prev_manifest.get('n_positions')}, seed={prev_manifest.get('split_seed')}, "
                f"split_by={prev_manifest.get('split_by', 'position')}; "
                f"now pgn={args.pgn_file_path}, n={n}, seed={args.split_seed}, "
                f"split_by={args.split_by}). "
                "Resuming a pre-game-split run? Pass --split_by position. "
                "Pass --allow_dataset_change to fine-tune on new data deliberately."
            )

    if args.checkpoint_dir:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        manifest = dict(vars(args), n_positions=n)
        with open(Path(args.checkpoint_dir) / "board_training_args.json", "w") as f:
            json.dump(manifest, f, indent=4, default=str)

    train_loader = _make_loader(train_ds, args, shuffle=True)
    val_loader = _make_loader(val_ds, args, shuffle=False)
    test_loader = _make_loader(test_ds, args, shuffle=False)

    model = BoardLightningModule(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        policy_head_dim=args.policy_head_dim,
        value_loss_weight=args.value_loss_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    callbacks = []
    step_checkpointing = args.save_every_n_steps > 0
    if step_checkpointing:
        # Mid-epoch snapshots so an interrupted (e.g. spot/rented) run loses at
        # most save_every_n_steps of work; also maintains last.ckpt.
        callbacks.append(
            ModelCheckpoint(
                dirpath=args.checkpoint_dir,
                filename="board_former-step-{step}",
                every_n_train_steps=args.save_every_n_steps,
                save_top_k=1,
                save_last=True,
            )
        )
    if val_loader is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="board_former-{epoch:02d}-{val_loss:.3f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=not step_checkpointing,
        )
        callbacks.append(checkpoint_callback)
        if args.early_stopping_patience > 0:
            callbacks.append(
                EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, mode="min", verbose=True)
            )
    elif step_checkpointing:
        checkpoint_callback = callbacks[0]
    else:
        # No validation set: still checkpoint the last model so training yields an artifact.
        checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir, filename="board_former-last", save_last=True)
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=callbacks,
    )

    print("Starting training...")
    ckpt_path = str(resume_ckpt) if resume_ckpt is not None else None
    if val_loader is not None:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, train_loader, ckpt_path=ckpt_path)
    print("Training finished.")

    if test_loader is not None:
        ckpt = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path or None
        trainer.test(model, dataloaders=test_loader, ckpt_path=ckpt if ckpt else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the board-state ElephantFormer.")
    # Data
    parser.add_argument(
        "--pgn_file_path", type=str, nargs="+", required=True,
        help="One or more PGN files; each gets its own cache and they are concatenated.",
    )
    parser.add_argument("--val_split_ratio", type=float, default=0.1, help="Fraction of non-test positions used for validation.")
    parser.add_argument("--test_split_ratio", type=float, default=0.1, help="Fraction of all positions used for the test set.")
    parser.add_argument("--subset_ratio", type=float, default=1.0, help="Fraction of games to use (disables caching when < 1).")
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--no_cache", action="store_true", help="Disable the on-disk position cache.")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--split_by", type=str, default="game", choices=["game", "position"],
        help="'game' (default) keeps whole games in one split; 'position' is the "
        "legacy leaky split, needed to resume runs started before game-level splits.",
    )
    parser.add_argument(
        "--engine_annotations", type=str, nargs="+", default=None,
        help="Engine-label files as PGN=NPZ pairs (e.g. data/WXF-41743games.pgns="
        "data/annotations/WXF-pikafish-n10k-full.npz). Adds engine-best-move policy "
        "targets for train-split games only. Requires --split_by game.",
    )
    parser.add_argument(
        "--engine_repeat", type=int, default=1,
        help="Times each engine-labeled set is repeated in the train mix "
        "(oversampling knob for the human:engine ratio).",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    # Model
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--policy_head_dim", type=int, default=256)
    parser.add_argument("--value_loss_weight", type=float, default=0.5)
    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--accelerator", type=str, default="auto", choices=["cpu", "gpu", "tpu", "mps", "auto"])
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/board")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    # Resume / portability
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Checkpoint to resume from (path), or 'auto' for <checkpoint_dir>/last.ckpt.",
    )
    parser.add_argument(
        "--save_every_n_steps", type=int, default=2000,
        help="Also checkpoint mid-epoch every N train steps (0 disables).",
    )
    parser.add_argument(
        "--allow_dataset_change", action="store_true",
        help="Permit resuming with a different dataset/split (deliberate fine-tuning).",
    )

    main(parser.parse_args())
