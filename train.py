import argparse
from pathlib import Path
import random # For shuffling games before split
import json # For saving arguments

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split # random_split might also be an option for datasets
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping # Added callbacks

from elephant_former.data.elephant_parser import parse_iccs_pgn_file, save_games_to_pgn_file, ElephantGame
from elephant_former.data_utils.dataset import ElephantChessDataset, elephant_collate_fn
from elephant_former.training.lightning_module import LightningElephantFormer
from elephant_former import constants
from typing import List # For type hinting

def main(args):
    # 0. Load all games from the main PGN file
    print(f"Loading all games from: {args.pgn_file_path}")
    all_games: List[ElephantGame] = parse_iccs_pgn_file(args.pgn_file_path)
    if not all_games:
        print(f"No games found in {args.pgn_file_path}. Exiting.")
        return
    print(f"Loaded {len(all_games)} games in total.")

    # Apply subset_ratio if specified
    if 0.0 < args.subset_ratio < 1.0:
        random.shuffle(all_games) # Shuffle before taking a subset
        subset_size = int(len(all_games) * args.subset_ratio)
        all_games = all_games[:subset_size]
        print(f"Using a subset of {subset_size} games ({args.subset_ratio*100:.2f}% of total) for training/validation/testing.")
    elif args.subset_ratio < 0.0 or args.subset_ratio > 1.0:
        print(f"Warning: subset_ratio ({args.subset_ratio}) is outside the valid range (0.0, 1.0]. Using all data.")

    # Save training arguments
    if args.checkpoint_dir:
        output_dir = Path(args.checkpoint_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        args_save_path = output_dir / "training_args.json"
        try:
            with open(args_save_path, 'w') as f:
                json.dump(vars(args), f, indent=4)
            print(f"Saved training arguments to: {args_save_path}")
        except Exception as e:
            print(f"Error saving training arguments: {e}")

    train_games: List[ElephantGame]
    val_games: List[ElephantGame] = []
    test_games: List[ElephantGame] = []

    remaining_games = all_games
    if args.test_split_ratio > 0 and args.test_split_ratio < 1:
        random.shuffle(remaining_games) # Shuffle before splitting
        test_split_idx = int(len(remaining_games) * args.test_split_ratio)
        test_games = remaining_games[:test_split_idx]
        remaining_games = remaining_games[test_split_idx:]
        print(f"Split off {len(test_games)} test games. {len(remaining_games)} games remaining for train/validation.")
        if args.output_split_dir and test_games:
            output_dir = Path(args.output_split_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            test_split_path = output_dir / "test_split.pgn"
            print(f"Saving test split to: {test_split_path}")
            save_games_to_pgn_file(test_games, test_split_path)
    else:
        print("No test set split requested or ratio is invalid.")

    if args.val_split_ratio > 0 and args.val_split_ratio < 1 and remaining_games:
        random.shuffle(remaining_games) # Shuffle the remainder before splitting into train/val
        val_split_idx = int(len(remaining_games) * args.val_split_ratio) # val_split_ratio is applied to remaining games
        val_games = remaining_games[:val_split_idx]
        train_games = remaining_games[val_split_idx:]
        print(f"Splitting remaining data: {len(train_games)} training games, {len(val_games)} validation games.")
    else:
        train_games = remaining_games
        print("Using all remaining games for training. No validation split from remaining, or invalid ratio.")

    if args.output_split_dir and train_games:
        output_dir = Path(args.output_split_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_split_path = output_dir / "train_split.pgn"
        print(f"Saving training split to: {train_split_path}")
        save_games_to_pgn_file(train_games, train_split_path)
    
    if args.output_split_dir and val_games:
        output_dir = Path(args.output_split_dir)
        # Directory should already be created by train or test split save
        val_split_path = output_dir / "val_split.pgn"
        print(f"Saving validation split to: {val_split_path}")
        save_games_to_pgn_file(val_games, val_split_path)

    # 1. Setup Data
    # Modify ElephantChessDataset to accept List[ElephantGame] directly
    train_dataset = ElephantChessDataset(games=train_games, max_seq_len=args.max_seq_len)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=elephant_collate_fn,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_dataloader = None
    if val_games:
        val_dataset = ElephantChessDataset(games=val_games, max_seq_len=args.max_seq_len)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=elephant_collate_fn,
            shuffle=False,
            num_workers=args.num_workers
        )
        print(f"Train Dataloader: {len(train_dataloader)} batches. Validation Dataloader: {len(val_dataloader)} batches.")
    else:
        print(f"Train Dataloader: {len(train_dataloader)} batches. No validation Dataloader.")

    test_dataloader = None
    if test_games:
        test_dataset = ElephantChessDataset(games=test_games, max_seq_len=args.max_seq_len)
        if len(test_dataset) > 0:
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                collate_fn=elephant_collate_fn,
                shuffle=False,
                num_workers=args.num_workers
            )
            print(f"Test Dataloader: {len(test_dataloader)} batches.")
        else:
            print("Test dataset was empty after processing, no test dataloader created.")
    else:
        print("No test games, no test dataloader created.")

    # 2. Setup Model
    model = LightningElephantFormer(
        vocab_size=constants.UNIFIED_VOCAB_SIZE,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 3. Setup Trainer
    callbacks = []
    if val_dataloader: # Only add ModelCheckpoint if there is a validation set to monitor
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="elephant_former-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3, # Save top 3 models
            monitor="val_loss",
            mode="min",
            save_last=True # Also save the last model checkpoint
        )
        callbacks.append(checkpoint_callback)

    if args.early_stopping_patience > 0 and val_dataloader:
        early_stop_callback = EarlyStopping(
           monitor="val_loss",
           patience=args.early_stopping_patience,
           verbose=True,
           mode="min"
        )
        callbacks.append(early_stop_callback)
        print(f"Early stopping enabled with patience {args.early_stopping_patience} monitoring val_loss.")
    else:
        print("Early stopping not enabled (patience <=0 or no validation loader).")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=callbacks, # Add callbacks here
        # logger=TensorBoardLogger("tb_logs", name="elephant_former"), # Example logger
        # progress_bar_refresh_rate=20, # For older PL versions
        # val_check_interval=0.25, # Validate every 25% of an epoch
    )

    # 4. Train
    print("Starting training...")
    ckpt_path_to_resume = None
    if args.resume_from_checkpoint and Path(args.resume_from_checkpoint).exists():
        ckpt_path_to_resume = args.resume_from_checkpoint
        print(f"Resuming training from checkpoint: {ckpt_path_to_resume}")
    elif args.resume_from_checkpoint:
        print(f"WARNING: Specified checkpoint {args.resume_from_checkpoint} not found. Starting training from scratch.")

    if val_dataloader:
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path_to_resume)
    else:
        trainer.fit(model, train_dataloader, ckpt_path=ckpt_path_to_resume)
    print("Training finished.")

    # 5. Test
    if test_dataloader:
        print("Starting testing...")
        ckpt_path_to_test = None
        if 'checkpoint_callback' in locals() and hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path and Path(checkpoint_callback.best_model_path).exists():
            print(f"Found best model path: {checkpoint_callback.best_model_path}")
            ckpt_path_to_test = checkpoint_callback.best_model_path
        elif 'checkpoint_callback' in locals() and hasattr(checkpoint_callback, 'last_model_path') and checkpoint_callback.last_model_path and Path(checkpoint_callback.last_model_path).exists():
            print(f"Best model path not found or invalid. Using last model path: {checkpoint_callback.last_model_path}")
            ckpt_path_to_test = checkpoint_callback.last_model_path
        
        if ckpt_path_to_test:
            print(f"Loading model from checkpoint for testing: {ckpt_path_to_test}")
            trainer.test(model, dataloaders=test_dataloader, ckpt_path=ckpt_path_to_test)
        else:
            print("No suitable checkpoint found (best or last). Testing with the current model state from training.")
            trainer.test(model, dataloaders=test_dataloader) # Test with the model in its current state (after fit)
        print("Testing finished.")
    else:
        print("No test dataloader available, skipping testing phase.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ElephantFormer model.")

    # Data args
    parser.add_argument("--pgn_file_path", type=str, required=True, help="Path to the PGN file for training.")
    parser.add_argument("--val_split_ratio", type=float, default=0.1, help="Ratio of NON-TEST data to use for validation (e.g., 0.1 for 10% of non-test data).")
    parser.add_argument("--test_split_ratio", type=float, default=0.1, help="Ratio of total data to use for the test set (e.g., 0.1 for 10% of total). Set to 0 to disable test set.")
    parser.add_argument("--subset_ratio", type=float, default=1.0, help="Ratio of the total dataset to use (e.g., 0.1 for 10%). Applied after loading all games, before splitting. Default 1.0 (use all data).")
    parser.add_argument("--output_split_dir", type=str, default=None, help="Directory to save train/validation/test split PGN files. If None, splits are not saved.")
    # parser.add_argument("--val_pgn_file_path", type=str, default=None, help="Path to the PGN file for validation.")
    # parser.add_argument("--test_pgn_file_path", type=str, default=None, help="Path to the PGN file for testing.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader.")

    # Model HParams (from LightningElephantFormer)
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length.") # Also used by Dataset
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension.")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of encoder layers.")
    parser.add_argument("--dim_feedforward", type=int, default=1024, help="Dimension of feedforward network.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")

    # Training HParams (from LightningElephantFormer & Trainer)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of training epochs.")
    parser.add_argument("--accelerator", type=str, default="auto", choices=["cpu", "gpu", "tpu", "mps", "auto"], help="Accelerator to use.")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use (e.g., GPUs).")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--early_stopping_patience", type=int, default=2, help="Number of epochs with no improvement after which training will be stopped. Set to 0 to disable.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint file to resume training from (e.g., checkpoints/last.ckpt).")
    
    args = parser.parse_args()
    main(args) 