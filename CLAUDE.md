# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ElephantFormer is a Transformer-based move prediction model for Elephant Chess (Chinese Chess). The project implements a GPT-style architecture that learns to predict chess moves as sequences of coordinate tokens. Each move is represented as a 4-tuple: `(from_x, from_y, to_x, to_y)` where coordinates are tokenized into a unified vocabulary.

## Development Commands

The project uses `uv` as the package manager. All commands should be run from the project root.

### Training
```bash
# Quick test training run with sample data
uv run python train.py \
    --pgn_file_path data/sample_games.pgn \
    --test_split_ratio 0.2 \
    --val_split_ratio 0.2 \
    --subset_ratio 0.001 \
    --output_split_dir data/splits \
    --max_epochs 1 \
    --d_model 32 \
    --nhead 2 \
    --num_encoder_layers 1 \
    --dim_feedforward 64 \
    --max_seq_len 32 \
    --accelerator cpu

# For full training options:
uv run python train.py --help
```

### Inference
```bash
# Run move generation with best trained model
uv run python -m elephant_former.inference.generator \
    --model_checkpoint_path checkpoints/trial-2-resume-1/elephant_former-epoch=22-val_loss=6.36.ckpt \
    --device cpu

# Run with custom model
uv run python -m elephant_former.inference.generator \
    --model_checkpoint_path checkpoints/your_trial_dir/your_model.ckpt \
    --device cuda
```

### Evaluation
```bash
# Calculate win rate
uv run python -m elephant_former.evaluation.evaluator \
    --model_path checkpoints/trial-2-resume-1/elephant_former-epoch=22-val_loss=6.36.ckpt \
    --pgn_file_path data/real/test_split.pgn \
    --device cpu \
    --metric win_rate \
    --num_win_rate_games 50 \
    --max_turns_win_rate 150

# Other metrics: accuracy, perplexity
uv run python -m elephant_former.evaluation.evaluator --help
```

### Examples and Utilities
```bash
# Parse games and display information
uv run python -m examples.parse_games

# Inspect sequence lengths in dataset
uv run python scripts/inspect_sequence_lengths.py
```

## Architecture Overview

### Core Components

1. **Data Pipeline** (`elephant_former/data/`)
   - `elephant_parser.py`: Parses PGN files into `ElephantGame` objects with ICCS move notation
   - `dataset.py`: PyTorch Dataset implementation with collate function for batched training
   - `tokenization_utils.py`: Converts ICCS moves to coordinate tuples and unified token IDs

2. **Model Architecture** (`elephant_former/models/`)
   - `transformer_model.py`: `ElephantFormerGPT` - GPT-style transformer with 4 output heads
   - Each head predicts one coordinate component: from_x, from_y, to_x, to_y
   - Uses learned positional embeddings and causal attention masking

3. **Training** (`elephant_former/training/`)
   - `lightning_module.py`: PyTorch Lightning wrapper with multi-head loss calculation
   - Combines CrossEntropyLoss from all 4 heads for coordinate prediction

4. **Game Engine** (`elephant_former/engine/`)
   - `elephant_chess_game.py`: Full Elephant Chess rules implementation
   - Handles legal move generation, game state tracking, draw conditions
   - Supports perpetual chase detection and repetition rules

5. **Inference** (`elephant_former/inference/`)
   - `generator.py`: Interactive move generation with legal move filtering
   - Scores all legal moves using model logits and selects highest-scoring valid move

6. **Evaluation** (`elephant_former/evaluation/`)
   - `evaluator.py`: Metrics calculation (accuracy, perplexity, win rate against random opponent)

### Key Design Patterns

- **Tokenization Strategy**: Moves are tokenized as sequences of 4 coordinate tokens with unified vocabulary
- **Multi-Head Output**: Model predicts all 4 move components simultaneously using separate classification heads
- **Legal Move Filtering**: During inference, only legal moves from the game engine are considered
- **Autoregressive Training**: Model learns to predict next move given sequence of previous moves

### Data Formats

- **Input**: PGN files with ICCS notation (e.g., "H2-E2", "C9-E7")
- **Tokenization**: Each move becomes 4 tokens from unified vocabulary (fx_0, fy_1, tx_2, ty_3, etc.)
- **Model Output**: 4 classification heads predicting coordinate values (0-8 for x, 0-9 for y)

### Directory Structure

- `data/`: Training datasets (PGN format) and splits
- `checkpoints/`: Saved model checkpoints from training trials
- `elephant_former/`: Main package with all core modules
- `tests/`: Test files for various components
- `demos/`: Demonstration scripts for specific features
- `examples/`: Usage examples and utilities

### Constants and Configuration

All vocabularies, token mappings, and board dimensions are defined in `elephant_former/constants.py`. The unified vocabulary includes special tokens (`<pad>`, `<start>`, `<unk>`) plus coordinate tokens for each move component.

## Testing

No formal test framework is configured. Test files exist in `tests/` directory for specific components like perpetual rules and move highlighting. Run individual test files directly:

```bash
uv run python tests/test_corrected_perpetual_rules.py
uv run python tests/test_move_highlighting.py
uv run python tests/test_perpetual_implementation.py
```