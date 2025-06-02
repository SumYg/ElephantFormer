# ElephantFormer
A Transformer-based move prediction model for Elephant Chess.

## Getting Started

### Running the Example Parser
To try out the ICCS format parser with the example dataset:

```bash
uv run python -m examples.parse_games
```

This will parse the sample games file and display information about the first game, including:
- Player names
- Game result
- Number of moves
- First 10 moves in ICCS format

# Plan
Here's my plan **without an `<end>` token**, focusing on GPT-style modeling of Chinese chess moves as `(from_x, from_y, to_x, to_y)` token sequences:

---

## Step 1: Define Vocabulary & Tokenization

*   **Move Representation**: Each move is a tuple of 4 components: `(from_x, from_y, to_x, to_y)`.
*   **Component Vocabularies**:
    *   `from_x`: 9 tokens (0–8)
    *   `from_y`: 10 tokens (0–9)
    *   `to_x`: 9 tokens (0–8)
    *   `to_y`: 10 tokens (0–9)
*   **Input Tokenization**:
    *   We need a unified vocabulary for the input to the Transformer's embedding layer. This could be achieved by:
        1.  **Concatenated Sequence of Individual Component Tokens**: Each move `(fx, fy, tx, ty)` is represented as 4 consecutive tokens in the input sequence. The total vocabulary for the embedding layer will include all component tokens plus special tokens.
        2.  **Combined Move Tokens (Alternative, more complex)**: Create a unique ID for each possible 4-tuple move. This would lead to a very large vocabulary. (Sticking to 1 for now as per current plan)
      *   Strategy 1. **Concatenated Sequence of Individual Component Tokens** is chosen for its small vocabulary.
*   Special token: `<start>` for generation start.
*   Input sequence example (using concatenated component tokens): `<start> fx₁ fy₁ tx₁ ty₁ fx₂ fy₂ tx₂ ty₂ ...`

---

## Step 2: Dataset Preparation

*   Collect full games. Each move is represented as a 4-tuple: `(fx, fy, tx, ty)`.
*   **Input sequences**: A sequence of previous moves, where each move is 4 tokens.
    *   Example: `[<start>, fx₁, fy₁, tx₁, ty₁, fx₂, fy₂, tx₂, ty₂]`
*   **Target sequences**: For each input sequence, the target is the next move, also as a 4-tuple of token IDs `(next_fx, next_fy, next_tx, next_ty)`.
    *   The model will predict these 4 components *simultaneously* for a given timestep.
*   Pad sequences if batching.

---

## Step 3: Model Architecture

*   Use a GPT-style transformer model with:
    *   **Token Embedding Layer**: Input tokens (from the unified vocabulary if using concatenated component tokens) are embedded.
    *   Positional Embeddings.
    *   Several Transformer Blocks.
    *   **Output Layer**: Four separate classification heads, one for each component of the move:
        *   Head_from_x: Projects to 9 logits (for `from_x` tokens).
        *   Head_from_y: Projects to 10 logits (for `from_y` tokens).
        *   Head_to_x: Projects to 9 logits (for `to_x` tokens).
        *   Head_to_y: Projects to 10 logits (for `to_y` tokens).
        *   Each head will have a softmax activation.
*   Model input shape: `(batch_size, seq_len)` where `seq_len` is the number of individual tokens (e.g., if 10 moves, `seq_len = 1 + 10*4 = 41` which means using `<start>` and concatenated tokens).
*   Model output shape (at each relevant prediction step): A tuple of 4 logit tensors:
    *   `(batch_size, num_from_x_classes)`
    *   `(batch_size, num_from_y_classes)`
    *   `(batch_size, num_to_x_classes)`
    *   `(batch_size, num_to_y_classes)`
    *   (Alternatively, if predicting at every sequence position: `(batch_size, seq_len, num_classes)` for each head, but the primary interest is the prediction at the end of the input sequence to generate the next move.)

---

## Step 4: Training Procedure

*   Train autoregressively to predict the 4 components of the *next full move* given the sequence of previous moves (as individual tokens).
*   **Loss Function**: Use four separate CrossEntropyLoss functions, one for each output head.
    *   `loss = loss_fx + loss_fy + loss_tx + loss_ty` (or an average).
*   Use teacher forcing: feed the true previous move (as 4 tokens) to predict the current move's 4 components.
*   Batch training with padding and attention masks. The attention mask ensures the model only attends to previous tokens.

---

## Step 5: Move Generation (Inference)

*   Start with an input sequence (e.g., `[<start>]` or `[<start>, fx₁, fy₁, tx₁, ty₁, ..., fxₖ, fyₖ, txₖ, tyₖ]`).
*   At each generation step:
    1.  **Get Model Output**: Feed the current sequence into the model. It outputs four sets of logits: `L_fx`, `L_fy`, `L_tx`, `L_ty` (one for each component of the potential next move).
    2.  **Get All Legal Moves**: Consult the game engine to get a list of all valid 4-tuple moves `(legal_fx, legal_fy, legal_tx, legal_ty)` from the current board state.
    3.  **Score Legal Moves**: For each legal move `(m_fx, m_fy, m_tx, m_ty)` from the list:
        *   Retrieve the corresponding logits from the model's output (e.g., `L_fx[m_fx_token_id]`, etc.).
        *   Calculate a score for this move. This is typically the sum of the log-probabilities (or just sum of logits if only taking the max). For example: `score = log_softmax(L_fx)[m_fx] + log_softmax(L_fy)[m_fy] + log_softmax(L_tx)[m_tx] + log_softmax(L_ty)[m_ty]`.
    4.  **Select Best Legal Move**: Choose the legal move with the highest score. This is the predicted move `(selected_fx, selected_fy, selected_tx, selected_ty)`.
    5.  The 4 token IDs corresponding to this selected move form the next part of the sequence.
*   Append the 4 predicted token IDs (`selected_fx_id`, `selected_fy_id`, `selected_tx_id`, `selected_ty_id`) to the input sequence.
*   Apply the `(selected_fx, selected_fy, selected_tx, selected_ty)` move to the game engine to update the board state.
*   Check if the game has ended (win, loss, draw).
*   If not ended, use the new, extended sequence as input for the next prediction step.
*   Stop when the game engine signals termination or a maximum number of moves is reached.

---

## Step 6: Evaluation Metrics

*   **Prediction Accuracy:** All 4 components of the move `(fx, fy, tx, ty)` are correctly predicted for a given step.
*   **Perplexity:** Can be calculated for each head or as a combined measure.
*   **Win Rate:** Assess performance by playing against existing Chinese chess agents.

---

## Step 7: Optional - Add Board State Conditioning (Advanced)

* Encode current board as additional input tokens or features
* Concatenate or combine with move tokens embeddings
* Model predicts moves conditioned on actual board state, improving accuracy

---

## Summary Checklist

| Step                                     | Status        |
| ---------------------------------------- | ------------- |
| Vocabulary & Tokenization                | Done          |
| Dataset Preparation                      | Done          |
| GPT Model Architecture                   | Done          |
| Training Loop                            | Done          |
| Move Generation Logic (incl. Game Logic) | In Progress   |
| Evaluation Metrics                       | To Do         |
| Optional Board Conditioning              | To Do         |


## Current Progress (October 2023 - Evolving)

*   **Project Setup**: Initialized project with `uv` and basic structure.
*   **Constants**: Defined board dimensions, special tokens (`<pad>`, `<start>`, `<unk>`), unified input vocabulary (41 tokens: `fx_0..8`, `fy_0..9`, etc.), and output class counts in `elephant_former/constants.py`.
*   **PGN Parser (`elephant_former/data/elephant_parser.py`)**: 
    *   `ElephantGame` dataclass stores metadata and `iccs_moves_string`.
    *   `parsed_moves` property on `ElephantGame` lazily parses the `iccs_moves_string` into a list of individual move strings (e.g., `["H2-E2", "C9-E7"]`).
    *   `parse_iccs_pgn_file` parses PGN files into a list of `ElephantGame` objects.
    *   `games_to_pgn_string` and `save_games_to_pgn_file` utilities for writing games back to PGN format.
    *   Tested with a `__main__` block including save and re-parse validation.
*   **Tokenization Utilities (`elephant_former/data_utils/tokenization_utils.py`)**:
    *   `parse_iccs_move_to_coords`: Converts ICCS move string (e.g., "A0-B0") to `(fx, fy, tx, ty)` integer tuple.
    *   `coords_to_unified_token_ids`: Converts `(fx, fy, tx, ty)` tuple to a list of 4 unified input token IDs.
    *   `generate_training_sequences_from_game`: Takes an `ElephantGame` (using its `parsed_moves` property) and produces `(input_token_ids_sequence, target_coordinate_tuple)` pairs for training.
*   **PyTorch Dataset & DataLoader (`elephant_former/data_utils/dataset.py`)**:
    *   `ElephantChessDataset` (PyTorch `Dataset`):
        *   Accepts either `file_paths` to PGNs or a list of pre-loaded `ElephantGame` objects.
        *   Uses the parser and tokenization utils to generate all training instances.
        *   Implements `__len__` and `__getitem__`.
        *   Includes `min_game_len_moves` to filter short games.
    *   `elephant_collate_fn`:
        *   Pads input sequences in a batch to the same length using `PAD_TOKEN_ID`.
        *   Formats targets into four separate tensors for the four output heads.
*   **Model Architecture (`elephant_former/models/transformer_model.py`)**:
    *   `ElephantFormerGPT` (`nn.Module`):
        *   Token embedding, learned positional embedding.
        *   `nn.TransformerEncoder` (using `nn.TransformerEncoderLayer`, `batch_first=True`).
        *   Four linear output heads (for fx, fy, tx, ty).
        *   `forward` method takes `src`, `src_mask` (causal), `src_padding_mask` (boolean) and returns 4 logit tensors.
    *   `generate_square_subsequent_mask` helper for causal masking.
*   **PyTorch Lightning Module (`elephant_former/training/lightning_module.py`)**:
    *   `LightningElephantFormer` (`pl.LightningModule`):
        *   Initializes `ElephantFormerGPT` model and four `nn.CrossEntropyLoss` functions (corrected to not use `ignore_index`).
        *   `_calculate_loss` helper for summing losses from the four heads, using logits from the last non-padded input token.
        *   `training_step`, `validation_step`, `test_step` defined for loss calculation and logging.
        *   `configure_optimizers` (currently `AdamW`).
*   **Training Script (`train.py`)**:
    *   Uses PyTorch Lightning `Trainer`.
    *   Command-line arguments for hyperparameters (data paths, model dims, training params, etc.).
    *   **Data Handling**:
        *   Loads all games from a PGN file.
        *   Splits data into **train, validation, and test sets** based on specified ratios (`--test_split_ratio`, `--val_split_ratio`).
        *   Optionally saves these three data splits to new PGN files in `--output_split_dir`.
        *   Creates `DataLoader`s for train, validation, and test sets using `ElephantChessDataset`.
    *   **Callbacks**:
        *   `ModelCheckpoint`: Saves model checkpoints based on `val_loss` (best and last), configured via `--checkpoint_dir`.
        *   `EarlyStopping`: Stops training if `val_loss` doesn't improve for a given patience (`--early_stopping_patience`).
    *   **Workflow**: `trainer.fit()` for training and validation, followed by `trainer.test()` on the test set (loading the best checkpoint).
*   **Design Notes (`design_notes.md`)**: Notes on board state conditioning importance.
*   **Dependencies**: `pytorch-lightning`, `numpy`.

**Next Steps (High-Level from Plan):**
*   Focus on **Step 5: Move Generation/Inference**.
*   Implement Evaluation Metrics (Step 6).
*   Consider Board State Conditioning (Step 7).


