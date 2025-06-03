# elephant_former/inference/generator.py

import torch
import torch.nn.functional as F # For log_softmax
import random
import argparse # For command-line arguments
from typing import List, Tuple, Optional
from pathlib import Path

from elephant_former.engine.elephant_chess_game import ElephantChessGame, Move, RED, BLACK, PIECE_NAMES
from elephant_former.training.lightning_module import LightningElephantFormer # To load the model
from elephant_former.models.transformer_model import generate_square_subsequent_mask # Import for causal mask
from elephant_former.data_utils.tokenization_utils import coords_to_unified_token_ids
from elephant_former import constants

class MoveGenerator:
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.game = ElephantChessGame()
        self.model: Optional[LightningElephantFormer] = None
        self.device = torch.device(device)
        self.max_seq_len = None # Will be set from loaded model

        if model_path:
            if not Path(model_path).exists():
                print(f"Error: Model checkpoint not found at {model_path}")
                self.model = None
            else:
                try:
                    print(f"Loading model from checkpoint: {model_path}")
                    self.model = LightningElephantFormer.load_from_checkpoint(
                        checkpoint_path=model_path, 
                        map_location=self.device
                    )
                    self.model.to(self.device)
                    self.model.eval()
                    self.max_seq_len = self.model.hparams.max_seq_len
                    print(f"MoveGenerator initialized with model. Max sequence length: {self.max_seq_len}")
                except Exception as e:
                    print(f"Error loading model from {model_path}: {e}")
                    self.model = None
        
        if not self.model:
            print("MoveGenerator initialized without a pre-trained model (random moves will be made).")
            # Set a default max_seq_len if no model is loaded, though it won't be used for model prediction
            self.max_seq_len = 512 # Or some other reasonable default or from constants


    def reset_game(self, fen: Optional[str] = None):
        """Resets the game to the initial state or a given FEN."""
        self.game = ElephantChessGame(fen=fen)
        print("Game reset.")

    def get_model_predicted_move(self, legal_moves: List[Move]) -> Optional[Move]:
        if not legal_moves:
            return None

        if not self.model or not self.max_seq_len:
            print("No model loaded or max_seq_len not set, selecting random move.")
            return random.choice(legal_moves)

        # 1. Prepare input sequence
        unpadded_sequence = [constants.START_TOKEN_ID]
        for move in self.game.move_history:
            fx, fy, tx, ty = move
            token_ids = coords_to_unified_token_ids((fx, fy, tx, ty))
            unpadded_sequence.extend(token_ids)
        
        # Handle sequence length (truncate if too long, keep start token and recent history)
        if len(unpadded_sequence) > self.max_seq_len:
            # Keep START_TOKEN and the last (max_seq_len - 1) tokens
            unpadded_sequence = [constants.START_TOKEN_ID] + unpadded_sequence[-(self.max_seq_len - 1):]
        
        current_seq_len = len(unpadded_sequence)
        padding_needed = self.max_seq_len - current_seq_len
        
        final_sequence = unpadded_sequence + [constants.PAD_TOKEN_ID] * padding_needed
        input_tensor = torch.tensor([final_sequence], dtype=torch.long, device=self.device)

        # 2. Create masks
        # Causal mask
        causal_mask = generate_square_subsequent_mask(self.max_seq_len, device=self.device)
        
        # Padding mask: True where padded
        padding_mask_tensor = (input_tensor == constants.PAD_TOKEN_ID)
        if torch.all(~padding_mask_tensor): # if no padding tokens found
            padding_mask_tensor = None # Pass None if no padding, as per TransformerEncoder docs

        # 3. Model prediction
        with torch.no_grad():
            logits_fx, logits_fy, logits_tx, logits_ty = self.model(input_tensor, src_mask=causal_mask, src_padding_mask=padding_mask_tensor)
        
        # We need logits for the token *after* the last actual input token.
        # The model outputs (batch, seq_len, num_classes). `current_seq_len` is the length of our actual input.
        # So we are interested in the prediction at index `current_seq_len - 1` in the output sequence.
        last_actual_token_idx = current_seq_len - 1 
        if last_actual_token_idx < 0: # Should not happen if there's at least a START_TOKEN
            print("Error: last_actual_token_idx is negative. Defaulting to random move.")
            return random.choice(legal_moves)

        pred_logits_fx = logits_fx[0, last_actual_token_idx, :] 
        pred_logits_fy = logits_fy[0, last_actual_token_idx, :] 
        pred_logits_tx = logits_tx[0, last_actual_token_idx, :] 
        pred_logits_ty = logits_ty[0, last_actual_token_idx, :] 

        # 4. Score legal moves
        best_move = None
        max_score = -float('inf')

        for move_coords in legal_moves:
            m_fx, m_fy, m_tx, m_ty = move_coords
            
            # Ensure indices are within bounds for the specific head's logits
            if not (0 <= m_fx < pred_logits_fx.size(0) and \
                    0 <= m_fy < pred_logits_fy.size(0) and \
                    0 <= m_tx < pred_logits_tx.size(0) and \
                    0 <= m_ty < pred_logits_ty.size(0)):
                # This should not happen if constants.NUM_..._CLASSES are correct and moves are valid
                # print(f"Warning: Move component out of bounds for logits: {move_coords}")
                continue

            score_fx = F.log_softmax(pred_logits_fx, dim=-1)[m_fx]
            score_fy = F.log_softmax(pred_logits_fy, dim=-1)[m_fy]
            score_tx = F.log_softmax(pred_logits_tx, dim=-1)[m_tx]
            score_ty = F.log_softmax(pred_logits_ty, dim=-1)[m_ty]
            
            current_score = score_fx + score_fy + score_tx + score_ty
            
            if current_score.item() > max_score:
                max_score = current_score.item()
                best_move = move_coords
        
        if best_move is None and legal_moves: # If all scores were -inf or no valid scores
            print("Warning: Could not determine best move from model, picking random among legal.")
            return random.choice(legal_moves)
            
        return best_move

    def play_a_turn(self) -> Optional[str]: # Returns game_status or None
        print(self.game)
        player = self.game.get_current_player()
        player_name = "RED" if player == RED else "BLACK"
        print(f"Current player: {player_name} (Move {self.game.fullmove_number}.{self.game.halfmove_clock % 2 + 1})")

        game_status = self.game.check_game_over()
        if game_status:
            print(f"Game Over! Result: {game_status}")
            return game_status

        legal_moves = self.game.get_all_legal_moves(player)
        if not legal_moves:
            final_status = self.game.check_game_over() 
            print(f"No legal moves for {player_name}. Final status check: {final_status}")
            return final_status if final_status else "STUCK_NO_LEGAL_MOVES_UNEXPECTED"

        # print(f"Legal moves for {player_name}: {len(legal_moves)}")
        selected_move = self.get_model_predicted_move(legal_moves)

        if selected_move:
            fx, fy, tx, ty = selected_move
            piece_val = self.game.get_piece_at(fx, fy)
            piece_name = PIECE_NAMES.get(piece_val, "UnknownPiece")
            print(f"{player_name} selects: {piece_name} from ({fx},{fy}) to ({tx},{ty})")
            self.game.apply_move(selected_move)
        else:
            print(f"Error: No move could be selected for {player_name}. This might indicate an issue.")
            # This could happen if legal_moves was empty but somehow passed the earlier check,
            # or if get_model_predicted_move returned None despite having legal_moves.
            return "ERROR_NO_MOVE_SELECTED"
        
        return self.game.check_game_over()

    def run_game_loop(self, max_turns=100, fen: Optional[str] = None):
        """Runs a game loop for a maximum number of turns or until game over."""
        self.reset_game(fen=fen)
        for i in range(max_turns):
            print(f"\n--- Turn {i+1} ---")
            status = self.play_a_turn()
            if status:
                print(f"\nGame finished after {i+1} turns. Final status: {status}")
                print("Final Board:")
                print(self.game)
                return status
        print(f"\nGame stopped after {max_turns} turns (max_turns reached).")
        print("Final Board:")
        print(self.game)
        return "MAX_TURNS_REACHED"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ElephantFormer Move Generator")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model checkpoint (.ckpt)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device to run inference on (cpu, cuda, mps)")
    parser.add_argument("--max_turns", type=int, default=50, help="Maximum number of turns to play in the game loop.")
    parser.add_argument("--fen", type=str, default=None, help="FEN string to start the game from a custom position.")
    
    cli_args = parser.parse_args()

    if cli_args.model_path:
        print(f"Running game with model: {cli_args.model_path}")
        generator = MoveGenerator(model_path=cli_args.model_path, device=cli_args.device)
    else:
        print("Running game with random moves (no model specified).")
        generator = MoveGenerator(device=cli_args.device)
    
    generator.run_game_loop(max_turns=cli_args.max_turns, fen=cli_args.fen) 