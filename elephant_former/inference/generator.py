# elephant_former/inference/generator.py

import torch
import torch.nn.functional as F # For log_softmax
import random
import argparse # For command-line arguments
from typing import List, Tuple, Optional
from pathlib import Path

from elephant_former.engine.elephant_chess_game import ElephantChessGame, Move, Player, PIECE_NAMES
from elephant_former.training.lightning_module import LightningElephantFormer # To load the model
from elephant_former.models.transformer_model import generate_square_subsequent_mask # Import for causal mask
from elephant_former.data_utils.tokenization_utils import coords_to_unified_token_ids
from elephant_former.constants import START_TOKEN_ID, PAD_TOKEN_ID # Corrected path

class MoveGenerator:
    def __init__(self, model_checkpoint_path: str, device: str = 'cpu', initial_fen: Optional[str] = None):
        self.game = ElephantChessGame(fen=initial_fen)
        self.model_checkpoint_path = model_checkpoint_path
        self.device = torch.device(device)
        self.current_fen = initial_fen
        self.model: Optional[LightningElephantFormer] = None
        self.max_seq_len = 0 # Initialize, will be set from loaded model
        self.current_game_token_history: List[int] = [START_TOKEN_ID]

        if model_checkpoint_path:
            if not Path(model_checkpoint_path).exists():
                print(f"Error: Model checkpoint not found at {model_checkpoint_path}")
                self.model = None
            else:
                try:
                    print(f"Loading model from checkpoint: {model_checkpoint_path}")
                    self.model = LightningElephantFormer.load_from_checkpoint(
                        checkpoint_path=model_checkpoint_path, 
                        map_location=self.device
                    )
                    self.model.to(self.device)
                    self.model.eval()
                    self.max_seq_len = self.model.hparams.max_seq_len
                    print(f"MoveGenerator initialized with model. Max sequence length: {self.max_seq_len}")
                except Exception as e:
                    print(f"Error loading model from {model_checkpoint_path}: {e}")
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
        unpadded_sequence = [START_TOKEN_ID]
        for move in self.game.move_history:
            fx, fy, tx, ty = move
            token_ids = coords_to_unified_token_ids((fx, fy, tx, ty))
            unpadded_sequence.extend(token_ids)
        
        # Handle sequence length (truncate if too long, keep start token and recent history)
        if len(unpadded_sequence) > self.max_seq_len:
            # Keep START_TOKEN and the last (max_seq_len - 1) tokens
            unpadded_sequence = [START_TOKEN_ID] + unpadded_sequence[-(self.max_seq_len - 1):]
        
        current_seq_len = len(unpadded_sequence)
        padding_needed = self.max_seq_len - current_seq_len
        
        final_sequence = unpadded_sequence + [PAD_TOKEN_ID] * padding_needed
        input_tensor = torch.tensor([final_sequence], dtype=torch.long, device=self.device)

        # 2. Create masks
        # Causal mask
        causal_mask = generate_square_subsequent_mask(self.max_seq_len, device=self.device)
        
        # Padding mask: True where padded
        padding_mask_tensor = (input_tensor == PAD_TOKEN_ID)
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

    def play_a_turn(self) -> Tuple[Optional[str], Optional[Player]]:
        """
        Plays a single turn of the game.
        Returns a tuple (game_over_status, winner).
        """
        current_player_enum = self.game.get_current_player()
        player_name = current_player_enum.name

        print(f"\n--- {player_name}'s turn ({self.game.fullmove_number}{'.' if player_name == Player.RED.name else '...'}) ---")

        predicted_move_coords = self.get_model_predicted_move(self.game.get_all_legal_moves(current_player_enum))

        if predicted_move_coords:
            fx, fy, tx, ty = predicted_move_coords
            print(f"{player_name} (Model) plays: ({fx},{fy}) -> ({tx},{ty})")
            self.game.apply_move(predicted_move_coords)
            
            move_token_ids = coords_to_unified_token_ids(predicted_move_coords)
            self.current_game_token_history.extend(move_token_ids)
            if len(self.current_game_token_history) > self.max_seq_len:
                self.current_game_token_history = [START_TOKEN_ID] + \
                                                 self.current_game_token_history[-(self.max_seq_len-1):]

            print(self.game.__str__(last_move=predicted_move_coords)) # Pass the move to __str__
            return self.game.check_game_over()
        else:
            print(f"{player_name} (Model) has no legal moves.")
            # Check if it's checkmate or stalemate
            if self.game.is_king_in_check(current_player_enum):
                opponent = self.game.get_opponent(current_player_enum)
                print(f"Checkmate! {opponent.name} wins.")
                return "checkmate", opponent
            else:
                print("Stalemate! It's a draw.")
                return "stalemate", None

    def run_game_loop(self, max_turns: int = 100):
        """Runs the main game loop until game over or max_turns is reached."""
        print("Starting new game...")
        print(f"Model: {self.model_checkpoint_path}")
        print(f"Device: {self.device}")
        print(f"Max sequence length: {self.max_seq_len}")
        print(f"Initial FEN: {self.current_fen if self.current_fen else 'Default starting position'}")
        print(f"Max turns: {max_turns}")
        print("Initial board state:")
        print(self.game) # Print initial board

        for turn_count in range(max_turns):
            game_over_status, winner = self.play_a_turn()

            input("Press Enter to continue...")

            if game_over_status:
                print(f"\n--- Game Over --- ({game_over_status})")
                if winner:
                    print(f"Winner: {winner.name}")
                else:
                    print("Result: Draw")
                break
            
            if turn_count == max_turns - 1:
                print("\n--- Game Over ---")
                print(f"Maximum turns ({max_turns}) reached. Game is a draw.")
                break

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser(description="Play a game using a trained ElephantFormer model.")
    cli_parser.add_argument("--model_checkpoint_path", type=str, required=True, help="Path to the model checkpoint (.ckpt)")
    cli_parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device to use (cpu, cuda, mps)")
    cli_parser.add_argument("--max_turns", type=int, default=100, help="Maximum number of turns for the game.")
    cli_parser.add_argument("--fen", type=str, default=None, help="Initial FEN string to start the game from. Uses default start if not provided.")
    
    cli_args = cli_parser.parse_args()

    try:
        generator = MoveGenerator(
            model_checkpoint_path=cli_args.model_checkpoint_path, 
            device=cli_args.device, 
            initial_fen=cli_args.fen # FEN is passed here
        )
        # The FEN is handled by the constructor. If cli_args.fen is None, MoveGenerator uses default.
        # If a FEN is provided, it's used to initialize self.game.
        # So, run_game_loop doesn't need it again.
        generator.run_game_loop(max_turns=cli_args.max_turns) # Removed fen=cli_args.fen
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # raise # Uncomment for full traceback during development 