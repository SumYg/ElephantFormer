# elephant_former/evaluation/evaluator.py

import torch
import argparse
import torch.nn.functional as F
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import random # Added for random opponent

from elephant_former.training.lightning_module import LightningElephantFormer
from elephant_former.data.elephant_parser import ElephantGame, parse_iccs_pgn_file
from elephant_former.data_utils.tokenization_utils import generate_training_sequences_from_game, coords_to_unified_token_ids, parse_iccs_move_to_coords # Added parse_iccs_move_to_coords
from elephant_former.engine.elephant_chess_game import ElephantChessGame, Player # Added ElephantChessGame, Player
from elephant_former import constants
from elephant_former.models.transformer_model import generate_square_subsequent_mask

class ModelEvaluator:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model: LightningElephantFormer
        self.device = torch.device(device)
        self.max_seq_len = None # Will be set from loaded model

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        
        try:
            print(f"Loading model from checkpoint: {model_path}")
            self.model = LightningElephantFormer.load_from_checkpoint(
                checkpoint_path=model_path, 
                map_location=self.device
            )
            self.model.to(self.device)
            self.model.eval()
            self.max_seq_len = self.model.hparams.max_seq_len # type: ignore
            print(f"ModelEvaluator initialized. Max sequence length: {self.max_seq_len}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            raise

    def calculate_prediction_accuracy(self, pgn_file_path: str) -> Dict[str, float]:
        """Calculates the prediction accuracy of the model on a given PGN file.

        Accuracy is defined as the percentage of times the model correctly predicts 
        all four components (fx, fy, tx, ty) of the next move.
        """
        if not Path(pgn_file_path).exists():
            raise FileNotFoundError(f"PGN file not found at {pgn_file_path}")

        games: List[ElephantGame] = parse_iccs_pgn_file(pgn_file_path)
        if not games:
            print(f"No games found or parsed from {pgn_file_path}")
            return {
                "total_predictions": 0.0,
                "correct_predictions": 0.0,
                "accuracy": 0.0
            }

        total_predictions = 0
        correct_predictions = 0

        for i, game in enumerate(games):
            print(f"Processing game {i+1}/{len(games)}: {game.metadata.get('Event', 'Unknown Event')}")
            training_instances = generate_training_sequences_from_game(game)
            
            for input_token_ids, target_coords in training_instances:
                if not input_token_ids or not target_coords:
                    continue # Should not happen with valid game data

                # Prepare input tensor
                current_seq_len = len(input_token_ids)
                if current_seq_len == 0: continue # Should not happen if START_TOKEN is always there

                if current_seq_len > self.max_seq_len:
                    # Truncate if too long, keeping the most recent history
                    input_token_ids_final = input_token_ids[-(self.max_seq_len):]
                    current_seq_len = self.max_seq_len
                else:
                    input_token_ids_final = input_token_ids
                
                padding_needed = self.max_seq_len - current_seq_len
                padded_sequence = input_token_ids_final + [constants.PAD_TOKEN_ID] * padding_needed
                
                input_tensor = torch.tensor([padded_sequence], dtype=torch.long, device=self.device)

                # Create masks
                causal_mask = generate_square_subsequent_mask(self.max_seq_len, device=self.device)
                padding_mask_tensor = (input_tensor == constants.PAD_TOKEN_ID)
                if torch.all(~padding_mask_tensor):
                    padding_mask_tensor = None
                
                # Model prediction
                with torch.no_grad():
                    logits_fx, logits_fy, logits_tx, logits_ty = self.model(input_tensor, src_mask=causal_mask, src_padding_mask=padding_mask_tensor)
                
                # Get logits for the prediction at the end of the actual input sequence
                # The model outputs (batch, seq_len, num_classes).
                # current_seq_len is the length of our actual input before padding.
                # So we are interested in the prediction at index `current_seq_len - 1`.
                last_actual_token_idx = current_seq_len - 1
                if last_actual_token_idx < 0: continue

                pred_logits_fx = logits_fx[0, last_actual_token_idx, :]
                pred_logits_fy = logits_fy[0, last_actual_token_idx, :]
                pred_logits_tx = logits_tx[0, last_actual_token_idx, :]
                pred_logits_ty = logits_ty[0, last_actual_token_idx, :]

                # Get predicted components
                predicted_fx = torch.argmax(pred_logits_fx).item()
                predicted_fy = torch.argmax(pred_logits_fy).item()
                predicted_tx = torch.argmax(pred_logits_tx).item()
                predicted_ty = torch.argmax(pred_logits_ty).item()

                # Compare with target
                target_fx, target_fy, target_tx, target_ty = target_coords
                
                total_predictions += 1
                if (predicted_fx == target_fx and 
                    predicted_fy == target_fy and 
                    predicted_tx == target_tx and 
                    predicted_ty == target_ty):
                    correct_predictions += 1
        
        accuracy = (correct_predictions / total_predictions) if total_predictions > 0 else 0.0
        
        results = {
            "total_predictions": float(total_predictions),
            "correct_predictions": float(correct_predictions),
            "accuracy": accuracy * 100.0 # As percentage
        }
        print(f"Accuracy Calculation Results: {results}")
        return results

    def calculate_perplexity(self, pgn_file_path: str) -> float:
        """
        Calculates the perplexity of the model on a PGN file.
        Perplexity = exp(average_cross_entropy_loss).
        Lower is better.
        """
        self.model.eval()
        all_losses = []

        games = parse_iccs_pgn_file(pgn_file_path)
        if not games:
            print(f"No games found in {pgn_file_path}")
            return float('inf')

        print(f"Calculating perplexity for {len(games)} games from {pgn_file_path}...")
        for game in tqdm(games, desc="Evaluating Games for Perplexity"):
            # The sequences from generate_training_sequences_from_game can be up to
            # 1 (start_token) + MAX_GAME_HISTORY_MOVES * 4 tokens long.
            # Subsequent logic in this method handles truncation to self.max_seq_len.
            training_instances = generate_training_sequences_from_game(
                game
            )
            for input_token_ids, target_coords in training_instances:
                if not input_token_ids or not target_coords: # Basic check
                    continue

                current_seq_len = len(input_token_ids)
                if current_seq_len == 0: continue 

                # Explicit truncation and padding to self.max_seq_len, similar to accuracy method
                if current_seq_len > self.max_seq_len:
                    input_token_ids_final = input_token_ids[-(self.max_seq_len):]
                    current_seq_len = self.max_seq_len # This is the length of actual tokens we feed, pre-padding for model
                else:
                    input_token_ids_final = input_token_ids
                
                padding_needed = self.max_seq_len - len(input_token_ids_final) # Recalculate padding for potentially truncated
                padded_sequence = input_token_ids_final + [constants.PAD_TOKEN_ID] * padding_needed
                
                input_tensor = torch.tensor([padded_sequence], dtype=torch.long, device=self.device)

                # Create masks for the fixed length self.max_seq_len
                causal_mask = generate_square_subsequent_mask(self.max_seq_len, device=self.device)
                padding_mask_tensor = (input_tensor == constants.PAD_TOKEN_ID)
                # TransformerEncoder expects None if no padding, not an all-False mask.
                # However, if there is padding, padding_mask_tensor will correctly indicate it.
                # If there's no padding, padding_mask_tensor will be all False.
                # Let's ensure it's None if effectively no padding tokens.
                # (Though Lightning module might handle this internally for nn.TransformerEncoder)
                # For consistency with accuracy method:
                if torch.all(~padding_mask_tensor): # if no True values (no PAD tokens)
                    actual_padding_mask = None
                else:
                    actual_padding_mask = padding_mask_tensor

                with torch.no_grad():
                    logits_fx, logits_fy, logits_tx, logits_ty = self.model(
                        input_tensor, src_mask=causal_mask, src_padding_mask=actual_padding_mask
                    )
                
                # Logits are for the prediction *after* the last actual input token.
                # current_seq_len here is the length of input_token_ids_final (actual tokens)
                last_actual_token_idx = len(input_token_ids_final) - 1
                
                if last_actual_token_idx < 0: 
                    continue

                # Ensure that last_actual_token_idx does not exceed logits sequence dimension (self.max_seq_len -1)
                if last_actual_token_idx >= logits_fx.size(1): # logits_fx is (1, self.max_seq_len, num_classes)
                    # This should not happen if last_actual_token_idx is len(input_token_ids_final)-1 and input_token_ids_final is <= self.max_seq_len
                    # print(f"Warning: last_actual_token_idx {last_actual_token_idx} exceeds logits seq dim {logits_fx.size(1)}. Skipping.")
                    continue

                pred_logits_fx_at_last_step = logits_fx[0, last_actual_token_idx, :]
                pred_logits_fy_at_last_step = logits_fy[0, last_actual_token_idx, :]
                pred_logits_tx_at_last_step = logits_tx[0, last_actual_token_idx, :]
                pred_logits_ty_at_last_step = logits_ty[0, last_actual_token_idx, :]

                target_fx, target_fy, target_tx, target_ty = target_coords
                
                target_fx_tensor = torch.tensor([target_fx], device=self.device, dtype=torch.long)
                target_fy_tensor = torch.tensor([target_fy], device=self.device, dtype=torch.long)
                target_tx_tensor = torch.tensor([target_tx], device=self.device, dtype=torch.long)
                target_ty_tensor = torch.tensor([target_ty], device=self.device, dtype=torch.long)

                loss_fx = F.cross_entropy(pred_logits_fx_at_last_step.unsqueeze(0), target_fx_tensor)
                loss_fy = F.cross_entropy(pred_logits_fy_at_last_step.unsqueeze(0), target_fy_tensor)
                loss_tx = F.cross_entropy(pred_logits_tx_at_last_step.unsqueeze(0), target_tx_tensor)
                loss_ty = F.cross_entropy(pred_logits_ty_at_last_step.unsqueeze(0), target_ty_tensor)
                
                current_total_loss = loss_fx + loss_fy + loss_tx + loss_ty
                all_losses.append(current_total_loss.item())
        
        if not all_losses:
            print("No losses were calculated. Cannot compute perplexity.")
            return float('inf')

        mean_loss = sum(all_losses) / len(all_losses)
        perplexity = math.exp(mean_loss)
        return perplexity

    def _get_random_move(self, game: ElephantChessGame) -> Tuple[int, int, int, int] | None:
        """Helper function to get a random legal move for the current player."""
        legal_moves = game.get_all_legal_moves(game.current_player)
        if not legal_moves:
            return None
        return random.choice(legal_moves)

    def _get_model_move(self, game: ElephantChessGame, current_game_token_history: List[int]) -> Tuple[int, int, int, int] | None:
        """
        Uses the loaded model to predict the best legal move.
        (Adapted from inference/generator.py)
        """
        self.model.eval()
        legal_moves = game.get_all_legal_moves(game.current_player)
        if not legal_moves:
            return None

        # Prepare input sequence for the model
        # Sequence is <start> t1 t2 t3 t4 ... tk
        # We need to ensure it's truncated/padded to self.max_seq_len
        
        input_token_ids = current_game_token_history[:self.max_seq_len] # Truncate if longer
        
        current_input_len = len(input_token_ids)
        padding_needed = self.max_seq_len - current_input_len
        padded_input_token_ids = input_token_ids + [constants.PAD_TOKEN_ID] * padding_needed
        
        src = torch.tensor([padded_input_token_ids], dtype=torch.long, device=self.device)
        
        # Create masks
        causal_mask = generate_square_subsequent_mask(self.max_seq_len, device=self.device)
        
        src_padding_mask = (src == constants.PAD_TOKEN_ID)
        if torch.all(~src_padding_mask): 
            actual_padding_mask = None
        else:
            actual_padding_mask = src_padding_mask

        with torch.no_grad():
            logits_fx, logits_fy, logits_tx, logits_ty = self.model(
                src, src_mask=causal_mask, src_padding_mask=actual_padding_mask
            )

        idx_for_prediction = current_input_len - 1
        if idx_for_prediction < 0: 
            return self._get_random_move(game) 

        if idx_for_prediction >= logits_fx.size(1):
            print(f"Warning: idx_for_prediction {idx_for_prediction} out of bounds for logits size {logits_fx.size(1)}. Defaulting to random.")
            return self._get_random_move(game)


        log_probs_fx = F.log_softmax(logits_fx[0, idx_for_prediction, :], dim=-1)
        log_probs_fy = F.log_softmax(logits_fy[0, idx_for_prediction, :], dim=-1)
        log_probs_tx = F.log_softmax(logits_tx[0, idx_for_prediction, :], dim=-1)
        log_probs_ty = F.log_softmax(logits_ty[0, idx_for_prediction, :], dim=-1)

        best_score = -float('inf')
        best_move = None

        for move_coords in legal_moves:
            fx, fy, tx, ty = move_coords
            score = log_probs_fx[fx] + log_probs_fy[fy] + log_probs_tx[tx] + log_probs_ty[ty]
            
            if score > best_score:
                best_score = score
                best_move = move_coords
        
        if best_move is None and legal_moves: 
            return self._get_random_move(game)
            
        return best_move

    def calculate_win_rate(self, num_games: int = 10, max_turns_per_game: int = 200, ai_plays_red: bool = True) -> Dict[str, Any]:
        """
        Calculates win rate by simulating games against a random opponent.
        The AI will play as Red by default (player making the first move).
        """
        print(f"Calculating win rate over {num_games} games against a random opponent...")
        print(f"AI plays as Red: {ai_plays_red}, Max turns per game: {max_turns_per_game}")

        wins = 0
        losses = 0
        draws = 0
        
        for i in tqdm(range(num_games), desc="Simulating Games for Win Rate"):
            game = ElephantChessGame() # Resets for each game
            current_game_token_history = [constants.START_TOKEN_ID]
            
            # Determine who is AI based on ai_plays_red for this specific game
            # Note: ElephantChessGame starts with Player.RED
            model_player_this_game = Player.RED if ai_plays_red else Player.BLACK
            # opponent_player_this_game = Player.BLACK if ai_plays_red else Player.RED

            game_over_status = None
            winner = None

            for turn_num in range(max_turns_per_game):
                player_to_move = game.current_player
                move_to_apply = None

                if player_to_move == model_player_this_game:
                    move_to_apply = self._get_model_move(game, current_game_token_history)
                else: # Opponent's turn
                    move_to_apply = self._get_random_move(game)

                if move_to_apply is None: # No legal moves available for the current player
                    if game.is_king_in_check(player_to_move):
                        game_over_status = "checkmate"
                        winner = game.get_opponent(player_to_move)
                    else:
                        game_over_status = "stalemate"
                    break 

                game.apply_move(move_to_apply)
                
                move_token_ids = coords_to_unified_token_ids(move_to_apply)
                current_game_token_history.extend(move_token_ids)
                
                # Optimized history truncation: keep only up to max_seq_len tokens for the model, always keeping START_TOKEN
                if len(current_game_token_history) > self.max_seq_len:
                     current_game_token_history = [constants.START_TOKEN_ID] + \
                                                 current_game_token_history[-(self.max_seq_len -1):]

                # Check game over *after* applying the move
                current_game_over_status, current_winner = game.check_game_over() 
                if current_game_over_status:
                    game_over_status = current_game_over_status
                    winner = current_winner
                    break
            
            if game_over_status == "checkmate":
                if winner == model_player_this_game:
                    wins += 1
                elif winner is not None: # Opponent won
                    losses += 1
                else: # Should not happen if winner is set on checkmate
                    draws +=1 
            elif game_over_status == "stalemate":
                draws += 1
            else: # Max turns reached
                draws += 1
        
        total_played = wins + losses + draws
        win_rate = (wins / total_played * 100) if total_played > 0 else 0.0
        loss_rate = (losses / total_played * 100) if total_played > 0 else 0.0
        draw_rate = (draws / total_played * 100) if total_played > 0 else 0.0

        results = {
            "total_games": total_played,
            "wins_ai": wins,
            "losses_ai": losses,
            "draws": draws,
            "win_rate_ai_percent": win_rate,
            "loss_rate_ai_percent": loss_rate,
            "draw_rate_percent": draw_rate,
            "ai_player_color_was_red": ai_plays_red, # Record the configuration for this run
            "opponent_type": "random",
            "max_turns_per_game": max_turns_per_game
        }
        print(f"Win Rate Calculation Results: {results}")
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ElephantFormer Model Evaluator")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (.ckpt)")
    parser.add_argument("--pgn_file_path", type=str, required=True, help="Path to the PGN file for evaluation (e.g., test set)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device to run evaluation on")
    parser.add_argument("--metric", type=str, default="accuracy", choices=["accuracy", "perplexity", "win_rate"], help="Evaluation metric to calculate")
    parser.add_argument("--num_win_rate_games", type=int, default=10, help="Number of games to simulate for win rate calculation.")
    parser.add_argument("--max_turns_win_rate", type=int, default=150, help="Max turns per game for win rate calculation.")
    parser.add_argument("--ai_plays_red_win_rate", type=lambda x: (str(x).lower() == 'true'), default=True, help="Set to False if AI should play Black in win rate games (default: True).")

    cli_args = parser.parse_args()

    try:
        evaluator = ModelEvaluator(model_path=cli_args.model_path, device=cli_args.device)

        if cli_args.metric == "accuracy":
            results = evaluator.calculate_prediction_accuracy(pgn_file_path=cli_args.pgn_file_path)
            print(f"\nFinal Prediction Accuracy: {results['accuracy']:.2f}%")
            print(f"(Correct: {results['correct_predictions']}, Total: {results['total_predictions']})")
        elif cli_args.metric == "perplexity":
            results = evaluator.calculate_perplexity(pgn_file_path=cli_args.pgn_file_path)
            print(f"\nPerplexity: {results:.4f}")
        elif cli_args.metric == "win_rate":
            results = evaluator.calculate_win_rate(
                num_games=cli_args.num_win_rate_games,
                max_turns_per_game=cli_args.max_turns_win_rate,
                ai_plays_red=cli_args.ai_plays_red_win_rate
            )
            print(f"\n--- Win Rate vs Random Opponent ---")
            print(f"AI played Red: {results['ai_player_color_was_red']}")
            print(f"Total Games: {results['total_games']}")
            print(f"AI Wins: {results['wins_ai']} ({results['win_rate_ai_percent']:.2f}%)")
            print(f"AI Losses: {results['losses_ai']} ({results['loss_rate_ai_percent']:.2f}%)")
            print(f"Draws: {results['draws']} ({results['draw_rate_percent']:.2f}%)")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
