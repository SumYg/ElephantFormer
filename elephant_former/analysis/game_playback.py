# elephant_former/analysis/game_playback.py

import torch
import torch.nn.functional as F
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import random

from elephant_former.training.lightning_module import LightningElephantFormer
from elephant_former.engine.elephant_chess_game import ElephantChessGame, Player, Move
from elephant_former.data_utils.tokenization_utils import coords_to_unified_token_ids
from elephant_former.models.transformer_model import generate_square_subsequent_mask
from elephant_former import constants

@dataclass
class MoveAnalysis:
    """Analysis of a single move including model predictions and confidence."""
    move_number: int
    player: str
    actual_move: Move
    legal_moves: List[Move]
    model_predictions: Optional[List[Tuple[Move, float]]]  # Top moves with confidence scores
    was_model_move: bool
    confidence_score: Optional[float]  # Log probability of the actual move
    top_choice_match: bool  # Whether actual move was model's top choice
    board_state_fen: Optional[str]

@dataclass
class GameRecord:
    """Complete record of a game with detailed analysis."""
    game_id: str
    ai_color: str
    opponent_type: str
    result: str  # "win", "loss", "draw"
    winner: Optional[str]
    total_moves: int
    max_turns_reached: bool
    moves: List[MoveAnalysis]
    final_position_fen: str
    ai_accuracy: float  # Percentage of moves where AI chose top predicted move
    avg_confidence: float  # Average confidence score for AI moves

class GamePlaybackAnalyzer:
    """Analyze and replay games played by the model."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model_path = model_path
        self.device = torch.device(device)
        self.model: Optional[LightningElephantFormer] = None
        self.max_seq_len = 0
        
        self._load_model()
    
    def _load_model(self):
        """Load the model from checkpoint."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")
        
        try:
            print(f"Loading model from checkpoint: {self.model_path}")
            self.model = LightningElephantFormer.load_from_checkpoint(
                checkpoint_path=self.model_path, 
                map_location=self.device
            )
            self.model.to(self.device)
            self.model.eval()
            self.max_seq_len = self.model.hparams.max_seq_len
            print(f"Model loaded. Max sequence length: {self.max_seq_len}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def get_model_move_predictions(self, game: ElephantChessGame, 
                                 token_history: List[int], 
                                 top_k: int = 5) -> List[Tuple[Move, float]]:
        """Get top-k move predictions from the model with confidence scores."""
        legal_moves = game.get_all_legal_moves(game.current_player)
        if not legal_moves:
            return []
        
        # Prepare input for model
        input_token_ids = token_history[:self.max_seq_len]
        current_input_len = len(input_token_ids)
        padding_needed = self.max_seq_len - current_input_len
        padded_input = input_token_ids + [constants.PAD_TOKEN_ID] * padding_needed
        
        src = torch.tensor([padded_input], dtype=torch.long, device=self.device)
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
        if idx_for_prediction < 0 or idx_for_prediction >= logits_fx.size(1):
            return []
        
        # Get log probabilities for each component
        log_probs_fx = F.log_softmax(logits_fx[0, idx_for_prediction, :], dim=-1)
        log_probs_fy = F.log_softmax(logits_fy[0, idx_for_prediction, :], dim=-1)
        log_probs_tx = F.log_softmax(logits_tx[0, idx_for_prediction, :], dim=-1)
        log_probs_ty = F.log_softmax(logits_ty[0, idx_for_prediction, :], dim=-1)
        
        # Score all legal moves
        move_scores = []
        for move in legal_moves:
            fx, fy, tx, ty = move
            score = (log_probs_fx[fx] + log_probs_fy[fy] + 
                    log_probs_tx[tx] + log_probs_ty[ty]).item()
            move_scores.append((move, score))
        
        # Sort by score and return top-k
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return move_scores[:top_k]
    
    def analyze_game(self, ai_plays_red: bool = True, 
                    max_turns: int = 200, 
                    game_id: Optional[str] = None) -> GameRecord:
        """Play and analyze a single game."""
        if game_id is None:
            game_id = f"game_{random.randint(1000, 9999)}"
        
        print(f"Analyzing game {game_id} (AI plays {'Red' if ai_plays_red else 'Black'})")
        
        game = ElephantChessGame()
        token_history = [constants.START_TOKEN_ID]
        ai_player = Player.RED if ai_plays_red else Player.BLACK
        moves_analysis = []
        
        game_over = False
        winner = None
        result = None
        move_count = 0
        
        for turn_num in range(max_turns):
            if game_over:
                break
                
            current_player = game.current_player
            legal_moves = game.get_all_legal_moves(current_player)
            
            if not legal_moves:
                if game.is_king_in_check(current_player):
                    winner = game.get_opponent(current_player)
                    result = "win" if winner == ai_player else "loss"
                    game_over = True
                else:
                    result = "draw"
                    game_over = True
                break
            
            # Get model predictions for this position
            model_predictions = self.get_model_move_predictions(game, token_history)
            
            # Determine the move to play
            if current_player == ai_player:
                # AI move - use model's top prediction
                actual_move = model_predictions[0][0] if model_predictions else random.choice(legal_moves)
                was_model_move = True
            else:
                # Opponent move - random
                actual_move = random.choice(legal_moves)
                was_model_move = False
            
            # Calculate confidence and accuracy metrics
            confidence_score = None
            top_choice_match = False
            
            if model_predictions:
                # Find the confidence score for the actual move
                for move, score in model_predictions:
                    if move == actual_move:
                        confidence_score = score
                        break
                
                # Check if actual move matches top prediction
                if model_predictions[0][0] == actual_move:
                    top_choice_match = True
            
            # Record move analysis
            move_analysis = MoveAnalysis(
                move_number=move_count + 1,
                player=current_player.name,
                actual_move=actual_move,
                legal_moves=legal_moves.copy(),
                model_predictions=model_predictions,
                was_model_move=was_model_move,
                confidence_score=confidence_score,
                top_choice_match=top_choice_match,
                board_state_fen=game.get_fen()
            )
            moves_analysis.append(move_analysis)
            
            # Apply the move
            game.apply_move(actual_move)
            move_count += 1
            
            # Update token history
            move_tokens = coords_to_unified_token_ids(actual_move)
            token_history.extend(move_tokens)
            
            # Truncate history if needed
            if len(token_history) > self.max_seq_len:
                token_history = [constants.START_TOKEN_ID] + token_history[-(self.max_seq_len-1):]
            
            # Check for game over
            game_status, game_winner = game.check_game_over()
            if game_status:
                if game_status == "checkmate":
                    winner = game_winner
                    result = "win" if winner == ai_player else "loss"
                else:
                    result = "draw"
                game_over = True
        
        # Handle max turns reached
        if not game_over:
            result = "draw"
        
        # Calculate AI performance metrics
        ai_moves = [m for m in moves_analysis if m.was_model_move]
        ai_accuracy = sum(1 for m in ai_moves if m.top_choice_match) / len(ai_moves) if ai_moves else 0
        avg_confidence = sum(m.confidence_score for m in ai_moves if m.confidence_score is not None) / len(ai_moves) if ai_moves else 0
        
        return GameRecord(
            game_id=game_id,
            ai_color=ai_player.name,
            opponent_type="random",
            result=result,
            winner=winner.name if winner else None,
            total_moves=move_count,
            max_turns_reached=(turn_num >= max_turns - 1),
            moves=moves_analysis,
            final_position_fen=game.get_fen(),
            ai_accuracy=ai_accuracy * 100,
            avg_confidence=avg_confidence
        )
    
    def run_analysis_batch(self, num_games: int = 10, 
                          ai_plays_red: bool = True,
                          max_turns: int = 200,
                          save_path: Optional[str] = None) -> List[GameRecord]:
        """Run a batch of games and analyze them."""
        print(f"Running analysis on {num_games} games...")
        
        games = []
        wins = losses = draws = 0
        
        for i in tqdm(range(num_games), desc="Analyzing games"):
            game_record = self.analyze_game(
                ai_plays_red=ai_plays_red,
                max_turns=max_turns,
                game_id=f"batch_{i+1:03d}"
            )
            games.append(game_record)
            
            if game_record.result == "win":
                wins += 1
            elif game_record.result == "loss":
                losses += 1
            else:
                draws += 1
        
        # Print summary
        print(f"\nBatch Analysis Summary:")
        print(f"Wins: {wins}/{num_games} ({wins/num_games*100:.1f}%)")
        print(f"Losses: {losses}/{num_games} ({losses/num_games*100:.1f}%)")
        print(f"Draws: {draws}/{num_games} ({draws/num_games*100:.1f}%)")
        
        avg_accuracy = sum(g.ai_accuracy for g in games) / len(games)
        avg_confidence = sum(g.avg_confidence for g in games) / len(games)
        print(f"Average AI accuracy: {avg_accuracy:.1f}%")
        print(f"Average confidence: {avg_confidence:.3f}")
        
        # Save if requested
        if save_path:
            self.save_games(games, save_path)
            print(f"Games saved to {save_path}")
        
        return games
    
    def save_games(self, games: List[GameRecord], filepath: str):
        """Save game records to JSON file."""
        # Convert to serializable format
        games_data = []
        for game in games:
            game_dict = asdict(game)
            # Convert moves to serializable format
            game_dict['moves'] = [
                {
                    **asdict(move),
                    'actual_move': list(move.actual_move),
                    'legal_moves': [list(m) for m in move.legal_moves],
                    'model_predictions': [(list(m), s) for m, s in (move.model_predictions or [])]
                }
                for move in game.moves
            ]
            games_data.append(game_dict)
        
        with open(filepath, 'w') as f:
            json.dump(games_data, f, indent=2)
    
    def load_games(self, filepath: str) -> List[GameRecord]:
        """Load game records from JSON file."""
        with open(filepath, 'r') as f:
            games_data = json.load(f)
        
        games = []
        for game_dict in games_data:
            # Convert moves back to proper format
            moves = []
            for move_dict in game_dict['moves']:
                move_dict['actual_move'] = tuple(move_dict['actual_move'])
                move_dict['legal_moves'] = [tuple(m) for m in move_dict['legal_moves']]
                if move_dict['model_predictions']:
                    move_dict['model_predictions'] = [(tuple(m), s) for m, s in move_dict['model_predictions']]
                moves.append(MoveAnalysis(**move_dict))
            
            game_dict['moves'] = moves
            games.append(GameRecord(**game_dict))
        
        return games

class GameReplay:
    """Interactive game replay system."""
    
    def __init__(self, analyzer: GamePlaybackAnalyzer):
        self.analyzer = analyzer
    
    def replay_game(self, game_record: GameRecord, step_by_step: bool = True):
        """Replay a game with detailed analysis."""
        print(f"\n{'='*60}")
        print(f"GAME REPLAY: {game_record.game_id}")
        print(f"{'='*60}")
        print(f"AI Color: {game_record.ai_color}")
        print(f"Result: {game_record.result.upper()}")
        print(f"Winner: {game_record.winner or 'Draw'}")
        print(f"Total Moves: {game_record.total_moves}")
        print(f"AI Accuracy: {game_record.ai_accuracy:.1f}%")
        print(f"Avg Confidence: {game_record.avg_confidence:.3f}")
        print()
        
        # Create game for visualization
        game = ElephantChessGame()
        
        print("Initial Position:")
        print(game)
        print()
        
        if step_by_step:
            input("Press Enter to start replay...")
        
        for i, move_analysis in enumerate(game_record.moves):
            print(f"\nMove {move_analysis.move_number}: {move_analysis.player}")
            print(f"Move: {move_analysis.actual_move}")
            
            if move_analysis.was_model_move:
                print(f"🤖 AI Move (Confidence: {move_analysis.confidence_score:.3f})")
                print(f"Top choice match: {'✅' if move_analysis.top_choice_match else '❌'}")
                
                if move_analysis.model_predictions:
                    print("Top 3 model predictions:")
                    for j, (move, score) in enumerate(move_analysis.model_predictions[:3]):
                        marker = "👑" if j == 0 else f"{j+1}."
                        chosen = "← CHOSEN" if move == move_analysis.actual_move else ""
                        print(f"  {marker} {move} (score: {score:.3f}) {chosen}")
            else:
                print("🎲 Random opponent move")
              # Apply move to show resulting position
            game.apply_move(move_analysis.actual_move)
            print(f"\nPosition after move:")
            print(game.__str__(last_move=move_analysis.actual_move))
            
            if step_by_step and i < len(game_record.moves) - 1:
                input("Press Enter for next move...")
        
        print(f"\n🏁 Game ended: {game_record.result.upper()}")
        if game_record.winner:
            print(f"Winner: {game_record.winner}")
    
    def analyze_patterns(self, games: List[GameRecord]):
        """Analyze patterns across multiple games."""
        print(f"\n{'='*60}")
        print(f"PATTERN ANALYSIS ({len(games)} games)")
        print(f"{'='*60}")
        
        wins = [g for g in games if g.result == "win"]
        losses = [g for g in games if g.result == "loss"]
        draws = [g for g in games if g.result == "draw"]
        
        print(f"\nGame Results:")
        print(f"Wins: {len(wins)} ({len(wins)/len(games)*100:.1f}%)")
        print(f"Losses: {len(losses)} ({len(losses)/len(games)*100:.1f}%)")
        print(f"Draws: {len(draws)} ({len(draws)/len(games)*100:.1f}%)")
        
        # Accuracy comparison
        if wins and losses:
            win_accuracy = sum(g.ai_accuracy for g in wins) / len(wins)
            loss_accuracy = sum(g.ai_accuracy for g in losses) / len(losses)
            print(f"\nAccuracy Analysis:")
            print(f"Average accuracy in wins: {win_accuracy:.1f}%")
            print(f"Average accuracy in losses: {loss_accuracy:.1f}%")
            print(f"Accuracy difference: {win_accuracy - loss_accuracy:+.1f}%")
        
        # Confidence comparison
        if wins and losses:
            win_confidence = sum(g.avg_confidence for g in wins) / len(wins)
            loss_confidence = sum(g.avg_confidence for g in losses) / len(losses)
            print(f"\nConfidence Analysis:")
            print(f"Average confidence in wins: {win_confidence:.3f}")
            print(f"Average confidence in losses: {loss_confidence:.3f}")
            print(f"Confidence difference: {win_confidence - loss_confidence:+.3f}")
        
        # Game length analysis
        win_lengths = [g.total_moves for g in wins]
        loss_lengths = [g.total_moves for g in losses]
        
        if win_lengths and loss_lengths:
            avg_win_length = sum(win_lengths) / len(win_lengths)
            avg_loss_length = sum(loss_lengths) / len(loss_lengths)
            print(f"\nGame Length Analysis:")
            print(f"Average moves in wins: {avg_win_length:.1f}")
            print(f"Average moves in losses: {avg_loss_length:.1f}")
        
        # Find interesting games
        print(f"\nInteresting Games:")
        if wins:
            best_win = max(wins, key=lambda g: g.ai_accuracy)
            print(f"Best win: {best_win.game_id} (accuracy: {best_win.ai_accuracy:.1f}%)")
        
        if losses:
            worst_loss = min(losses, key=lambda g: g.ai_accuracy)
            print(f"Worst loss: {worst_loss.game_id} (accuracy: {worst_loss.ai_accuracy:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="ElephantFormer Game Playback Analyzer")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"],
                       help="Device to use")
    parser.add_argument("--num_games", type=int, default=10,
                       help="Number of games to analyze")
    parser.add_argument("--ai_plays_red", type=bool, default=True,
                       help="Whether AI plays as Red")
    parser.add_argument("--max_turns", type=int, default=200,
                       help="Maximum turns per game")
    parser.add_argument("--save_path", type=str, default=None,
                       help="Path to save game analysis")
    parser.add_argument("--load_path", type=str, default=None,
                       help="Path to load existing game analysis")
    parser.add_argument("--replay_game", type=str, default=None,
                       help="Game ID to replay interactively")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = GamePlaybackAnalyzer(args.model_path, args.device)
    replay = GameReplay(analyzer)
    
    if args.load_path:
        # Load existing games
        print(f"Loading games from {args.load_path}")
        games = analyzer.load_games(args.load_path)
        print(f"Loaded {len(games)} games")
        
        if args.replay_game:
            # Find and replay specific game
            target_game = next((g for g in games if g.game_id == args.replay_game), None)
            if target_game:
                replay.replay_game(target_game)
            else:
                print(f"Game {args.replay_game} not found")
                print("Available games:", [g.game_id for g in games])
        else:
            # Analyze patterns
            replay.analyze_patterns(games)
    
    else:
        # Run new analysis
        games = analyzer.run_analysis_batch(
            num_games=args.num_games,
            ai_plays_red=args.ai_plays_red,
            max_turns=args.max_turns,
            save_path=args.save_path
        )
        
        # Show pattern analysis
        replay.analyze_patterns(games)
        
        # Offer to replay interesting games
        wins = [g for g in games if g.result == "win"]
        losses = [g for g in games if g.result == "loss"]
        
        if wins:
            best_win = max(wins, key=lambda g: g.ai_accuracy)
            print(f"\nWould you like to replay the best win ({best_win.game_id})? (y/n)")
            if input().lower() == 'y':
                replay.replay_game(best_win)
        
        if losses:
            worst_loss = min(losses, key=lambda g: g.ai_accuracy)
            print(f"\nWould you like to replay the worst loss ({worst_loss.game_id})? (y/n)")
            if input().lower() == 'y':
                replay.replay_game(worst_loss)


if __name__ == "__main__":
    main()
