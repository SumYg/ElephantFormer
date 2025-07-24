#!/usr/bin/env python3
"""
Script to replay saved evaluation games with board display.
"""

import sys
from pathlib import Path
import time

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from elephant_former.data.elephant_parser import parse_iccs_pgn_file
from elephant_former.engine.elephant_chess_game import ElephantChessGame
from elephant_former.data_utils.tokenization_utils import parse_iccs_move_to_coords

def replay_game_with_board(pgn_file_path, step_by_step=True, delay=1.0):
    """Replay a saved game from PGN file with board display."""
    if not Path(pgn_file_path).exists():
        print(f"File not found: {pgn_file_path}")
        return
    
    print(f"Loading game from: {pgn_file_path}")
    
    try:
        # Parse the PGN file
        games = parse_iccs_pgn_file(pgn_file_path)
        if not games:
            print("No games found in file")
            return
        
        game_data = games[0]
        print("\n" + "="*60)
        print("GAME INFORMATION")
        print("="*60)
        print(f"Event: {game_data.metadata.get('Event', 'Unknown')}")
        print(f"Red Player: {game_data.red_player}")
        print(f"Black Player: {game_data.black_player}")
        print(f"Result: {game_data.result}")
        print(f"Date: {game_data.metadata.get('Date', 'Unknown')}")
        print(f"Total moves: {len(game_data.parsed_moves)}")
        
        # Create game engine for replay
        game = ElephantChessGame()
        
        print("\n" + "="*60)
        print("GAME REPLAY")
        print("="*60)
        print("Initial Position:")
        
        try:
            print(game)
        except UnicodeEncodeError:
            print("Initial board (Unicode display unavailable)")
            print("Piece layout: Standard xiangqi starting position")
        
        if step_by_step:
            input("Press Enter to start replay...")
        
        # Replay each move
        for i, move_iccs in enumerate(game_data.parsed_moves):
            current_player = "Red" if game.current_player.name == "RED" else "Black"
            
            print(f"\n{'='*40}")
            print(f"Move {i+1}: {current_player} plays {move_iccs}")
            print(f"{'='*40}")
            
            try:
                # Convert ICCS to coordinates
                move_coords = parse_iccs_move_to_coords(move_iccs)
                print(f"Coordinates: {move_coords}")
                
                # Apply the move
                game.apply_move(move_coords)
                
                # Show board position after move
                try:
                    print("\nBoard after move:")
                    print(game.__str__(last_move=move_coords))
                except UnicodeEncodeError:
                    print(f"Board position after {move_iccs}")
                    print("(Unicode display unavailable - Chinese characters not supported)")
                    print(f"Move applied: from ({move_coords[0]},{move_coords[1]}) to ({move_coords[2]},{move_coords[3]})")
                
                # Check for game over
                game_status, winner = game.check_game_over()
                if game_status:
                    print(f"\n🏁 GAME ENDED: {game_status.upper()}")
                    if winner:
                        print(f"Winner: {winner.name}")
                    else:
                        print("Result: Draw")
                    break
                
                # Pause between moves if step-by-step
                if step_by_step:
                    if i < len(game_data.parsed_moves) - 1:  # Don't pause after last move
                        input("Press Enter for next move (or Ctrl+C to exit)...")
                else:
                    time.sleep(delay)
                        
            except Exception as e:
                print(f"❌ Error applying move {move_iccs}: {e}")
                print("Stopping replay...")
                break
        
        print(f"\n🎉 Replay complete!")
        print(f"Final result: {game_data.result}")
        
        # Summary
        total_moves = len(game_data.parsed_moves)
        print(f"\nGame Summary:")
        print(f"- Total moves played: {total_moves}")
        print(f"- Average moves per player: {total_moves/2:.1f}")
        if "win" in pgn_file_path.lower():
            print("- Outcome: AI Victory 🏆")
        elif "loss" in pgn_file_path.lower():
            print("- Outcome: AI Loss 😞")
        else:
            print("- Outcome: Draw 🤝")
        
    except Exception as e:
        print(f"❌ Error parsing game: {e}")
        import traceback
        traceback.print_exc()

def list_saved_games(directory="evaluation_games"):
    """List all saved evaluation games."""
    games_dir = Path(directory)
    if not games_dir.exists():
        print(f"Directory {directory} does not exist")
        return []
    
    pgn_files = list(games_dir.glob("*.pgn"))
    if not pgn_files:
        print(f"No PGN files found in {directory}")
        return []
    
    print(f"Found {len(pgn_files)} saved games in {directory}:")
    
    wins = [f for f in pgn_files if "win" in f.name]
    losses = [f for f in pgn_files if "loss" in f.name]
    draws = [f for f in pgn_files if "draw" in f.name]
    
    all_games = []
    
    if wins:
        print(f"\n🏆 AI Wins ({len(wins)}):")
        for i, f in enumerate(sorted(wins), 1):
            print(f"  {i:2d}. {f.name}")
            all_games.append(f)
    
    if losses:
        print(f"\n😞 AI Losses ({len(losses)}):")
        for i, f in enumerate(sorted(losses), 1):
            print(f"  {i+len(wins):2d}. {f.name}")
            all_games.append(f)
    
    if draws:
        print(f"\n🤝 Draws ({len(draws)}):")
        for i, f in enumerate(sorted(draws), 1):
            print(f"  {i+len(wins)+len(losses):2d}. {f.name}")
            all_games.append(f)
    
    return all_games

def main():
    if len(sys.argv) < 2:
        print("🏁 ElephantFormer Game Replay Tool")
        print("="*50)
        print("Usage:")
        print("  python replay_with_board.py <pgn_file>         # Replay specific game")
        print("  python replay_with_board.py list              # List all saved games")
        print("  python replay_with_board.py auto <pgn_file>   # Auto-play with delay")
        print("\nOptions:")
        print("  Interactive mode: Press Enter between moves")
        print("  Auto mode: 1 second delay between moves")
        return
    
    if sys.argv[1] == "list":
        games = list_saved_games()
        if games:
            print(f"\nTo replay a game, use:")
            print(f"python replay_with_board.py <filename>")
    elif sys.argv[1] == "auto" and len(sys.argv) > 2:
        print("🚀 Auto-replay mode (1 second delays)")
        replay_game_with_board(sys.argv[2], step_by_step=False, delay=1.0)
    else:
        print("🎮 Interactive replay mode")
        replay_game_with_board(sys.argv[1], step_by_step=True)

if __name__ == "__main__":
    main()