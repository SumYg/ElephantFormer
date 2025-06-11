#!/usr/bin/env python3
"""
Quick demo of the game replay system with move highlighting restored.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from elephant_former.analysis.game_playback import GamePlaybackAnalyzer, GameReplay

def quick_replay_demo():
    """Quick demo showing that game replay now has move highlighting."""
    print("🎮 GAME REPLAY MOVE HIGHLIGHTING DEMO")
    print("=" * 50)
    
    # Check if we have analysis results
    results_path = "analysis_results.json"
    if not Path(results_path).exists():
        print("❌ No analysis results found. Please run the analysis first:")
        print("   python -m elephant_former.analysis.game_playback --model_path <path> --num_games 2")
        return
    
    # Load existing analysis
    try:
        # We'll create a dummy analyzer just to load the games
        # (we don't need the model for replay)
        analyzer = GamePlaybackAnalyzer.__new__(GamePlaybackAnalyzer)
        games = analyzer.load_games(results_path)
        
        if not games:
            print("❌ No games found in analysis results")
            return
        
        print(f"✅ Loaded {len(games)} games from analysis results")
        
        # Create replay system
        replay = GameReplay(analyzer)
        
        # Find an interesting game to replay
        game_to_replay = games[0]  # Just take the first game
        
        print(f"\n🎬 Replaying game: {game_to_replay.game_id}")
        print(f"   Result: {game_to_replay.result}")
        print(f"   AI Color: {game_to_replay.ai_color}")
        print(f"   Total Moves: {game_to_replay.total_moves}")
        
        # Replay first few moves with highlighting
        print(f"\n🔍 Showing first 3 moves with move highlighting:")
        print("=" * 50)
        
        # Manually replay to show highlighting
        from elephant_former.engine.elephant_chess_game import ElephantChessGame
        
        game = ElephantChessGame()
        print("Initial Position:")
        print(game)
        
        # Show first 3 moves
        for i, move_analysis in enumerate(game_to_replay.moves[:3]):
            print(f"\n{'='*30}")
            print(f"Move {move_analysis.move_number}: {move_analysis.player}")
            print(f"Move: {move_analysis.actual_move}")
            
            if move_analysis.was_model_move:
                print(f"🤖 AI Move (Confidence: {move_analysis.confidence_score:.3f})")
                print(f"Top choice match: {'✅' if move_analysis.top_choice_match else '❌'}")
            else:
                print("🎲 Random opponent move")
            
            # Apply move and show with highlighting
            game.apply_move(move_analysis.actual_move)
            print(f"\nPosition after move (WITH HIGHLIGHTING):")
            print(game.__str__(last_move=move_analysis.actual_move))
        
        print(f"\n🎉 SUCCESS! Move highlighting is working in game replay!")
        print("   + shows where piece moved FROM")
        print("   * shows where piece moved TO")
        
    except Exception as e:
        print(f"❌ Error loading games: {e}")
        print("This demo requires existing analysis results.")

if __name__ == "__main__":
    quick_replay_demo()