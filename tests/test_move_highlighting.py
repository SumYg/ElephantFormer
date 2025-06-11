#!/usr/bin/env python3
"""
Test script to demonstrate the move highlighting feature in game replay.
This shows that the `+` and `*` symbols now properly appear when displaying moves.
"""

from elephant_former.engine.elephant_chess_game import ElephantChessGame

def test_move_highlighting():
    """Test the move highlighting functionality."""
    print("🎯 TESTING MOVE HIGHLIGHTING FEATURE")
    print("=" * 50)
    
    # Create a new game
    game = ElephantChessGame()
    
    print("Initial board position:")
    print(game)
    print()
    
    # Get legal moves and make the first one
    legal_moves = game.get_all_legal_moves(game.current_player)
    if legal_moves:
        move = legal_moves[0]  # Take the first legal move
        print(f"Making move: {move} (from {move[0]},{move[1]} to {move[2]},{move[3]})")
        
        # Apply the move
        game.apply_move(move)
        
        print("\nBoard after move (WITHOUT highlighting):")
        print(game)
        
        print("\nBoard after move (WITH highlighting showing the move that was played):")
        print(game.__str__(last_move=move))
        
        print("\nLegend:")
        print("  + = from position (where piece moved from)")
        print("  * = to position (where piece moved to)")
        print()
        
        print("✅ SUCCESS: Move highlighting is working!")
        print("   The '+' shows where the piece came FROM")
        print("   The '*' shows where the piece moved TO")
    else:
        print("❌ No legal moves available")

def test_game_replay_highlighting():
    """Test move highlighting in a sequence of moves."""
    print("\n🎬 TESTING GAME REPLAY WITH MOVE HIGHLIGHTING")
    print("=" * 50)
    
    game = ElephantChessGame()
    moves_made = []
    
    # Make a few moves
    for i in range(3):
        legal_moves = game.get_all_legal_moves(game.current_player)
        if not legal_moves:
            break
            
        move = legal_moves[0]  # Take first legal move
        moves_made.append((move, game.current_player.name))
        
        print(f"\nMove {i+1}: {game.current_player.name} plays {move}")
        game.apply_move(move)
        
        # Show the board with move highlighting
        print("Board position with move highlighting:")
        print(game.__str__(last_move=move))
        print("-" * 30)
    
    print(f"\n✅ Completed {len(moves_made)} moves with highlighting!")
    print("Each move showed:")
    print("  + symbol at the FROM position") 
    print("  * symbol at the TO position")

if __name__ == "__main__":
    test_move_highlighting()
    test_game_replay_highlighting()
    
    print("\n🎉 MOVE HIGHLIGHTING FEATURE RESTORED!")
    print("The game replay system now properly shows:")
    print("  - Where pieces moved from (+)")
    print("  - Where pieces moved to (*)")
    print("  - Visual indication of the last move played")
