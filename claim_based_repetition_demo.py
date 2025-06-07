#!/usr/bin/env python3
"""
Demonstration of Claim-Based vs Automatic Threefold Repetition

This file shows the difference between:
1. Automatic threefold repetition (old system)
2. Claim-based threefold repetition (traditional system)

In traditional chess/Elephant Chess rules:
- Threefold repetition allows a player to CLAIM a draw
- The game doesn't end automatically
- Players can choose to continue playing even with repetitions
"""

from elephant_former.engine.elephant_chess_game import ElephantChessGame, Player


def demonstrate_claim_based_repetition():
    """Demonstrate the new claim-based repetition system."""
    print("=" * 70)
    print("CLAIM-BASED THREEFOLD REPETITION DEMONSTRATION")
    print("=" * 70)
    
    game = ElephantChessGame()
    
    print("Making moves to create position repetition...")
    
    # Create a sequence that will repeat the position
    move_sequence = [
        (1, 0, 2, 2),  # Red horse out
        (1, 9, 2, 7),  # Black horse out  
        (2, 2, 1, 0),  # Red horse back
        (2, 7, 1, 9),  # Black horse back
    ]
    
    print(f"Starting position repetition count: {game.get_repetition_count()}")
    
    # Execute the sequence multiple times
    for cycle in range(3):
        print(f"\n--- Cycle {cycle + 1} ---")
        for i, move in enumerate(move_sequence):
            player_name = game.current_player.name
            game.apply_move(move)
            print(f"{player_name} moves: {move}")
            
            # Check repetition status after each move
            rep_count = game.get_repetition_count()
            can_claim = game.can_claim_draw_by_repetition()
            
            print(f"  Position repetitions: {rep_count}")
            print(f"  Can claim draw: {can_claim}")
            
            # Check if game ends automatically (it shouldn't now)
            status, winner = game.check_game_over()
            if status:
                print(f"  ❌ Game ended automatically: {status}")
                return
            else:
                print(f"  ✓ Game continues (no automatic end)")
            
            # Demonstrate draw claiming when possible
            if can_claim and cycle >= 2:  # Only claim on the last cycle
                print(f"  🎯 {game.current_player.name} chooses to claim draw!")
                claim_success = game.claim_draw_by_repetition()
                if claim_success:
                    print(f"  ✓ Draw claim accepted - game ends by player choice")
                    return
    
    print("\n✓ Demonstration completed - game continued despite repetitions")


def compare_automatic_vs_claim_based():
    """Compare the old automatic system with new claim-based system."""
    print("\n" + "=" * 70)
    print("AUTOMATIC vs CLAIM-BASED COMPARISON")
    print("=" * 70)
    
    print("""
AUTOMATIC THREEFOLD REPETITION (Old System):
❌ Game ends immediately when position repeats 3 times
❌ No player choice or control
❌ Can interrupt interesting positions
❌ Not traditional chess/Elephant Chess behavior

CLAIM-BASED THREEFOLD REPETITION (New System):
✅ Player can choose to claim draw when eligible
✅ Game continues unless draw is claimed
✅ Allows for strategic decisions about when to draw
✅ Follows traditional chess/Elephant Chess rules
✅ Perpetual check/chase still automatically resolved (as per rules)

HYBRID APPROACH (Recommended):
✅ Perpetual check/chase: Automatic resolution (official rules)
✅ Regular repetition: Claim-based (traditional rules)
✅ Gives players control while enforcing important regulations
    """)


def demonstrate_perpetual_vs_regular_repetition():
    """Show the difference between perpetual patterns and regular repetition."""
    print("\n" + "=" * 70)
    print("PERPETUAL vs REGULAR REPETITION")
    print("=" * 70)
    
    print("""
PERPETUAL CHECK/CHASE (Automatic Resolution):
- These are considered violations of fair play
- Automatically resolved according to official rules
- Player causing the perpetual situation loses (or draw for mutual)
- No choice given to players

REGULAR REPETITION (Claim-Based):
- Normal positional repetition without aggressive intent
- Players can choose to claim draw or continue
- Strategic decision left to the players
- Traditional chess/Elephant Chess behavior

EXAMPLE SCENARIOS:

1. Horse moving back and forth (Regular Repetition):
   → Players can claim draw or continue playing
   
2. Cannon continuously checking king (Perpetual Check):
   → Checking player automatically loses
   
3. Chariot continuously threatening enemy chariot (Perpetual Chase):
   → Chasing player automatically loses
   
4. Both players checking each other (Mutual Perpetual Check):
   → Automatic draw
    """)


def show_usage_examples():
    """Show how to use the new claim-based system."""
    print("\n" + "=" * 70)
    print("USAGE EXAMPLES")
    print("=" * 70)
    
    print("""
# Check if current player can claim draw
if game.can_claim_draw_by_repetition():
    print("Draw available by repetition!")
    
    # Player chooses whether to claim
    user_choice = input("Claim draw? (y/n): ")
    if user_choice.lower() == 'y':
        if game.claim_draw_by_repetition():
            print("Draw claimed successfully!")
        
# Check repetition count
rep_count = game.get_repetition_count()
print(f"Current position has occurred {rep_count} times")

# Normal game over check (perpetual patterns still automatic)
status, winner = game.check_game_over()
if status == "perpetual_check":
    print("Game over - perpetual check detected!")
elif status is None and game.can_claim_draw_by_repetition():
    print("Game continues, but draw available by claim")
    """)


def main():
    """Run the complete demonstration."""
    print("THREEFOLD REPETITION: CLAIM-BASED vs AUTOMATIC")
    print("Implementation demonstrates traditional chess/Elephant Chess behavior")
    
    demonstrate_claim_based_repetition()
    compare_automatic_vs_claim_based()
    demonstrate_perpetual_vs_regular_repetition()
    show_usage_examples()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The updated implementation now distinguishes between:

1. PERPETUAL PATTERNS → Automatic resolution (official rules)
   - Perpetual check: Checking player loses
   - Perpetual chase: Chasing player loses
   - Mutual perpetual: Draw
   - Check vs chase: Chasing player loses

2. REGULAR REPETITION → Claim-based (traditional rules)
   - Players can choose to claim draw when eligible
   - Game continues unless draw is explicitly claimed   - Follows traditional chess/Elephant Chess behavior

This provides the best of both worlds:
- Official regulation enforcement for unfair perpetual patterns
- Traditional player choice for normal repetitions
    """)


if __name__ == "__main__":
    main()
