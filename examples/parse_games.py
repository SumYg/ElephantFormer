"""Example script demonstrating how to use the Elephant Chess parser."""

from elephant_former.data.elephant_parser import parse_iccs_pgn_file, format_moves

def main():
    # Parse the games file (in ICCS format)
    games = parse_iccs_pgn_file('data/WXF-41743games.pgns')
    
    # Print details of the first game
    if games:
        game = games[0]
        print("First Game Details:")
        print(f"Red Player: {game.red_player}")
        print(f"Black Player: {game.black_player}")
        print(f"Result: {game.result}")
        print(f"Number of moves: {len(game.moves)}")
        print(f"Move format: {game.move_format}")
        
        # Print first 10 moves
        print("\nFirst 10 moves (Red-Black):")
        print(format_moves(game, num_moves=10))
        
        # Print some statistics
        print(f"\nTotal games parsed: {len(games)}")

if __name__ == "__main__":
    main() 
