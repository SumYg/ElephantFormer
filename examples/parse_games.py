"""Example script demonstrating how to use the Elephant Chess parser."""

from elephant_former.data.elephant_parser import parse_iccs_pgn_file, format_moves

def main():
    import sys
    
    # Get file path from command line or use default
    if len(sys.argv) > 1:
        pgn_file = sys.argv[1]
    else:
        pgn_file = 'data/WXF-41743games.pgns'
    
    print(f"Parsing file: {pgn_file}")
    
    # Parse the games file (in ICCS format)  
    games = parse_iccs_pgn_file(pgn_file)
    
    # Print details of the first game
    if games:
        game = games[0]
        print("First Game Details:")
        print(f"Red Player: {game.red_player}")
        print(f"Black Player: {game.black_player}")
        print(f"Result: {game.result}")
        print(f"Number of moves: {len(game.parsed_moves)}")
        print(f"Move format: {game.move_format}")
        
        # Print first 10 moves
        print("\nFirst 10 moves (Red-Black):")
        print(format_moves(game, num_moves=10))
        
        # Print some statistics
        print(f"\nTotal games parsed: {len(games)}")

if __name__ == "__main__":
    main() 
