# elephant_former/data_utils/repetition_checker.py
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path to allow running as a script
# This is a common pattern for utility scripts within a package
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from elephant_former.data.elephant_parser import ElephantGame, parse_iccs_pgn_file
from elephant_former.data_utils.tokenization_utils import parse_iccs_move_to_coords
from elephant_former.engine.elephant_chess_game import ElephantChessGame


def check_games_for_repetition(games: List[ElephantGame]) -> List[Dict[str, Any]]:
    """
    Simulates games from a list of ElephantGame objects to check for threefold repetition.

    Args:
        games: A list of ElephantGame objects.

    Returns:
        A list of dictionaries, where each dictionary contains information
        about a game that ended in a draw by repetition.
    """
    repetition_games = []

    print(f"Checking {len(games)} games for threefold repetition...")

    for i, game_data in enumerate(games):
        game_engine = ElephantChessGame()
        
        # Use the parsed_moves property from the ElephantGame dataclass
        parsed_moves = game_data.parsed_moves
        if not parsed_moves:
            continue

        for move_num, move_str in enumerate(parsed_moves, 1):
            try:
                # Convert ICCS move string to coordinate tuple
                move_coords = parse_iccs_move_to_coords(move_str)
                if move_coords is None:
                    # print(f"Warning: Could not parse move '{move_str}' in game {i+1}. Skipping rest of game.")
                    break
                
                # Check if the move is legal before applying. This is a sanity check.
                # The repetition check is inside the engine's state update.
                legal_moves = game_engine.get_all_legal_moves(game_engine.current_player)
                if move_coords not in legal_moves:
                    # print(f"Warning: Illegal move '{move_str}' encountered in game {i+1} at move {move_num}. Skipping rest of game.")
                    break

                game_engine.apply_move(move_coords)
                
                status, _ = game_engine.check_game_over()

                if status == "draw_by_repetition":
                    game_info = {
                        "game_index": i + 1,
                        "event": game_data.metadata.get("Event", "Unknown Event"),
                        "red_player": game_data.metadata.get("Red", "Unknown"),
                        "black_player": game_data.metadata.get("Black", "Unknown"),
                        "repetition_at_move": move_num
                    }
                    repetition_games.append(game_info)
                    # Once repetition is found, we can stop processing this game
                    break

            except Exception as e:
                # print(f"An error occurred while processing game {i+1}, move '{move_str}': {e}")
                break # Move to the next game

    return repetition_games


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Check a PGN file for games containing threefold repetition draws."
    )
    parser.add_argument(
        "--pgn_file_path",
        type=str,
        required=True,
        help="Path to the PGN file to check."
    )
    args = parser.parse_args()

    pgn_path = Path(args.pgn_file_path)
    if not pgn_path.is_file():
        print(f"Error: PGN file not found at '{pgn_path}'")
        sys.exit(1)

    print(f"Loading games from {pgn_path}...")
    all_games = parse_iccs_pgn_file(str(pgn_path))

    if not all_games:
        print("No games found in the PGN file.")
        sys.exit(0)

    games_with_reps = check_games_for_repetition(all_games)

    print("\n--- Repetition Check Complete ---")
    if not games_with_reps:
        print("No games with threefold repetition draws were found.")
    else:
        print(f"Found {len(games_with_reps)} games with threefold repetition draws:")
        for game in games_with_reps:
            print(
                f"  - Game {game['game_index']}: '{game['event']}' "
                f"({game['red_player']} vs {game['black_player']}) "
                f"- Repetition at move {game['repetition_at_move']}"
            )
    print("-" * 31) 