"""Parser for Elephant Chess game records in ICCS format."""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from pathlib import Path

# Regex to capture individual moves like H2-E2 or C9-E7
MOVE_REGEX = re.compile(r"[A-I][0-9]-[A-I][0-9]")

@dataclass
class ElephantGame:
    """Represents a single game of Elephant Chess with metadata and moves."""
    metadata: Dict[str, str]
    iccs_moves_string: str # Raw, concatenated move string as it appears in PGN (e.g., "1. H2-E2 C9-E7 2. E2-D2 ... Result")
    move_format: str = "ICCS"

    # Store parsed moves to avoid re-parsing.
    _parsed_moves_cache: Optional[List[str]] = field(default=None, repr=False, init=False)

    @property
    def parsed_moves(self) -> List[str]:
        """Parses the iccs_moves_string into a list of individual move strings (e.g., ["H2-E2", "C9-E7"])."""
        if self._parsed_moves_cache is not None:
            return self._parsed_moves_cache

        if not self.iccs_moves_string:
            self._parsed_moves_cache = []
            return []

        # Find all occurrences of the move pattern
        self._parsed_moves_cache = MOVE_REGEX.findall(self.iccs_moves_string)
        return self._parsed_moves_cache
    
    @property
    def red_player(self) -> str:
        """Get the name of the red player."""
        return self.metadata.get('Red', 'Unknown')
    
    @property
    def black_player(self) -> str:
        """Get the name of the black player."""
        return self.metadata.get('Black', 'Unknown')
    
    @property
    def result(self) -> str:
        """Get the game result."""
        return self.metadata.get('Result', 'Unknown')
    
    @property
    def initial_fen(self) -> Optional[str]:
        """Get the initial FEN position if available."""
        return self.metadata.get('FEN')

def parse_iccs_game(game_text: str) -> ElephantGame:
    """Parse a single game string in ICCS format into an ElephantGame object.
    
    Args:
        game_text: Raw game text in PGN format with ICCS moves
        
    Returns:
        ElephantGame object containing the parsed game data
    """
    # Extract metadata
    metadata = {}
    metadata_pattern = r'\[(.*?) "(.*?)"\]'
    for key, value in re.findall(metadata_pattern, game_text):
        metadata[key] = value
    
    # Extract moves
    # Find the moves section (everything after the last metadata entry)
    moves_section = game_text.split(']')[-1].strip()
    # Remove the result from the end if present
    moves_section = re.sub(r'\s*(?:1-0|0-1|1/2-1/2)\s*$', '', moves_section)
    
    # Extract moves using regex
    move_pattern = r'\d+\.\s+([A-I]\d-[A-I]\d)(?:\s+([A-I]\d-[A-I]\d))?'
    moves = []
    for match in re.finditer(move_pattern, moves_section):
        moves.append(match.group(1))  # Red's move
        if match.group(2):  # Black's move (if exists)
            moves.append(match.group(2))
    
    return ElephantGame(metadata=metadata, iccs_moves_string=" ".join(moves), move_format="ICCS")

def parse_iccs_pgn_file(file_path: Union[str, Path]) -> List[ElephantGame]:
    """Parse a PGN file containing multiple Elephant Chess games in ICCS format.
    
    Args:
        file_path: Path to the PGN file containing ICCS format moves
        
    Returns:
        List of ElephantGame objects
    """
    file_path = Path(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    games_data = []
    current_game_lines = []
    in_game_moves_section = False

    for line in content.split('\n'):
        stripped_line = line.strip()
        if stripped_line.startswith("[Event ") and current_game_lines:
            # Start of a new game, process the previous one
            game_str = "\n".join(current_game_lines)
            meta = {}
            moves_list_for_game = [] # Stores lines belonging to moves
            
            meta_part_ended = False
            for game_line_raw in game_str.split('\n'):
                game_line = game_line_raw.strip()
                if not game_line: # Empty lines can separate meta from moves or appear in moves
                    if not meta_part_ended and meta: # If we have metadata and see an empty line, assume meta ended.
                        meta_part_ended = True
                    elif meta_part_ended: # if empty line is within moves, preserve it (or rather, the space)
                        moves_list_for_game.append(" ") 
                    continue

                if game_line.startswith("[") and not meta_part_ended:
                    match = re.match(r'\[(.*?) "(.*?)"\]', game_line)
                    if match:
                        key, value = match.groups()
                        meta[key.strip()] = value.strip()
                else: # This line is part of moves or comments after metadata
                    meta_part_ended = True # any non-metadata line means metadata section is over
                    moves_list_for_game.append(game_line)
            
            if meta: # Only proceed if we have some metadata
                full_moves_string = " ".join(moves_list_for_game).strip()
                # The result is often part of the PGN move string, keep it.
                games_data.append(ElephantGame(metadata=meta, iccs_moves_string=full_moves_string))
            current_game_lines = []

        if stripped_line: # Add non-empty lines to current game buffer
            current_game_lines.append(stripped_line)

    # Process the last game in the file
    if current_game_lines:
        game_str = "\n".join(current_game_lines)
        meta = {}
        moves_list_for_game = []
        meta_part_ended = False
        for game_line_raw in game_str.split('\n'):
            game_line = game_line_raw.strip()
            if not game_line:
                if not meta_part_ended and meta:
                    meta_part_ended = True
                elif meta_part_ended:
                     moves_list_for_game.append(" ")
                continue
            
            if game_line.startswith("[") and not meta_part_ended:
                match = re.match(r'\[(.*?) "(.*?)"\]', game_line)
                if match:
                    key, value = match.groups()
                    meta[key.strip()] = value.strip()
            else:
                meta_part_ended = True
                moves_list_for_game.append(game_line)
        
        if meta:
            full_moves_string = " ".join(moves_list_for_game).strip()
            games_data.append(ElephantGame(metadata=meta, iccs_moves_string=full_moves_string))
            
    return games_data

def format_moves(game: ElephantGame, num_moves: int = 10) -> str:
    """Format the first N moves of a game for display, using parsed_moves."""
    moves_to_display = []
    # Access parsed_moves via the property
    actual_moves = game.parsed_moves 
    for i in range(0, min(num_moves * 2, len(actual_moves)), 2):
        move_num = (i // 2) + 1
        red_move = actual_moves[i]
        black_move = actual_moves[i + 1] if i + 1 < len(actual_moves) else ""
        moves_to_display.append(f"{move_num}. {red_move} {black_move}")
    return "\n".join(moves_to_display)

def games_to_pgn_string(games: List[ElephantGame]) -> str:
    """
    Converts a list of ElephantGame objects back into a PGN formatted string.
    """
    pgn_parts = []
    for game in games:
        game_str_parts = []
        for key, value in game.metadata.items():
            game_str_parts.append(f'[{key} "{value}"]')
        game_str_parts.append("") # Empty line after metadata
        
        # game.iccs_moves_string should already contain the full move text including result
        game_str_parts.append(game.iccs_moves_string) 
        pgn_parts.append("\n".join(game_str_parts))
    
    return "\n\n".join(pgn_parts)

def save_games_to_pgn_file(games: List[ElephantGame], file_path: Union[str, Path]):
    """
    Saves a list of ElephantGame objects to a PGN file.
    """
    pgn_content = games_to_pgn_string(games)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(pgn_content)

if __name__ == '__main__':
    # Create a dummy PGN file for testing
    test_pgn_content = ("""
[Event "Test Game 1"]
[Site "Local"]
[Date "2023.10.26"]
[Round "1"]
[White "PlayerA"]
[Black "PlayerB"]
[Result "1-0"]

1. H2-E2 C9-E7
2. E2-D2 H9-G7
1-0

[Event "Test Game 2"]
[Site "Internet"]
[Date "2023.10.27"]
[Result "0-1"]
[FEN "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"]

1. A0-A1 B0-C0
2. A1-A2 C0-C1
0-1
""")
    test_pgn_file = Path("test_sample.pgn")
    with open(test_pgn_file, 'w', encoding='utf-8') as f:
        f.write(test_pgn_content)

    print(f"Testing PGN parsing with {test_pgn_file}...")
    parsed_games = parse_iccs_pgn_file(test_pgn_file)
    print(f"Parsed {len(parsed_games)} games.")

    if not parsed_games:
        print("No games were parsed. Check parse_iccs_pgn_file logic.")
    else:
        for i, game in enumerate(parsed_games):
            print(f"\nGame {i+1}:")
            print(f"  Metadata: {game.metadata}")
            print(f"  ICCS Moves String: {game.iccs_moves_string}")
            print(f"  Parsed Moves List: {game.parsed_moves}")
            if game.initial_fen:
                print(f"  FEN: {game.initial_fen}")
            # print(f"  Formatted Moves (first 3):\n{format_moves(game, num_moves=3)}")

        # Test saving the parsed games
        saved_pgn_file = Path("test_saved_games.pgn")
        print(f"\nSaving parsed games to {saved_pgn_file}...")
        save_games_to_pgn_file(parsed_games, saved_pgn_file)
        print(f"Games saved. Load and verify {saved_pgn_file} manually or by parsing again.")

        # Test parsing the saved file
        print(f"\nTesting PGN parsing with the saved file {saved_pgn_file}...")
        re_parsed_games = parse_iccs_pgn_file(saved_pgn_file)
        print(f"Parsed {len(re_parsed_games)} games from the saved file.")
        
        assert len(parsed_games) == len(re_parsed_games), "Number of games mismatch after save and re-parse!"
        
        for i, (original_game, re_parsed_game) in enumerate(zip(parsed_games, re_parsed_games)):
            assert original_game.metadata == re_parsed_game.metadata, f"Metadata mismatch in game {i+1}\nOriginal: {original_game.metadata}\nRe-parsed: {re_parsed_game.metadata}"
            
            # Normalize spaces for comparison, as PGN move formatting can vary slightly
            original_moves_str = ' '.join(original_game.iccs_moves_string.split())
            re_parsed_moves_str = ' '.join(re_parsed_game.iccs_moves_string.split())
            assert original_moves_str == re_parsed_moves_str, (
                f"ICCS Moves String mismatch in game {i+1}\n"
                f"Original:  '{original_moves_str}' (Length: {len(original_moves_str)})\n"
                f"Re-parsed: '{re_parsed_moves_str}' (Length: {len(re_parsed_moves_str)})"
            )
            assert original_game.parsed_moves == re_parsed_game.parsed_moves, f"Parsed moves list mismatch in game {i+1}"

        print("\nAll parser tests passed (including save and re-parse).")

    # Clean up test files
    if test_pgn_file.exists():
        test_pgn_file.unlink()
    if 'saved_pgn_file' in locals() and saved_pgn_file.exists():
        saved_pgn_file.unlink()
    print("Cleaned up test files.") 