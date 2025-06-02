"""Parser for Elephant Chess (Chinese Chess) game records in ICCS format."""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class ElephantGame:
    """Represents a single game of Elephant Chess with metadata and moves."""
    metadata: Dict[str, str]
    moves: List[str]
    move_format: str = "ICCS"  # Format of the moves (ICCS, WXF, etc.)
    
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
    
    return ElephantGame(metadata=metadata, moves=moves, move_format="ICCS")

def parse_iccs_pgn_file(file_path: str | Path) -> List[ElephantGame]:
    """Parse a PGN file containing multiple Elephant Chess games in ICCS format.
    
    Args:
        file_path: Path to the PGN file containing ICCS format moves
        
    Returns:
        List of ElephantGame objects
    """
    file_path = Path(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into individual games
    games = []
    current_game = []
    for line in content.split('\n'):
        if line.startswith('[Game') and current_game:
            games.append('\n'.join(current_game))
            current_game = []
        if line.strip():
            current_game.append(line)
    if current_game:
        games.append('\n'.join(current_game))
    
    return [parse_iccs_game(game) for game in games]

def format_moves(game: ElephantGame, num_moves: int = 10) -> str:
    """Format the first N moves of a game for display.
    
    Args:
        game: ElephantGame object
        num_moves: Number of move pairs to format (default: 10)
        
    Returns:
        Formatted string of moves
    """
    moves = []
    for i in range(0, min(num_moves * 2, len(game.moves)), 2):
        move_num = (i // 2) + 1
        red_move = game.moves[i]
        black_move = game.moves[i + 1] if i + 1 < len(game.moves) else ""
        moves.append(f"{move_num}. {red_move} {black_move}")
    return "\n".join(moves) 