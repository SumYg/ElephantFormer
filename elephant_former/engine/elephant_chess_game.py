# elephant_former/engine/elephant_chess_game.py

import numpy as np
from typing import Tuple, List, Optional, Dict
from enum import Enum
from collections import Counter, defaultdict
from rich.text import Text

# --- Constants ---

# Board dimensions (standard Elephant Chess)
BOARD_WIDTH = 9
BOARD_HEIGHT = 10

# Players - Replaced with Enum below
# RED = 0
# BLACK = 1

class Player(Enum):
    RED = 0
    BLACK = 1

# Piece Names (integers for board representation, sign for color)
# Positive for RED, Negative for BLACK
EMPTY = 0
# Red pieces
R_KING = 1
R_ADVISOR = 2
R_ELEPHANT = 3
R_HORSE = 4
R_CHARIOT = 5
R_CANNON = 6
R_SOLDIER = 7
# Black pieces
B_KING = -1
B_ADVISOR = -2
B_ELEPHANT = -3
B_HORSE = -4
B_CHARIOT = -5
B_CANNON = -6
B_SOLDIER = -7

PIECE_NAMES = {
    EMPTY: "Empty",
    R_KING: "R_King", R_ADVISOR: "R_Advisor", R_ELEPHANT: "R_Elephant",
    R_HORSE: "R_Horse", R_CHARIOT: "R_Chariot", R_CANNON: "R_Cannon", R_SOLDIER: "R_Soldier",
    B_KING: "B_King", B_ADVISOR: "B_Advisor", B_ELEPHANT: "B_Elephant",
    B_HORSE: "B_Horse", B_CHARIOT: "B_Chariot", B_CANNON: "B_Cannon", B_SOLDIER: "B_Soldier",
}

# Dictionary for a clear, unambiguous character representation of pieces.
# Using Chinese characters for authentic representation with colors.
PIECE_CHARS = {
    EMPTY: Text("．", style="dim"),  # Full-width dot for spacing
    # Red pieces - styled in red/bright colors
    R_KING: Text("帥", style="red bold"), 
    R_ADVISOR: Text("仕", style="red"), 
    R_ELEPHANT: Text("相", style="red"), 
    R_HORSE: Text("傌", style="red"), 
    R_CHARIOT: Text("俥", style="red"), 
    R_CANNON: Text("炮", style="red"), 
    R_SOLDIER: Text("兵", style="red"),
    # Black pieces - styled in black/dark colors
    B_KING: Text("將", style="black bold"), 
    B_ADVISOR: Text("士", style="black"), 
    B_ELEPHANT: Text("象", style="black"), 
    B_HORSE: Text("馬", style="black"), 
    B_CHARIOT: Text("車", style="black"), 
    B_CANNON: Text("砲", style="black"), 
    B_SOLDIER: Text("卒", style="black"),
}

INITIAL_BOARD_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1" # Standard FEN for Elephant Chess

# --- Type Aliases ---
Move = Tuple[int, int, int, int]  # (from_x, from_y, to_x, to_y)
Board = np.ndarray # Should be BOARD_HEIGHT x BOARD_WIDTH, dtype=np.int8

# --- Board Setup ---

def get_piece_from_char(char: str) -> int:
    """Converts FEN character to piece integer."""
    if 'a' <= char <= 'z': # Black pieces
        color_multiplier = -1
    else: # Red pieces
        color_multiplier = 1

    piece_char = char.lower()
    if piece_char == 'k': return B_KING if color_multiplier == -1 else R_KING
    if piece_char == 'a': return B_ADVISOR if color_multiplier == -1 else R_ADVISOR
    if piece_char == 'b': # In FEN, 'b' is often elephant for black, 'e' for red might be used or context. Assuming 'b'/'e' or 'xiang'
        return B_ELEPHANT if color_multiplier == -1 else R_ELEPHANT # Or use 'e' if FEN uses 'e' for red elephants
    if piece_char == 'n': return B_HORSE if color_multiplier == -1 else R_HORSE # Knight/Horse
    if piece_char == 'r': return B_CHARIOT if color_multiplier == -1 else R_CHARIOT # Rook/Chariot
    if piece_char == 'c': return B_CANNON if color_multiplier == -1 else R_CANNON
    if piece_char == 'p': return B_SOLDIER if color_multiplier == -1 else R_SOLDIER
    return EMPTY


def fen_to_board(fen: str = INITIAL_BOARD_FEN) -> Tuple[Board, Player]:
    """
    Converts a FEN string (Forsyth-Edwards Notation for Elephant Chess) to a board representation.
    Only parses the piece placement part and current player.
    Example FEN: rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1
    (standard board, red to move)
    """
    parts = fen.split(' ')
    piece_placement = parts[0]
    player_to_move_char = parts[1]

    board = np.full((BOARD_HEIGHT, BOARD_WIDTH), EMPTY, dtype=np.int8)
    rows = piece_placement.split('/')

    # FEN rows are from Black's perspective (rank 9 down to 0)
    # Our board array is (0-9 for y, 0-8 for x) where (0,0) is bottom-left from Red's view
    # So, FEN's first row (Black's top rank) corresponds to our y=9.
    for i, row_str in enumerate(rows):
        y = BOARD_HEIGHT - 1 - i # y=9 for first FEN row, y=0 for last
        x = 0
        for char_in_row in row_str: # renamed char to char_in_row to avoid conflict with outer scope
            if char_in_row.isdigit():
                x += int(char_in_row)
            else:
                board[y, x] = get_piece_from_char(char_in_row)
                x += 1
    
    current_player = Player.RED if player_to_move_char == 'w' or player_to_move_char == 'r' else Player.BLACK
    return board, current_player


class ElephantChessGame:
    def __init__(self, fen: Optional[str] = None):
        if fen:
            self.board, self.current_player = fen_to_board(fen)
        else:
            self.board, self.current_player = fen_to_board(INITIAL_BOARD_FEN) # Default to start
        self.move_history: List[Move] = [] # Store (fx, fy, tx, ty)
        self.halfmove_clock = 0 # For draws, not fully implemented yet
        self.fullmove_number = 1 # For PGN, not fully implemented yet
        
        # Enhanced position and move tracking for perpetual check/chase detection
        self.position_history = Counter() # type: Counter[str] 
        self.position_sequence: List[str] = [] # Track sequence of positions
        self.move_sequence: List[Tuple[Move, Player]] = [] # Track (move, player_who_made_move)
        self._update_position_history() # Record the initial position

    def get_board_array(self) -> Board:
        """Returns the current board state as a NumPy array."""
        return np.copy(self.board)

    def get_current_player(self) -> Player:
        """Returns the current player (RED or BLACK)."""
        return self.current_player

    def get_opponent(self, player: Player) -> Player:
        """Returns the opponent of the given player."""
        return Player.BLACK if player == Player.RED else Player.RED

    def is_valid_coord(self, x: int, y: int) -> bool:
        """Checks if coordinates are within board limits."""
        return 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT

    def get_piece_at(self, x: int, y: int) -> Optional[int]:
        """Returns the piece at a given coordinate, or None if out of bounds."""
        if not self.is_valid_coord(x,y):
            return None
        return self.board[y, x]

    def get_rich_board(self, last_move: Optional[Move] = None) -> Text:
        """
        Returns a Rich Text object representation of the board with colors.
        Optionally highlights the last move made.
        """
        result = Text()
        result.append("  +------------------+\n")
        fx, fy, tx, ty = (-1,-1,-1,-1)
        if last_move:
            fx, fy, tx, ty = last_move

        for y_idx in range(BOARD_HEIGHT -1, -1, -1):
            result.append(f"{y_idx} |")
            for x_idx in range(BOARD_WIDTH):
                piece = self.board[y_idx, x_idx]
                char_repr = PIECE_CHARS.get(piece, Text('?'))
                
                # Highlight the 'to' square of the last move
                if x_idx == tx and y_idx == ty:
                    # Use background color to highlight the destination (no * marker needed)
                    result.append(" ")
                    if isinstance(char_repr, Text):
                        highlighted = Text(char_repr.plain, style=f"{char_repr.style} on yellow")
                        result.append(highlighted)
                    else:
                        result.append(Text(str(char_repr), style="on yellow"))
                    continue

                # The 'from' square will now be empty. We can mark it.
                if x_idx == fx and y_idx == fy:
                    # Use green background to show where the piece came from (no + marker needed)
                    result.append(" ", style="on green")
                    result.append("．", style="on green")  # Empty square with green background
                    continue
                
                result.append(" ")
                if isinstance(char_repr, Text):
                    result.append(char_repr)
                else:
                    result.append(str(char_repr))
            result.append(" |\n")
        result.append("  +------------------+\n")
        result.append("    0 1 2 3 4 5 6 7 8 (x)\n")
        result.append(f"Current player: ")
        # Use red color for RED player, black color for BLACK player
        player_color = "red bold" if self.current_player == Player.RED else "black bold"
        result.append(self.current_player.name, style=player_color)
        result.append("\n")
        return result

    def __str__(self, last_move: Optional[Move] = None) -> str:
        """
        String representation of the board for printing.
        Optionally highlights the last move made.
        """
        # For backward compatibility, return plain text version
        s = "  +------------------+\n"
        fx, fy, tx, ty = (-1,-1,-1,-1)
        if last_move:
            fx, fy, tx, ty = last_move

        for y_idx in range(BOARD_HEIGHT -1, -1, -1):
            s += f"{y_idx} |"
            for x_idx in range(BOARD_WIDTH):
                piece = self.board[y_idx, x_idx]
                char_repr = PIECE_CHARS.get(piece, Text('?'))
                char_plain = char_repr.plain if isinstance(char_repr, Text) else str(char_repr)
                
                # Highlight the 'to' square of the last move
                if x_idx == tx and y_idx == ty:
                    s += f"*{char_plain}"
                    continue

                # The 'from' square will now be empty. We can mark it.
                if x_idx == fx and y_idx == fy:
                    s += " +"
                    continue
                
                s += f" {char_plain}"
            s += " |\n"
        s += "  +------------------+\n"
        s += "    0 1 2 3 4 5 6 7 8 (x)\n"
        s += f"Current player: {self.current_player.name}\n"
        return s

    def _is_within_palace(self, x: int, y: int, player: Player) -> bool:
        """Checks if (x,y) is within the player's palace."""
        if not (3 <= x <= 5):
            return False
        if player == Player.RED:
            return 0 <= y <= 2
        else: # BLACK
            return 7 <= y <= 9

    def _get_king_moves(self, x: int, y: int, player: Player) -> List[Move]:
        """Generates legal moves for a King at (x,y) for the given player."""
        moves: List[Move] = []
        # King moves one step orthogonally
        deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dx, dy in deltas:
            nx, ny = x + dx, y + dy

            if self.is_valid_coord(nx, ny) and self._is_within_palace(nx, ny, player):
                target_piece = self.board[ny, nx]
                # Cannot capture own piece
                if target_piece == EMPTY or np.sign(target_piece) != np.sign(self.board[y,x]):
                    moves.append((x, y, nx, ny))
        
        # Flying General rule: Kings cannot face each other directly on the same file
        # without any intervening pieces.
        op_king_x, op_king_y = -1, -1
        opponent_player_enum = self.get_opponent(player)
        opponent_king_piece = B_KING if opponent_player_enum == Player.BLACK else R_KING
        
        found_op_king = False
        for r_idx in range(BOARD_HEIGHT): # Renamed r to r_idx
            for c_idx in range(BOARD_WIDTH): # Renamed c to c_idx
                if self.board[r_idx,c_idx] == opponent_king_piece:
                    op_king_x, op_king_y = c_idx, r_idx
                    found_op_king = True
                    break
            if found_op_king:
                break
        
        # print(f"Found opponent king: {found_op_king}")
        if found_op_king and op_king_x == x: # Kings are on the same file
            intervening_pieces = 0
            min_y, max_y = min(y, op_king_y), max(y, op_king_y)
            for i_y in range(min_y + 1, max_y):
                if self.board[i_y, x] != EMPTY:
                    intervening_pieces += 1
                    break
            
            if intervening_pieces == 0:
                potential_moves_after_flying_general_check = []
                for move_candidate in moves:
                    temp_fx, temp_fy, temp_tx, temp_ty = move_candidate
                    original_piece_at_target = self.board[temp_ty, temp_tx]
                    original_piece_at_source = self.board[temp_fy, temp_fx]
                    
                    self.board[temp_ty, temp_tx] = original_piece_at_source 
                    self.board[temp_fy, temp_fx] = EMPTY

                    faces_opponent_king_directly = False
                    if temp_tx == op_king_x: 
                        intervening = 0
                        m_y, M_y = min(temp_ty, op_king_y), max(temp_ty, op_king_y)
                        for i_check_y in range(m_y + 1, M_y):
                            if self.board[i_check_y, temp_tx] != EMPTY:
                                intervening +=1
                                break
                        if intervening == 0:
                            faces_opponent_king_directly = True
                    
                    self.board[temp_fy, temp_fx] = original_piece_at_source
                    self.board[temp_ty, temp_tx] = original_piece_at_target

                    if not faces_opponent_king_directly:
                        potential_moves_after_flying_general_check.append(move_candidate)
                moves = potential_moves_after_flying_general_check
        
        return moves

    def _get_advisor_moves(self, x: int, y: int, player: Player) -> List[Move]:
        """Generates legal moves for an Advisor at (x,y) for the given player."""
        moves: List[Move] = []
        piece_at_src = self.board[y,x] 

        deltas = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        for dx, dy in deltas:
            nx, ny = x + dx, y + dy

            if self.is_valid_coord(nx, ny) and self._is_within_palace(nx, ny, player):
                target_piece = self.board[ny, nx]
                if target_piece == EMPTY or np.sign(target_piece) != np.sign(piece_at_src):
                    moves.append((x, y, nx, ny))
        return moves

    def _get_elephant_moves(self, x: int, y: int, player: Player) -> List[Move]:
        """Generates legal moves for an Elephant at (x,y) for the given player."""
        moves: List[Move] = []
        piece_at_src = self.board[y,x]

        deltas = [(2, 2), (2, -2), (-2, 2), (-2, -2)]
        eye_deltas = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        for i in range(len(deltas)):
            dx, dy = deltas[i]
            edx, edy = eye_deltas[i]

            nx, ny = x + dx, y + dy
            eye_x, eye_y = x + edx, y + edy

            is_red_side = (0 <= ny <= 4)
            is_black_side = (5 <= ny <= 9)

            if player == Player.RED and not is_red_side:
                continue 
            if player == Player.BLACK and not is_black_side:
                continue 

            if self.is_valid_coord(nx, ny):
                if self.is_valid_coord(eye_x, eye_y) and self.board[eye_y, eye_x] == EMPTY:
                    target_piece = self.board[ny, nx]
                    if target_piece == EMPTY or np.sign(target_piece) != np.sign(piece_at_src):
                        moves.append((x, y, nx, ny))
        return moves

    def _get_horse_moves(self, x: int, y: int, player: Player) -> List[Move]:
        """Generates legal moves for a Horse at (x,y) for the given player."""
        moves: List[Move] = []
        piece_at_src = self.board[y,x] 

        potential_moves_horse = [ # Renamed to avoid conflict
            (1, 2, 0, 1), (-1, 2, 0, 1), 
            (1, -2, 0, -1), (-1, -2, 0, -1), 
            (2, 1, 1, 0), (2, -1, 1, 0),  
            (-2, 1, -1, 0), (-2, -1, -1, 0)  
        ]

        for dx, dy, leg_dx, leg_dy in potential_moves_horse:
            nx, ny = x + dx, y + dy
            leg_x, leg_y = x + leg_dx, y + leg_dy 

            if self.is_valid_coord(nx, ny):
                if self.is_valid_coord(leg_x, leg_y) and self.board[leg_y, leg_x] == EMPTY:
                    target_piece = self.board[ny, nx]
                    if target_piece == EMPTY or np.sign(target_piece) != np.sign(piece_at_src):
                        moves.append((x, y, nx, ny))
        return moves

    def _get_chariot_moves(self, x: int, y: int, player: Player) -> List[Move]:
        """Generates legal moves for a Chariot at (x,y) for the given player."""
        moves: List[Move] = []
        piece_at_src = self.board[y,x] 
        src_sign = np.sign(piece_at_src)

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for dx, dy in directions:
            for i in range(1, max(BOARD_WIDTH, BOARD_HEIGHT)): 
                nx, ny = x + i * dx, y + i * dy

                if not self.is_valid_coord(nx, ny):
                    break 

                target_piece = self.board[ny, nx]
                if target_piece == EMPTY:
                    moves.append((x, y, nx, ny))
                else:
                    if np.sign(target_piece) != src_sign:
                        moves.append((x, y, nx, ny))
                    break 
        return moves

    def _get_cannon_moves(self, x: int, y: int, player: Player) -> List[Move]:
        """Generates legal moves for a Cannon at (x,y) for the given player."""
        moves: List[Move] = []
        piece_at_src = self.board[y,x]
        src_sign = np.sign(piece_at_src)

        directions_cannon = [(1, 0), (-1, 0), (0, 1), (0, -1)] # Renamed

        for dx, dy in directions_cannon:
            has_jumped_one_piece = False 
            for i in range(1, max(BOARD_WIDTH, BOARD_HEIGHT)):
                nx, ny = x + i * dx, y + i * dy

                if not self.is_valid_coord(nx, ny):
                    break 

                target_piece = self.board[ny, nx]

                if not has_jumped_one_piece:
                    if target_piece == EMPTY:
                        moves.append((x, y, nx, ny))
                    else:
                        has_jumped_one_piece = True
                else:
                    if target_piece != EMPTY:
                        if np.sign(target_piece) != src_sign:
                            moves.append((x, y, nx, ny))
                        break 
        return moves

    def _get_soldier_moves(self, x: int, y: int, player: Player) -> List[Move]:
        """Generates legal moves for a Soldier at (x,y) for the given player."""
        moves: List[Move] = []
        piece_at_src = self.board[y,x] 
        src_sign = np.sign(piece_at_src)

        potential_deltas: List[Tuple[int,int]] = []

        if player == Player.RED:
            potential_deltas.append((0, 1))
            if y >= 5:
                potential_deltas.append((1, 0))  
                potential_deltas.append((-1, 0)) 
        else: # BLACK
            potential_deltas.append((0, -1))
            if y <= 4:
                potential_deltas.append((1, 0))  
                potential_deltas.append((-1, 0)) 

        for dx, dy in potential_deltas:
            nx, ny = x + dx, y + dy

            if self.is_valid_coord(nx, ny):
                target_piece = self.board[ny, nx]
                if target_piece == EMPTY or np.sign(target_piece) != src_sign:
                    moves.append((x, y, nx, ny))
        return moves

    def get_legal_moves_for_piece(self, x: int, y: int) -> List[Move]:
        """ 
        Generates all pseudo-legal moves for a single piece at coordinates (x,y).
        Does not check for checks against the own king yet.
        """
        piece = self.board[y,x]
        if piece == EMPTY:
            return []

        player_of_piece = Player.RED if piece > 0 else Player.BLACK

        if abs(piece) == R_KING: 
            return self._get_king_moves(x, y, player_of_piece)
        elif abs(piece) == R_ADVISOR:
            return self._get_advisor_moves(x, y, player_of_piece)
        elif abs(piece) == R_ELEPHANT:
            return self._get_elephant_moves(x, y, player_of_piece)
        elif abs(piece) == R_HORSE:
            return self._get_horse_moves(x, y, player_of_piece)
        elif abs(piece) == R_CHARIOT:
            return self._get_chariot_moves(x, y, player_of_piece)
        elif abs(piece) == R_CANNON:
            return self._get_cannon_moves(x, y, player_of_piece)
        elif abs(piece) == R_SOLDIER:
            return self._get_soldier_moves(x, y, player_of_piece)
        else:
            return [] 

    def _find_king(self, player: Player, board_state: Optional[Board] = None) -> Optional[Tuple[int, int]]:
        """Finds the coordinates (x,y) of the specified player's king."""
        b = board_state if board_state is not None else self.board
        king_piece = R_KING if player == Player.RED else B_KING
        for r_idx in range(BOARD_HEIGHT):
            for c_idx in range(BOARD_WIDTH):
                if b[r_idx,c_idx] == king_piece:
                    return c_idx, r_idx
        return None 

    def is_square_attacked_by(self, x: int, y: int, attacker_player: Player, board_state: Optional[Board] = None) -> bool:
        """
        Checks if the square (x,y) is attacked by any piece of attacker_player.
        Uses pseudo-legal moves (ignores checks on the attacker's king).
        If board_state is provided, uses that; otherwise, uses current game board.
        """
        target_board = board_state if board_state is not None else self.board

        # --- New block for Flying General Check ---
        # The "Flying General" rule is a form of attack. If the square (x,y) contains the
        # defending king, and it is on the same file as the attacking king with no
        # intervening pieces, it is considered attacked.
        defending_player = self.get_opponent(attacker_player)
        defending_king_pos = self._find_king(defending_player, target_board)

        # Check if the square we're interested in (x,y) actually holds the defending king.
        # This logic is only relevant for checking if a KING is being attacked by the FLYING GENERAL rule.
        if defending_king_pos and defending_king_pos == (x,y):
            attacker_king_pos = self._find_king(attacker_player, target_board)
            if attacker_king_pos:
                ak_x, ak_y = attacker_king_pos
                dk_x, dk_y = defending_king_pos # This is just x,y

                if ak_x == dk_x: # They are on the same file
                    intervening_pieces = 0
                    min_y, max_y = min(ak_y, dk_y), max(ak_y, dk_y)
                    for i_y in range(min_y + 1, max_y):
                        if target_board[i_y, ak_x] != EMPTY:
                            intervening_pieces += 1
                            break
                    if intervening_pieces == 0:
                        return True # The square is attacked by the Flying General rule.
        # --- End of New block ---

        # Original logic for checking attacks from other pieces
        for py in range(BOARD_HEIGHT):
            for px in range(BOARD_WIDTH):
                pseudo_piece_val = target_board[py, px]
                if pseudo_piece_val != EMPTY:
                    piece_player_for_pseudo_moves = Player.RED if pseudo_piece_val > 0 else Player.BLACK
                    if piece_player_for_pseudo_moves == attacker_player:
                        # We don't need to check King moves again here, as that would be a normal king move (1 step)
                        # and the special Flying General case is now handled above.
                        if abs(pseudo_piece_val) == R_KING:
                            continue

                        # Generate pseudo-legal moves for the attacker's piece
                        pseudo_moves: List[Move] = []
                        if abs(pseudo_piece_val) == R_ADVISOR:
                            pseudo_moves = self._get_advisor_moves(px, py, attacker_player)
                        elif abs(pseudo_piece_val) == R_ELEPHANT:
                            pseudo_moves = self._get_elephant_moves(px, py, attacker_player)
                        elif abs(pseudo_piece_val) == R_HORSE:
                            pseudo_moves = self._get_horse_moves(px, py, attacker_player)
                        elif abs(pseudo_piece_val) == R_CHARIOT:
                            pseudo_moves = self._get_chariot_moves(px, py, attacker_player)
                        elif abs(pseudo_piece_val) == R_CANNON:
                            pseudo_moves = self._get_cannon_moves(px, py, attacker_player)
                        elif abs(pseudo_piece_val) == R_SOLDIER:
                            pseudo_moves = self._get_soldier_moves(px, py, attacker_player)
                        
                        for _, _, p_tx, p_ty in pseudo_moves:
                            if p_tx == x and p_ty == y:
                                return True # The square is attacked
        return False

    def is_king_in_check(self, player: Player, board_state: Optional[Board] = None) -> bool:
        """Checks if the specified player's king is currently in check."""
        b = board_state if board_state is not None else self.board

        king_coords = self._find_king(player, b)
        if not king_coords:
            return True 
        
        king_x, king_y = king_coords
        opponent = self.get_opponent(player)
        
        return self.is_square_attacked_by(king_x, king_y, opponent, b)

    def get_all_legal_moves(self, player: Player) -> List[Move]:
        """
        Generates all fully legal moves for the specified player.
        A move is legal if it does not leave the player's own king in check.
        """
        legal_moves: List[Move] = []
        current_player_pieces = []

        for r_idx in range(BOARD_HEIGHT):
            for c_idx in range(BOARD_WIDTH):
                piece = self.board[r_idx,c_idx]
                if piece != EMPTY and (Player.RED if piece > 0 else Player.BLACK) == player:
                    current_player_pieces.append((c_idx,r_idx)) 

        for x, y in current_player_pieces:
            pseudo_legal_moves = self.get_legal_moves_for_piece(x, y)
            
            for move_candidate in pseudo_legal_moves:
                fx, fy, tx, ty = move_candidate
                
                temp_board = self.board.copy()
                
                piece_to_move = temp_board[fy, fx]
                temp_board[ty, tx] = piece_to_move
                temp_board[fy, fx] = EMPTY
                
                # First check: Does this move leave my own king in check?
                if not self.is_king_in_check(player, temp_board):
                    # Second check: Does this move allow opponent to deliver checkmate next turn?
                    if not self._move_allows_opponent_checkmate(move_candidate, player):
                        legal_moves.append(move_candidate)
        
        # print(f"Legal moves: {legal_moves}")
        return legal_moves

    def apply_move(self, move: Move):
        """
        Applies a given move to the board.
        Assumes the move is legal.
        Updates the board state, current player, and move history.
        """
        fx, fy, tx, ty = move
        
        piece_to_move = self.board[fy, fx]
        
        if piece_to_move == EMPTY:
            print(f"Warning: Attempting to move an empty square from ({fx},{fy}) in apply_move.")
            return

        self.board[ty, tx] = piece_to_move
        self.board[fy, fx] = EMPTY
        
        self.move_history.append(move)
        
        # Track the move and player for perpetual check/chase detection
        player_who_made_move = self.current_player  # Current player before switch
        self.move_sequence.append((move, player_who_made_move))
        
        self.current_player = self.get_opponent(self.current_player)
        
        self.halfmove_clock += 1 
        if self.current_player == Player.RED: 
            self.fullmove_number += 1

        # Update position history *after* the move and player switch
        self._update_position_history()

    def _get_position_hash(self) -> str:
        """
        Creates a simple hashable representation of the current game state.
        Includes board state and current player to move.
        """
        # Convert board to a tuple of tuples to make it hashable, then to string
        board_tuple_str = str(tuple(map(tuple, self.board)))
        player_str = self.current_player.name
        # The combination of board and player to move defines a "position" for repetition purposes.
        return f"{board_tuple_str}|{player_str}"

    def _update_position_history(self):
        """Updates the history of positions for repetition checking."""
        pos_hash = self._get_position_hash()
        self.position_history[pos_hash] += 1
        self.position_sequence.append(pos_hash)

    def _is_piece_attacking_target(self, piece_pos: Tuple[int, int], target_pos: Tuple[int, int], 
                                   board_state: Optional[Board] = None) -> bool:
        """
        Checks if a piece at piece_pos is attacking/threatening the target at target_pos.
        """
        px, py = piece_pos
        tx, ty = target_pos
        b = board_state if board_state is not None else self.board
        
        piece = b[py, px]
        if piece == EMPTY:
            return False
            
        piece_player = Player.RED if piece > 0 else Player.BLACK
        
        # Temporarily set board state for move generation
        original_board = self.board
        self.board = b
        moves = self.get_legal_moves_for_piece(px, py)
        self.board = original_board
        
        for _, _, mx, my in moves:
            if mx == tx and my == ty:
                return True
        return False

    def _detect_perpetual_check(self, lookback_moves: int = 12) -> Tuple[bool, Optional[Player]]:
        """
        Detects perpetual check patterns according to official Elephant Chess rules.
        Returns (is_perpetual_check, checking_player).
        
        A perpetual check is when:
        1. The same position appears 3 times
        2. The repetition is caused by one player continuously giving check
        3. The checking player must find a different move or lose
        """
        if len(self.position_sequence) < 6:  # Need at least 3 repetitions
            return False, None
            
        current_pos = self.position_sequence[-1]
        if self.position_history[current_pos] < 3:
            return False, None
            
        # Find the positions in the sequence where this position occurred
        occurrence_indices = []
        for i, pos in enumerate(self.position_sequence):
            if pos == current_pos:
                occurrence_indices.append(i)
                
        if len(occurrence_indices) < 3:
            return False, None
            
        # Check if the repetitions are caused by continuous checking
        # Look at the moves between repetitions
        last_three_occurrences = occurrence_indices[-3:]
        
        # Check if opponent king was in check at each occurrence
        checking_player = None
        continuous_check = True
        check_count = 0
        
        for occ_idx in last_three_occurrences:
            # Reconstruct board state at this position
            if occ_idx < len(self.move_sequence):
                # Check if either king was in check at this position
                # We need to determine who was giving check
                for player in [Player.RED, Player.BLACK]:
                    if self.is_king_in_check(player):
                        potential_checking_player = self.get_opponent(player)
                        if checking_player is None:
                            checking_player = potential_checking_player
                            check_count += 1
                        elif checking_player == potential_checking_player:
                            check_count += 1
                        else:
                            continuous_check = False
                            break
                        break
                if not continuous_check:
                    break
                    
        # Perpetual check detected if same player was giving check in all repetitions
        return continuous_check and check_count >= 3 and checking_player is not None, checking_player

    def _detect_perpetual_chase(self, lookback_moves: int = 12) -> Tuple[bool, Optional[Player]]:
        """
        Detects perpetual chase patterns according to official Elephant Chess rules.
        
        A perpetual chase is when:
        1. The same position appears 3 times
        2. One player continuously threatens to capture an unprotected opponent piece
        3. The chasing player must find a different move or lose
        
        Note: Official rules focus on whether the threatened piece is protected,
        not on piece value comparisons.
        """
        if len(self.position_sequence) < 6:
            return False, None
            
        current_pos = self.position_sequence[-1]
        if self.position_history[current_pos] < 3:
            return False, None
            
        # Find occurrence indices
        occurrence_indices = []
        for i, pos in enumerate(self.position_sequence):
            if pos == current_pos:
                occurrence_indices.append(i)
                
        if len(occurrence_indices) < 3:
            return False, None
            
        # Analyze the moves between repetitions to detect chasing
        last_three_occurrences = occurrence_indices[-3:]
        
        chasing_player = None
        chased_pieces = set()  # Track which pieces are being chased
        chase_detected = False
        
        # Look at moves between the repetitions
        for i in range(len(last_three_occurrences) - 1):
            start_idx = last_three_occurrences[i]
            end_idx = last_three_occurrences[i + 1]
            
            if start_idx >= len(self.move_sequence) or end_idx > len(self.move_sequence):
                continue
                
            # Analyze moves in this segment
            segment_moves = self.move_sequence[start_idx:end_idx]
            
            for move, player in segment_moves:
                fx, fy, tx, ty = move
                
                # Check if this move threatens an opponent piece
                # Look at all squares this piece can attack after the move
                piece_moves = self.get_legal_moves_for_piece(tx, ty)
                
                for _, _, threat_x, threat_y in piece_moves:
                    threatened_piece = self.board[threat_y, threat_x]
                    if (threatened_piece != EMPTY and 
                        np.sign(threatened_piece) != np.sign(self.board[ty, tx])):
                        
                        # Check if this is a valid chase according to official rules
                        # Official rules focus on whether the piece is protected, not piece values
                        # A chase occurs when continuously threatening an unprotected piece
                        if self._is_piece_unprotected(threat_x, threat_y):
                            if chasing_player is None:
                                chasing_player = player
                                chase_detected = True
                            elif chasing_player != player:
                                return False, None  # Not consistent chasing
                            
                            chased_pieces.add((threat_x, threat_y, threatened_piece))
        
        # A chase is detected if there's consistent threatening of the same pieces
        return chase_detected and len(chased_pieces) > 0 and chasing_player is not None, chasing_player

    def _get_piece_value(self, piece: int) -> int:
        """Returns the relative value of a piece for chase detection."""
        abs_piece = abs(piece)
        values = {
            R_KING: 1000,    # King is invaluable
            R_CHARIOT: 9,    # Rook/Chariot
            R_CANNON: 4.5,   # Cannon  
            R_HORSE: 4,      # Horse/Knight
            R_ADVISOR: 2,    # Advisor
            R_ELEPHANT: 2,   # Elephant
            R_SOLDIER: 1     # Soldier/Pawn
        }
        return values.get(abs_piece, 0)

    def _detect_mutual_perpetual_check(self) -> Tuple[bool, str]:
        """
        Detects if both players are giving perpetual check.
        This results in a draw according to official rules.
        """
        red_perpetual, _ = self._detect_perpetual_check()
        black_perpetual, _ = self._detect_perpetual_check()
        
        if red_perpetual and black_perpetual:
            return True, "mutual_perpetual_check"
        return False, ""

    def _detect_check_vs_chase(self) -> Tuple[bool, Optional[Player]]:
        """
        Detects the special case where one player gives perpetual check
        while the other gives perpetual chase.
        According to rules: the chasing player loses.
        """
        is_check, checking_player = self._detect_perpetual_check()
        is_chase, chasing_player = self._detect_perpetual_chase()
        
        if is_check and is_chase and checking_player != chasing_player:
            # The chasing player loses
            return True, self.get_opponent(chasing_player)
        return False, None

    def check_game_over(self) -> Tuple[Optional[str], Optional[Player]]:
        """
        Checks if the game has ended due to checkmate, stalemate, or perpetual check/chase.
        Returns a tuple: (status_string or None, winner_player_enum or None).
        Status can be "checkmate", "stalemate", "perpetual_check", "perpetual_chase", 
        "mutual_perpetual_check", "check_vs_chase", "draw_by_repetition".
        Winner is the player who won, or None if draw/ongoing.
        """
        # Check for mutual perpetual check first (draw)
        is_mutual_check, status = self._detect_mutual_perpetual_check()
        if is_mutual_check:
            return status, None
            
        # Check for check vs chase scenario
        is_check_vs_chase, winner = self._detect_check_vs_chase()
        if is_check_vs_chase:
            return "check_vs_chase", winner
        
        # Check for perpetual check (checking player loses)
        is_perpetual_check, checking_player = self._detect_perpetual_check()
        if is_perpetual_check:
            # The player giving perpetual check loses
            return "perpetual_check", self.get_opponent(checking_player)
        
        # Check for perpetual chase (chasing player loses)
        is_perpetual_chase, chasing_player = self._detect_perpetual_chase()
        if is_perpetual_chase:
            # The player giving perpetual chase loses
            return "perpetual_chase", self.get_opponent(chasing_player)
        
        # NOTE: Regular threefold repetition is now claim-based, not automatic
        # Use can_claim_draw_by_repetition() and claim_draw_by_repetition() methods
        
        player = self.current_player
        legal_moves = self.get_all_legal_moves(player)

        if not legal_moves: 
            is_in_check_flag = self.is_king_in_check(player)
            if is_in_check_flag:
                return "checkmate", self.get_opponent(player)
            else:
                return "stalemate", None
        
        return None, None

    def can_claim_draw_by_repetition(self) -> bool:
        """
        Checks if the current player can claim a draw by threefold repetition.
        This follows traditional chess/Elephant Chess rules where repetition is claim-based.
        """
        current_pos_hash = self._get_position_hash()
        return self.position_history[current_pos_hash] >= 3

    def claim_draw_by_repetition(self) -> bool:
        """
        Allows the current player to claim a draw by threefold repetition.
        Returns True if claim is valid, False otherwise.
        
        Note: In tournament play, a valid threefold repetition claim cannot be refused.
        In casual play, this would typically require opponent agreement, but we 
        auto-accept valid claims here for simplicity.
        """
        return self.can_claim_draw_by_repetition()

    def offer_draw(self) -> str:
        """
        Current player offers a draw to opponent.
        Returns status message indicating the offer was made.
        The opponent would need to accept/decline in a real game interface.
        """
        return f"{self.current_player.name} offers a draw"

    def is_drawn_by_insufficient_material(self) -> bool:
        """
        Checks if the game is drawn due to insufficient material.
        In Elephant Chess, this is rare but can occur with minimal pieces.
        """
        pieces = []
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                piece = self.board[y, x]
                if piece != EMPTY and abs(piece) != R_KING:
                    pieces.append(abs(piece))
        
        # Very basic insufficient material check
        # (This could be expanded with more sophisticated rules)
        if len(pieces) == 0:  # Only kings left
            return True
        return False

    def get_repetition_count(self) -> int:
        """Returns the number of times the current position has occurred."""
        current_pos_hash = self._get_position_hash()
        return self.position_history[current_pos_hash]

    def _is_piece_unprotected(self, x: int, y: int, board_state: Optional[Board] = None) -> bool:
        """
        Checks if a piece at (x,y) is unprotected according to official Elephant Chess rules.
        A piece is unprotected if no friendly piece can capture an attacker that captures it.
        """
        b = board_state if board_state is not None else self.board
        piece = b[y, x]
        
        if piece == EMPTY:
            return True
            
        piece_player = Player.RED if piece > 0 else Player.BLACK
        
        # Check if any friendly piece can "see" this square (i.e., can move to defend it)
        for py in range(BOARD_HEIGHT):
            for px in range(BOARD_WIDTH):
                defender_piece = b[py, px]
                if (defender_piece != EMPTY and 
                    (Player.RED if defender_piece > 0 else Player.BLACK) == piece_player):
                    
                    # Check if this piece can move to defend the target square
                    # Temporarily set board for move generation
                    original_board = self.board
                    self.board = b
                    defender_moves = self.get_legal_moves_for_piece(px, py)
                    self.board = original_board
                    
                    for _, _, mx, my in defender_moves:
                        if mx == x and my == y:
                            return False  # Piece is protected
                            
        return True  # No defender found, piece is unprotected
    
    def _move_allows_opponent_checkmate(self, move: Move, player: Player) -> bool:
        """
        Optimized check if making this move would allow the opponent to deliver checkmate 
        on their next turn. This prevents "suicide" moves.
        
        Returns True if the move allows opponent checkmate (move should be rejected).
        Returns False if the move is safe (move can be allowed).
        """
        fx, fy, tx, ty = move
        
        # Create a temporary board state after applying the move
        temp_board = self.board.copy()
        piece_to_move = temp_board[fy, fx]
        temp_board[ty, tx] = piece_to_move
        temp_board[fy, fx] = EMPTY
        
        opponent = self.get_opponent(player)
        
        # Quick optimization: Only check if we're exposing our king to immediate attack
        my_king_pos = self._find_king(player, temp_board)
        if not my_king_pos:
            return False  # No king found (shouldn't happen)
        
        king_x, king_y = my_king_pos
        
        # Check if opponent can attack our king after this move
        if not self.is_square_attacked_by(king_x, king_y, opponent, temp_board):
            return False  # King not under attack, no immediate checkmate possible
        
        # King is under attack - now check if we have any legal escape moves
        # This is where we need to be careful about recursion
        
        # Generate our possible response moves without checkmate prevention (to avoid recursion)
        escape_moves = []
        
        for r_idx in range(BOARD_HEIGHT):
            for c_idx in range(BOARD_WIDTH):
                piece = temp_board[r_idx, c_idx]
                if piece != EMPTY and (Player.RED if piece > 0 else Player.BLACK) == player:
                    # Get moves for this piece
                    piece_moves = self._get_piece_moves_on_board(c_idx, r_idx, player, temp_board)
                    
                    # Check if any of these moves get us out of check
                    for escape_move in piece_moves:
                        escape_fx, escape_fy, escape_tx, escape_ty = escape_move
                        
                        # Apply escape move
                        escape_board = temp_board.copy()
                        escape_piece = escape_board[escape_fy, escape_fx]
                        escape_board[escape_ty, escape_tx] = escape_piece
                        escape_board[escape_fy, escape_fx] = EMPTY
                        
                        # Check if this gets our king out of check
                        if not self.is_king_in_check(player, escape_board):
                            return False  # We found an escape move, not checkmate
        
        # No escape moves found while king is in check = checkmate
        return True
    
    def _get_piece_moves_on_board(self, x: int, y: int, player: Player, board_state: Board) -> List[Move]:
        """Get legal moves for a piece on a specific board state (used for optimization)."""
        # Temporarily set the board to get piece moves
        original_board = self.board
        try:
            self.board = board_state
            moves = self.get_legal_moves_for_piece(x, y)
            return moves
        finally:
            self.board = original_board
    
    def get_all_legal_moves_basic(self, player: Player) -> List[Move]:
        """
        Basic legal move generation without checkmate prevention (to avoid recursion).
        Only filters moves that leave own king in check.
        """
        legal_moves: List[Move] = []
        current_player_pieces = []

        for r_idx in range(BOARD_HEIGHT):
            for c_idx in range(BOARD_WIDTH):
                piece = self.board[r_idx,c_idx]
                if piece != EMPTY and (Player.RED if piece > 0 else Player.BLACK) == player:
                    current_player_pieces.append((c_idx,r_idx)) 

        for x, y in current_player_pieces:
            pseudo_legal_moves = self.get_legal_moves_for_piece(x, y)
            
            for move_candidate in pseudo_legal_moves:
                fx, fy, tx, ty = move_candidate
                
                temp_board = self.board.copy()
                
                piece_to_move = temp_board[fy, fx]
                temp_board[ty, tx] = piece_to_move
                temp_board[fy, fx] = EMPTY
                
                # Only check if move leaves own king in check (no recursion)
                if not self.is_king_in_check(player, temp_board):
                    legal_moves.append(move_candidate)
        
        return legal_moves
    
    def get_fen(self) -> str:
        """
        Returns a simple position string for game analysis.
        This is a simplified representation for debugging purposes.
        """
        return f"board_{hash(str(self.board.tobytes()))}_{self.current_player.name}_{len(self.move_history)}"

if __name__ == '__main__':
    game = ElephantChessGame()
    print(game)

    # game_custom_fen = ElephantChessGame(fen="rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKAB1R w - - 0 1") # Example: Black king moved
    # The FEN above is slightly problematic for the simplified parser if 'R' is meant to be King. K is standard.
    # Using the standard FEN to avoid issues with the simple parser:
    game_custom_fen = ElephantChessGame(fen="rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1") 
    
    print("\nCustom FEN example (initial board):")
    game_from_fen = ElephantChessGame(INITIAL_BOARD_FEN)
    print(game_from_fen)

    print(f"Piece at (0,0): {PIECE_NAMES.get(game_from_fen.get_piece_at(0,0))}") 
    print(f"Piece at (4,0): {PIECE_NAMES.get(game_from_fen.get_piece_at(4,0))}") 
    print(f"Piece at (0,9): {PIECE_NAMES.get(game_from_fen.get_piece_at(0,9))}") 

    print(f"Piece at (4,9) (Black King pos): {PIECE_NAMES.get(game_from_fen.get_piece_at(4,9))}") 
    print(f"Piece at (4,0) (Red King pos): {PIECE_NAMES.get(game_from_fen.get_piece_at(4,0))}")   

    print(f"Piece at (0,3) (Red Soldier pos): {PIECE_NAMES.get(game_from_fen.get_piece_at(0,3))}") 
    print(f"Piece at (1,2) (Red Cannon pos): {PIECE_NAMES.get(game_from_fen.get_piece_at(1,2))}") 
    
    print(f"Initial player: {game_from_fen.current_player.name}")
    
    custom_fen_black_turn = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1"
    game_black_turn = ElephantChessGame(fen=custom_fen_black_turn)
    print(f"Player for FEN with 'b': {game_black_turn.current_player.name}")