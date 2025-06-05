# elephant_former/engine/elephant_chess_game.py

import numpy as np
from typing import Tuple, List, Optional
from enum import Enum

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

    def __str__(self) -> str:
        """String representation of the board for printing."""
        s = "  +------------------+\n"
        for y_idx in range(BOARD_HEIGHT -1, -1, -1): # Print from y=9 down to y=0
            s += f"{y_idx} |"
            for x_idx in range(BOARD_WIDTH):
                piece = self.board[y_idx, x_idx]
                # Simple character representation (can be improved)
                char_repr = PIECE_NAMES[piece][0] if piece != EMPTY else '.' # Renamed char to char_repr
                if piece < 0 and piece != EMPTY : char_repr = char_repr.lower() # black pieces lowercase
                s += f" {char_repr}"
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
        Uses pseudo-legal moves (doesn't consider if attacker's king becomes checked).
        """
        b = board_state if board_state is not None else self.board
        original_board_state = self.board
        if board_state is not None:
            self.board = board_state 

        is_attacked = False
        for r_idx in range(BOARD_HEIGHT):
            for c_idx in range(BOARD_WIDTH):
                piece = b[r_idx,c_idx]
                if piece != EMPTY and (Player.RED if piece > 0 else Player.BLACK) == attacker_player:
                    attacker_moves = self.get_legal_moves_for_piece(c_idx, r_idx)
                    for move_candidate in attacker_moves:
                        _fx, _fy, tx, ty = move_candidate
                        if tx == x and ty == y:
                            is_attacked = True
                            break
                if is_attacked:
                    break
            if is_attacked:
                break
        
        if board_state is not None:
            self.board = original_board_state 
        return is_attacked

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
                
                if not self.is_king_in_check(player, temp_board):
                    legal_moves.append(move_candidate)
        
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
        
        self.current_player = self.get_opponent(self.current_player)
        
        self.halfmove_clock += 1 
        if self.current_player == Player.RED: 
            self.fullmove_number += 1

    def check_game_over(self) -> Tuple[Optional[str], Optional[Player]]:
        """
        Checks if the game has ended due to checkmate or stalemate.
        Returns a tuple: (status_string or None, winner_player_enum or None).
        Status can be "checkmate", "stalemate".
        Winner is the player who won, or None if draw/ongoing.
        """
        player = self.current_player
        legal_moves = self.get_all_legal_moves(player)

        if not legal_moves: 
            is_in_check_flag = self.is_king_in_check(player)
            if is_in_check_flag:
                return "checkmate", self.get_opponent(player)
            else:
                return "stalemate", None
        
        # TODO: Add other draw conditions like repetition, insufficient material (less common in Elephant Chess), etc.
        return None, None

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