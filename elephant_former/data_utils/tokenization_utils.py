"""Utilities for tokenizing Elephant Chess moves for the Transformer model."""

from typing import Tuple, List, Optional
from elephant_former import constants
from elephant_former.data.elephant_parser import ElephantGame

def parse_iccs_move_to_coords(iccs_move: str) -> Optional[Tuple[int, int, int, int]]:
    """Converts a single ICCS move string (e.g., 'A0-A1') to (fx, fy, tx, ty) coordinates.

    Assumes standard ICCS coordinates:
    - Columns 'A'-'I' map to x 0-8.
    - Rows '0'-'9' map to y 0-9.

    Args:
        iccs_move: The move string in ICCS format (e.g., "H2-E2").

    Returns:
        A tuple (from_x, from_y, to_x, to_y) or None if parsing fails.
    """
    if not isinstance(iccs_move, str) or len(iccs_move) != 5 or iccs_move[2] != '-':
        # Basic validation, can be made more robust
        return None

    try:
        from_col_char = iccs_move[0].upper()
        from_row_char = iccs_move[1]
        to_col_char = iccs_move[3].upper()
        to_row_char = iccs_move[4]

        if not ('A' <= from_col_char <= 'I' and 'A' <= to_col_char <= 'I'):
            return None
        if not ('0' <= from_row_char <= '9' and '0' <= to_row_char <= '9'):
            return None

        from_x = ord(from_col_char) - ord('A')
        from_y = int(from_row_char)
        to_x = ord(to_col_char) - ord('A')
        to_y = int(to_row_char)
        
        # Validate against board dimensions from constants
        if not (0 <= from_x <= constants.MAX_X and 0 <= from_y <= constants.MAX_Y and \
                0 <= to_x <= constants.MAX_X and 0 <= to_y <= constants.MAX_Y):
            # This should ideally not happen if char checks are correct, but good for safety
            return None

        return (from_x, from_y, to_x, to_y)
    except (ValueError, IndexError):
        return None

def coords_to_unified_token_ids(coords: Tuple[int, int, int, int]) -> Optional[List[int]]:
    """Converts (fx, fy, tx, ty) coordinates to a list of 4 unified token IDs.

    Args:
        coords: A tuple (from_x, from_y, to_x, to_y).

    Returns:
        A list of 4 token IDs corresponding to the move components, or None if 
        any coordinate is out of bounds.
    """
    fx, fy, tx, ty = coords

    # Validate coordinates against board dimensions
    if not (0 <= fx <= constants.MAX_X and 0 <= fy <= constants.MAX_Y and \
            0 <= tx <= constants.MAX_X and 0 <= ty <= constants.MAX_Y):
        return None

    try:
        fx_token = f"{constants.FROM_X_PREFIX}{fx}"
        fy_token = f"{constants.FROM_Y_PREFIX}{fy}"
        tx_token = f"{constants.TO_X_PREFIX}{tx}"
        ty_token = f"{constants.TO_Y_PREFIX}{ty}"

        token_ids = [
            constants.token_to_id[fx_token],
            constants.token_to_id[fy_token],
            constants.token_to_id[tx_token],
            constants.token_to_id[ty_token]
        ]
        return token_ids
    except KeyError: # Should not happen if constants are correct and coords validated
        return None

TrainingInstance = Tuple[List[int], Tuple[int, int, int, int]]

def generate_training_sequences_from_game(game: ElephantGame) -> List[TrainingInstance]:
    """Processes a single ElephantGame and generates (input_sequence, target_move_tuple) pairs.

    Args:
        game: An ElephantGame object.

    Returns:
        A list of training instances. Each instance is a tuple containing:
        - input_token_ids: A list of unified token IDs representing the game state 
                           up to the move *before* the target move. Starts with START_TOKEN_ID.
        - target_coords: A tuple (fx, fy, tx, ty) of the move to be predicted.
    """
    training_instances: List[TrainingInstance] = []
    current_input_sequence: List[int] = [constants.START_TOKEN_ID]

    for iccs_move_str in game.parsed_moves:
        target_coords = parse_iccs_move_to_coords(iccs_move_str)
        if not target_coords:
            # print(f"Skipping invalid ICCS move: {iccs_move_str}")
            continue # Skip this move and don't use it as a target

        # The current_input_sequence predicts the target_coords
        # We make a copy for this training instance
        training_instances.append((list(current_input_sequence), target_coords))

        # Now, for the *next* iteration, the move just processed (target_coords) 
        # becomes part of the input.
        move_token_ids = coords_to_unified_token_ids(target_coords)
        if move_token_ids: # Should be true if target_coords was valid
            current_input_sequence.extend(move_token_ids)
        else:
            # This case should ideally not be reached if parse_iccs_move_to_coords 
            # and coords_to_unified_token_ids are robust and target_coords was valid.
            # If it occurs, it means we can't extend the sequence with this move, 
            # potentially corrupting subsequent training instances from this game.
            # For now, we'll stop processing this game if a valid move can't be tokenized.
            # print(f"Warning: Could not tokenize valid coords {target_coords} from move {iccs_move_str}. Stopping processing for this game.")
            break 
            
    return training_instances

if __name__ == '__main__':
    test_moves = ["H2-E2", "A0-B0", "I9-H9", "C3-C4", "b0-c2"] # Valid and one lowercase
    invalid_moves = ["H2E2", "Z0-A1", "A10-A9", "A1-B", "A1-B11"]

    print("Testing valid moves:")
    for move_str in test_moves:
        coords = parse_iccs_move_to_coords(move_str)
        print(f"'{move_str}' -> {coords}")

    print("\nTesting invalid moves:")
    for move_str in invalid_moves:
        coords = parse_iccs_move_to_coords(move_str)
        print(f"'{move_str}' -> {coords}")
    
    print("\nTesting coords_to_unified_token_ids:")
    test_coord_tuples = [
        (7, 2, 4, 2), # H2-E2
        (0, 0, 1, 0), # A0-B0
        (2, 3, 2, 4), # C3-C4
        (8, 9, 7, 9)  # I9-H9
    ]
    for coord_tuple in test_coord_tuples:
        token_ids = coords_to_unified_token_ids(coord_tuple)
        if token_ids:
            tokens = [constants.id_to_token[tid] for tid in token_ids]
            print(f"{coord_tuple} -> IDs: {token_ids} -> Tokens: {tokens}")
        else:
            print(f"{coord_tuple} -> Failed to convert to token IDs")

    invalid_coord_tuple = (0, 0, 10, 0) # Invalid to_x
    token_ids = coords_to_unified_token_ids(invalid_coord_tuple)
    print(f"{invalid_coord_tuple} -> IDs: {token_ids}")

    print("\nTesting generate_training_sequences_from_game:")
    # Create a dummy ElephantGame object for testing
    sample_metadata = {"Event": "Test Game", "Red": "PlayerA", "Black": "PlayerB", "Result": "1-0"}
    # Moves for (0,0)->(1,0), then (5,5)->(5,4)
    sample_iccs_moves_list = ["A0-B0", "F5-F4", "B0-C0"] # (0,0,1,0), (5,5,5,4), (1,0,2,0)
    # Convert list of moves to a space-separated string for iccs_moves_string
    sample_iccs_moves_string = " ".join(sample_iccs_moves_list) 
    
    dummy_game = ElephantGame(metadata=sample_metadata, iccs_moves_string=sample_iccs_moves_string)

    training_data = generate_training_sequences_from_game(dummy_game)

    for i, (input_seq, target) in enumerate(training_data):
        input_tokens = [constants.id_to_token[tid] for tid in input_seq]
        print(f"Instance {i+1}:")
        print(f"  Input Tokens: {input_tokens}")
        print(f"  Target Coords: {target}")

    # Expected output for the dummy game:
    # Instance 1:
    #   Input Tokens: ['<start>']
    #   Target Coords: (0, 0, 1, 0)  (corresponds to A0-B0)
    # Instance 2:
    #   Input Tokens: ['<start>', 'fx_0', 'fy_0', 'tx_1', 'ty_0']
    #   Target Coords: (5, 5, 5, 4)  (corresponds to F5-F4)
    # Instance 3:
    #   Input Tokens: ['<start>', 'fx_0', 'fy_0', 'tx_1', 'ty_0', 'fx_5', 'fy_5', 'tx_5', 'ty_4']
    #   Target Coords: (1, 0, 2, 0)  (corresponds to B0-C0)

    # Test with a game that has an invalid move in the middle
    sample_iccs_moves_with_invalid_list = ["A0-B0", "INVALID-MOVE", "F5-F4"]
    sample_iccs_moves_with_invalid_string = " ".join(sample_iccs_moves_with_invalid_list)
    dummy_game_invalid = ElephantGame(metadata=sample_metadata, iccs_moves_string=sample_iccs_moves_with_invalid_string)
    training_data_invalid = generate_training_sequences_from_game(dummy_game_invalid)
    print("\nTesting with a game containing an invalid move:")
    for i, (input_seq, target) in enumerate(training_data_invalid):
        input_tokens = [constants.id_to_token[tid] for tid in input_seq]
        print(f"Instance {i+1} (invalid game):")
        print(f"  Input Tokens: {input_tokens}")
        print(f"  Target Coords: {target}")
    # Expected: Only one instance, as processing stops after invalid move.
    # Instance 1 (invalid game):
    # Input Tokens: ['<start>']
    # Target Coords: (0,0,1,0)

    # Test coordinate mapping
    # Expected: H(7)2 -> E(4)2  => (7,2,4,2)
    # Expected: A(0)0 -> B(1)0  => (0,0,1,0)
    # Expected: I(8)9 -> H(7)9  => (8,9,7,9)
    # Expected: C(2)3 -> C(2)4  => (2,3,2,4)
    # Expected: b(1)0 -> c(2)2  => (1,0,2,2) 