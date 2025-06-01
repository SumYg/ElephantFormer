# elephant_former/constants.py

# Board dimensions (0-indexed)
# Xiangqi board is 9 columns (0-8) by 10 rows (0-9)
MAX_X = 8  # Max column index
MAX_Y = 9  # Max row index

# Special Tokens
PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
UNK_TOKEN = "<unk>" # For completeness, though may not be used if all inputs are structured

SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, UNK_TOKEN]

# --- Unified Input Vocabulary for the Transformer's Embedding Layer ---
# Each component of a move (fx, fy, tx, ty) will have its own set of unique tokens.
# Example: "from_x coordinate 0" is a different token than "to_x coordinate 0".

# Component token prefixes to ensure uniqueness in the unified vocabulary
FROM_X_PREFIX = "fx_"
FROM_Y_PREFIX = "fy_"
TO_X_PREFIX = "tx_"
TO_Y_PREFIX = "ty_"

# Generate component tokens
fx_tokens = [f"{FROM_X_PREFIX}{i}" for i in range(MAX_X + 1)]
fy_tokens = [f"{FROM_Y_PREFIX}{i}" for i in range(MAX_Y + 1)]
tx_tokens = [f"{TO_X_PREFIX}{i}" for i in range(MAX_X + 1)]
ty_tokens = [f"{TO_Y_PREFIX}{i}" for i in range(MAX_Y + 1)]

# Unified vocabulary list
UNIFIED_VOCAB_LIST = SPECIAL_TOKENS + fx_tokens + fy_tokens + tx_tokens + ty_tokens

# Mappings for the unified vocabulary
token_to_id = {token: i for i, token in enumerate(UNIFIED_VOCAB_LIST)}
id_to_token = {i: token for i, token in enumerate(UNIFIED_VOCAB_LIST)}

UNIFIED_VOCAB_SIZE = len(UNIFIED_VOCAB_LIST)

PAD_TOKEN_ID = token_to_id[PAD_TOKEN]
START_TOKEN_ID = token_to_id[START_TOKEN]
UNK_TOKEN_ID = token_to_id[UNK_TOKEN]


# --- Component-Specific Vocabularies (for model output heads) ---
# These are simpler, representing the direct values 0-8 or 0-9.
# The model's output heads will predict an integer in these ranges.

# from_x: 0 to MAX_X (9 values)
NUM_FROM_X_CLASSES = MAX_X + 1
# from_y: 0 to MAX_Y (10 values)
NUM_FROM_Y_CLASSES = MAX_Y + 1
# to_x: 0 to MAX_X (9 values)
NUM_TO_X_CLASSES = MAX_X + 1
# to_y: 0 to MAX_Y (10 values)
NUM_TO_Y_CLASSES = MAX_Y + 1


if __name__ == '__main__':
    print(f"Unified Vocabulary Size: {UNIFIED_VOCAB_SIZE}")
    print(f"PAD ID: {PAD_TOKEN_ID}, START ID: {START_TOKEN_ID}, UNK ID: {UNK_TOKEN_ID}")
    # print("\nUnified Token to ID mapping:")
    # for token, tid in token_to_id.items():
    #     print(f"'{token}': {tid}")

    print(f"\nNumber of classes for output heads:")
    print(f"  From X: {NUM_FROM_X_CLASSES}")
    print(f"  From Y: {NUM_FROM_Y_CLASSES}")
    print(f"  To X: {NUM_TO_X_CLASSES}")
    print(f"  To Y: {NUM_TO_Y_CLASSES}")

    # Example: How to get the unified token ID for from_x = 0
    fx0_token = f"{FROM_X_PREFIX}0"
    fx0_id = token_to_id[fx0_token]
    print(f"\nToken for from_x=0 is '{fx0_token}', ID: {fx0_id}")

    # Example: How a move (0,1) -> (2,3) would be tokenized for input sequence
    move_fx, move_fy, move_tx, move_ty = 0, 1, 2, 3
    input_token_ids = [
        token_to_id[f"{FROM_X_PREFIX}{move_fx}"],
        token_to_id[f"{FROM_Y_PREFIX}{move_fy}"],
        token_to_id[f"{TO_X_PREFIX}{move_tx}"],
        token_to_id[f"{TO_Y_PREFIX}{move_ty}"]
    ]
    print(f"Move ({move_fx},{move_fy}) -> ({move_tx},{move_ty}) tokenized to IDs: {input_token_ids}")
    print(f"Corresponding tokens: {[id_to_token[tid] for tid in input_token_ids]}")

    # The target for this move (for the 4 output heads) would be:
    # Head_fx target: 0
    # Head_fy target: 1
    # Head_tx target: 2
    # Head_ty target: 3
    # These are direct indices for the component-specific vocabularies. 