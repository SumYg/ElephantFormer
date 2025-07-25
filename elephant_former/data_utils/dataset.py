"""PyTorch Dataset class for Elephant Chess games."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Union
from pathlib import Path

from elephant_former.data.elephant_parser import parse_iccs_pgn_file, ElephantGame
from elephant_former.data_utils.tokenization_utils import generate_training_sequences_from_game, TrainingInstance
from elephant_former import constants

class ElephantChessDataset(Dataset):
    """A PyTorch Dataset for Elephant Chess games.
    
    Loads games from PGN files, tokenizes them, and prepares them as 
    (input_sequence, target_move_tuple) pairs.
    """
    def __init__(self,
                 file_paths: Optional[List[Union[str, Path]]] = None,
                 games: Optional[List[ElephantGame]] = None,
                 max_seq_len: Optional[int] = None,
                 min_game_len_moves: int = 2
                ):
        """
        Args:
            file_paths: A list of paths to PGN files containing game data.
            games: A list of pre-parsed ElephantGame objects.
            max_seq_len: Optional maximum sequence length for input sequences.
                         If provided, any generated training instance where the
                         input token sequence is longer than this value will be
                         discarded. If None, all generated sequences are kept.
                         Must be >= 1 if set.
            min_game_len_moves: Minimum number of moves (not tokens) for a game to be included
        """
        self.training_instances: List[TrainingInstance] = []
        self.max_seq_len = max_seq_len

        if self.max_seq_len is not None and self.max_seq_len < 1:
            raise ValueError("max_seq_len must be None or an integer >= 1.")

        if games is not None:
            print(f"Loading dataset from {len(games)} pre-loaded games.")
            all_parsed_games = games
        elif file_paths is not None:
            print(f"Loading and processing games from: {file_paths}")
            all_parsed_games = []
            for file_path in file_paths:
                file_path = Path(file_path)
                print(f"Processing file: {file_path.name}...")
                # Make sure parse_iccs_pgn_file returns List[ElephantGame]
                parsed_games_from_file: List[ElephantGame] = parse_iccs_pgn_file(file_path)
                all_parsed_games.extend(parsed_games_from_file)
            print(f"Finished processing. Parsed {len(all_parsed_games)} games in total.")
        else:
            raise ValueError("Either file_paths or games must be provided to ElephantChessDataset.")

        if not all_parsed_games:
            print("Warning: No games were loaded or provided to the dataset.")
            return

        for i, game in enumerate(all_parsed_games):
            # game.parsed_moves should be used by generate_training_sequences_from_game implicitly
            if len(game.parsed_moves) < min_game_len_moves:
                 # print(f"Skipping game {i+1} with {len(game.parsed_moves)} moves (less than min {min_game_len_moves}). Metadata: {game.metadata.get('Event', 'N/A')}")
                continue
            
            game_instances = generate_training_sequences_from_game(game)
            for input_ids, target_coords in game_instances:
                if self.max_seq_len is not None and len(input_ids) > self.max_seq_len:
                    # Skip this instance as its input sequence is too long.
                    continue
                
                # Add the instance if its length is acceptable
                self.training_instances.append((input_ids, target_coords))
        
        if not self.training_instances:
            print("Warning: No training instances generated from the provided games. Check game length, content, and parsing.")
        else:
            print(f"Generated {len(self.training_instances)} training instances.")

    def __len__(self) -> int:
        """Returns the total number of training instances."""
        return len(self.training_instances)

    def __getitem__(self, idx: int) -> TrainingInstance:
        """Returns the training instance at the given index.
        
        Each instance is a tuple: (input_token_ids_sequence, target_coordinate_tuple)
        """
        return self.training_instances[idx]


def elephant_collate_fn(batch: List[TrainingInstance]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Collate function for preparing batches of Elephant Chess data.

    Args:
        batch: A list of TrainingInstance objects, where each instance is 
               (input_token_ids_sequence, target_coordinate_tuple).

    Returns:
        A tuple containing:
        - padded_input_sequences (torch.Tensor): Tensor of input sequences, padded to 
                                                 the max length in the batch, 
                                                 shape (batch_size, max_seq_len).
        - targets (Tuple[torch.Tensor, ...]): A tuple of four tensors 
                                              (target_fx, target_fy, target_tx, target_ty),
                                              each of shape (batch_size).
    """
    input_sequences, target_coords_list = zip(*batch)

    # Convert sequences to tensors and pad them
    # sequences_as_tensors = [torch.tensor(s, dtype=torch.long) for s in input_sequences]
    # Using a list comprehension for creating tensors first
    sequences_as_tensors = []
    for s in input_sequences:
        # Ensure all elements in s are integers before converting to tensor
        if not all(isinstance(item, int) for item in s):
            raise ValueError(f"Input sequence contains non-integer elements: {s}")
        sequences_as_tensors.append(torch.tensor(s, dtype=torch.long))

    padded_input_sequences = torch.nn.utils.rnn.pad_sequence(
        sequences_as_tensors, 
        batch_first=True, 
        padding_value=constants.PAD_TOKEN_ID
    )

    # Separate target coordinates and convert them to tensors
    target_fx, target_fy, target_tx, target_ty = zip(*target_coords_list)
    
    targets = (
        torch.tensor(target_fx, dtype=torch.long),
        torch.tensor(target_fy, dtype=torch.long),
        torch.tensor(target_tx, dtype=torch.long),
        torch.tensor(target_ty, dtype=torch.long)
    )

    return padded_input_sequences, targets


if __name__ == '__main__':
    # This assumes you have a sample PGN file in your 'data' directory
    # For example, create a small sample file named 'sample_games.pgn'
    # Or use one of the files you downloaded, e.g., 'data/WXF-41743games.pgns'
    # but be mindful that loading very large files can take time.

    # Create a dummy PGN file for testing if it doesn't exist
    sample_pgn_path = Path("data/sample_test_games.pgn")
    if not sample_pgn_path.exists():
        sample_pgn_path.parent.mkdir(parents=True, exist_ok=True)
        sample_game_content = """
[Game "Test Game 1"]
[Event "-"]
[Site "-"]
[Date "2023.01.01"]
[Round "-"]
[Red "Player A"]
[Black "Player B"]
[Result "1-0"]
[FEN "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"]

1. H2-E2 C9-E7
2. E2-D2 H9-G7
1-0

[Game "Test Game 2"]
[Event "-"]
[Site "-"]
[Date "2023.01.02"]
[Round "-"]
[Red "Player C"]
[Black "Player D"]
[Result "0-1"]
[FEN "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"]

1. A0-A1 B0-B1
2. A1-A2 B1-B2
0-1
        """
        with open(sample_pgn_path, 'w', encoding='utf-8') as f:
            f.write(sample_game_content)
        print(f"Created dummy PGN: {sample_pgn_path}")

    file_paths = [sample_pgn_path]
    # Set max_seq_len, e.g. 50 tokens (approx 12 moves + start token)
    # Each move is 4 tokens, plus 1 start token. Max moves in a game can be ~100-150.
    # (1 start + N moves * 4 tokens/move)
    dataset = ElephantChessDataset(file_paths=file_paths, max_seq_len=60) 

    print(f"\nDataset length: {len(dataset)}")
    if len(dataset) > 0:
        print("\nSample instance (0):")
        input_seq, target_coords = dataset[0]
        input_tokens = [constants.id_to_token[tid] for tid in input_seq]
        print(f"  Input ({len(input_seq)} tokens): {input_tokens}")
        print(f"  Target: {target_coords}")

        if len(dataset) > 2:
            print("\nSample instance (2):")
            input_seq_2, target_coords_2 = dataset[2]
            input_tokens_2 = [constants.id_to_token[tid] for tid in input_seq_2]
            print(f"  Input ({len(input_seq_2)} tokens): {input_tokens_2}")
            print(f"  Target: {target_coords_2}")

    print("\nTesting collate_fn:")
    if len(dataset) >= 2:
        # Create a small batch from the dataset
        sample_batch = [dataset[i] for i in range(min(2, len(dataset)))] # Take first 2 samples for batch
        
        # Manually create a batch with different lengths for more robust testing
        # Instance 0: len 1, Target (7,2,4,2)
        # Instance 1: len 1+4=5, Target (2,9,4,7) (from H2-E2, C9-E7)
        manual_batch = [
            ([constants.START_TOKEN_ID], (7,2,4,2)), # len 1
            ([constants.START_TOKEN_ID, 10,11,12,13], (2,9,4,7)) # len 5, dummy token IDs
        ]
        if len(dataset) >=4:
             manual_batch = [dataset[0], dataset[1], dataset[3]] # lengths 1, 5, 13 (approx)

        print(f"Manual batch items (lengths): {[len(item[0]) for item in manual_batch]}")

        padded_inputs, (tgt_fx, tgt_fy, tgt_tx, tgt_ty) = elephant_collate_fn(manual_batch)

        print("\nPadded Inputs Tensor Shape:", padded_inputs.shape)
        print("Padded Inputs Tensor (first 2 rows if batch > 1):\n", padded_inputs[:2])
        print("\nTarget Fx Tensor Shape:", tgt_fx.shape, "Values:", tgt_fx)
        print("Target Fy Tensor Shape:", tgt_fy.shape, "Values:", tgt_fy)
        print("Target Tx Tensor Shape:", tgt_tx.shape, "Values:", tgt_tx)
        print("Target Ty Tensor Shape:", tgt_ty.shape, "Values:", tgt_ty)

        # Example of DataLoader usage (uncomment to test if desired)
        from torch.utils.data import DataLoader
        test_dataloader = DataLoader(
            dataset, 
            batch_size=4, # Small batch size for testing
            shuffle=False, # No need to shuffle for this test
            collate_fn=elephant_collate_fn
        )

        print("\nTesting DataLoader output (first batch):")
        try:
            first_batch_inputs, first_batch_targets = next(iter(test_dataloader))
            print("Batch Inputs Shape:", first_batch_inputs.shape)
            print("Batch Target Fx Shape:", first_batch_targets[0].shape)
            print("First Padded Input Sequence in Batch:\n", first_batch_inputs[0])
            print("First Target Fx in Batch:", first_batch_targets[0][0])
        except Exception as e:
            print(f"Error during DataLoader iteration: {e}")
            print("This might happen if the dataset is smaller than the batch size.")
            print(f"Dataset size: {len(dataset)}, Batch size: 4")

    # Example of how it might be used with a DataLoader (conceptual)
    # from torch.utils.data import DataLoader
    # def collate_fn(batch):
    #     # Basic collate: list of tuples to tuple of lists, then convert to tensor
    #     # More advanced: padding sequences to max length in batch
    #     input_sequences, target_coords_list = zip(*batch)
    #     # Padded_input_sequences = torch.nn.utils.rnn.pad_sequence(
    #     # [torch.tensor(s) for s in input_sequences], batch_first=True, padding_value=constants.PAD_TOKEN_ID
    #     # )
    #     # target_fx, target_fy, target_tx, target_ty = zip(*target_coords_list)
    #     # targets = (
    #     # torch.tensor(target_fx, dtype=torch.long),
    #     # torch.tensor(target_fy, dtype=torch.long),
    #     # torch.tensor(target_tx, dtype=torch.long),
    #     # torch.tensor(target_ty, dtype=torch.long)
    #     # )
    #     # return Padded_input_sequences, targets
    #     pass # Implement proper collate_fn later
    
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    # For now, just showing dataset creation and direct access. 