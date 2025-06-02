"""PyTorch Lightning module for ElephantFormer."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Tuple, Dict, Any, Optional

from elephant_former.models.transformer_model import ElephantFormerGPT, generate_square_subsequent_mask
from elephant_former import constants

class LightningElephantFormer(pl.LightningModule):
    def __init__(self,
                 # Model HParams (passed to ElephantFormerGPT)
                 vocab_size: int = constants.UNIFIED_VOCAB_SIZE,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_seq_len: int = 512, 
                 # Training HParams
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01):
        super().__init__()
        self.save_hyperparameters() # Saves all __init__ args to self.hparams

        self.model = ElephantFormerGPT(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len
        )

        # Loss functions for each head
        self.loss_fn_fx = nn.CrossEntropyLoss() # Removed ignore_index
        self.loss_fn_fy = nn.CrossEntropyLoss() # Removed ignore_index
        self.loss_fn_tx = nn.CrossEntropyLoss() # Removed ignore_index
        self.loss_fn_ty = nn.CrossEntropyLoss() # Removed ignore_index

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(src, src_mask=src_mask, src_padding_mask=src_padding_mask)

    def _calculate_loss(
        self,
        model_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        input_sequences: torch.Tensor
    ) -> torch.Tensor:
        """Helper function to calculate loss, adapted from trainer.py logic."""
        logits_fx, logits_fy, logits_tx, logits_ty = model_outputs
        target_fx, target_fy, target_tx, target_ty = targets
        
        batch_size = logits_fx.size(0)
        actual_lengths = torch.sum(input_sequences != constants.PAD_TOKEN_ID, dim=1)
        gather_indices = (actual_lengths - 1).clamp(min=0)
        
        selected_logits_fx = logits_fx[torch.arange(batch_size), gather_indices, :]
        selected_logits_fy = logits_fy[torch.arange(batch_size), gather_indices, :]
        selected_logits_tx = logits_tx[torch.arange(batch_size), gather_indices, :]
        selected_logits_ty = logits_ty[torch.arange(batch_size), gather_indices, :]

        loss_fx = self.loss_fn_fx(selected_logits_fx, target_fx)
        loss_fy = self.loss_fn_fy(selected_logits_fy, target_fy)
        loss_tx = self.loss_fn_tx(selected_logits_tx, target_tx)
        loss_ty = self.loss_fn_ty(selected_logits_ty, target_ty)
        
        total_loss = loss_fx + loss_fy + loss_tx + loss_ty
        return total_loss

    def training_step(self, batch: Tuple[torch.Tensor, Tuple[torch.Tensor, ...]], batch_idx: int) -> torch.Tensor:
        input_sequences, targets = batch # targets is (target_fx, target_fy, target_tx, target_ty)
        
        # Create masks
        # Causal mask for decoder-style attention
        seq_len = input_sequences.size(1)
        causal_mask = generate_square_subsequent_mask(seq_len, device=self.device)
        
        # Padding mask: True where padded
        padding_mask = (input_sequences == constants.PAD_TOKEN_ID)
        if padding_mask.ndim == 1: # Ensure padding_mask has batch dimension
            padding_mask = padding_mask.unsqueeze(0)
        if torch.all(~padding_mask): # if no padding, set mask to None for transformer_encoder
             padding_mask = None

        model_outputs = self.model(src=input_sequences, src_mask=causal_mask, src_padding_mask=padding_mask)
        
        loss = self._calculate_loss(model_outputs, targets, input_sequences)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=input_sequences.size(0))
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, Tuple[torch.Tensor, ...]], batch_idx: int) -> None:
        input_sequences, targets = batch
        seq_len = input_sequences.size(1)
        causal_mask = generate_square_subsequent_mask(seq_len, device=self.device)
        padding_mask = (input_sequences == constants.PAD_TOKEN_ID)
        if padding_mask.ndim == 1:
            padding_mask = padding_mask.unsqueeze(0)
        if torch.all(~padding_mask):
             padding_mask = None
            
        model_outputs = self.model(src=input_sequences, src_mask=causal_mask, src_padding_mask=padding_mask)
        loss = self._calculate_loss(model_outputs, targets, input_sequences)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=input_sequences.size(0))
    
    # Add a_test_step for completeness, similar to validation_step
    def test_step(self, batch: Tuple[torch.Tensor, Tuple[torch.Tensor, ...]], batch_idx: int) -> None:
        input_sequences, targets = batch
        seq_len = input_sequences.size(1)
        causal_mask = generate_square_subsequent_mask(seq_len, device=self.device)
        padding_mask = (input_sequences == constants.PAD_TOKEN_ID)
        if padding_mask.ndim == 1:
            padding_mask = padding_mask.unsqueeze(0)
        if torch.all(~padding_mask):
             padding_mask = None

        model_outputs = self.model(src=input_sequences, src_mask=causal_mask, src_padding_mask=padding_mask)
        loss = self._calculate_loss(model_outputs, targets, input_sequences)
        
        self.log('test_loss', loss, on_epoch=True, logger=True, batch_size=input_sequences.size(0))

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        # Optionally, add a learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        return {"optimizer": optimizer}


if __name__ == '__main__':
    # Basic test to ensure the LightningModule can be instantiated
    print("Testing LightningElephantFormer instantiation...")
    module = LightningElephantFormer(
        d_model=64, nhead=2, num_encoder_layers=1, dim_feedforward=128, max_seq_len=50 # Small params for test
    )
    print("Model Hyperparameters:", module.hparams)
    print("Model Architecture:", module.model)
    print("LightningModule instantiated successfully.")

    # To run a full training test, you'd need a dummy DataLoader and a Trainer
    # from elephant_former.data_utils.dataset import ElephantChessDataset, elephant_collate_fn
    # from torch.utils.data import DataLoader

    # print("\nSetting up dummy data for a training step test...")
    # # Create a dummy PGN file for testing if it doesn't exist
    # from pathlib import Path
    # sample_pgn_path = Path("data/lightning_sample_test.pgn")
    # if not sample_pgn_path.exists():
    #     sample_pgn_path.parent.mkdir(parents=True, exist_ok=True)
    #     sample_game_content = """
    # [Game "Test Game L"]
    # 1. H2-E2 C9-E7
    # 2. E2-D2 H9-G7
    # 1-0
    #     """
    #     with open(sample_pgn_path, 'w', encoding='utf-8') as f:
    #         f.write(sample_game_content)
    
    # dummy_dataset = ElephantChessDataset(file_paths=[sample_pgn_path], max_seq_len=20)
    # if len(dummy_dataset) == 0:
    #     print("Error: Dummy dataset is empty. Cannot proceed with DataLoader test.")
    # else:
    #     dummy_dataloader = DataLoader(dummy_dataset, batch_size=2, collate_fn=elephant_collate_fn)
        
    #     print(f"Dummy dataset size: {len(dummy_dataset)}")
    #     print(f"Dummy dataloader created. Number of batches: {len(dummy_dataloader)}")
        
    #     # Get a batch
    #     try:
    #         batch = next(iter(dummy_dataloader))
    #         print("Simulating a training step...")
    #         # Manually move data to device if model is on CUDA (Lightning Trainer does this automatically)
    #         # if module.device.type == 'cuda':
    #         #     batch = (batch[0].to(module.device), 
    #         #              tuple(t.to(module.device) for t in batch[1]))

    #         loss = module.training_step(batch, 0)
    #         print(f"Training step simulated. Loss: {loss.item()}")
            
    #         # Test validation step if there are batches
    #         # Note: For a real validation_step, ensure a separate val_dataloader is used.
    #         # module.validation_step(batch, 0)
    #         # print("Validation step simulated.")

    #     except StopIteration:
    #         print("Error: Dummy dataloader is empty. Check dataset and collate_fn.")
    #     except Exception as e:
    #         print(f"Error during simulated training step: {e}")
    #         import traceback
    #         traceback.print_exc()

    print("\nBasic LightningModule tests completed.") 