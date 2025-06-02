"""Defines the Transformer model architecture for ElephantFormer."""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from elephant_former import constants

class ElephantFormerGPT(nn.Module):
    def __init__(self, 
                 vocab_size: int = constants.UNIFIED_VOCAB_SIZE,
                 d_model: int = 256,      # Embedding dimension
                 nhead: int = 8,          # Number of attention heads
                 num_encoder_layers: int = 6, # Number of Transformer encoder layers
                 dim_feedforward: int = 1024, # Dimension of the feedforward network in Transformer
                 dropout: float = 0.1,
                 max_seq_len: int = 512): # Max sequence length for positional encoding
        super().__init__()
        self.d_model = d_model

        # 1. Token Embedding Layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional Encoding Layer (Learned)
        self.positional_encoding = nn.Embedding(max_seq_len, d_model)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True # Crucial: our data is (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 4. Output Heads (one for each component of the move)
        self.head_fx = nn.Linear(d_model, constants.NUM_FROM_X_CLASSES)
        self.head_fy = nn.Linear(d_model, constants.NUM_FROM_Y_CLASSES)
        self.head_tx = nn.Linear(d_model, constants.NUM_TO_X_CLASSES)
        self.head_ty = nn.Linear(d_model, constants.NUM_TO_Y_CLASSES)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        # Initialize weights for linear layers and embeddings
        # This is a common practice for Transformer models
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.positional_encoding.weight.data.uniform_(-initrange, initrange)
        # For linear layers, PyTorch default init (Kaiming for weights, uniform for bias) is often good.
        # Optionally, you can customize further:
        # self.head_fx.bias.data.zero_()
        # self.head_fx.weight.data.uniform_(-initrange, initrange)
        # ... and for other heads

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            src (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len).
                                Contains token IDs.
            src_mask (Optional[torch.Tensor]): The causal mask for the Transformer encoder of shape (seq_len, seq_len).
                                               Ensures attention is only paid to previous positions.
            src_padding_mask (Optional[torch.Tensor]): Mask for padding tokens in src, shape (batch_size, seq_len).
                                                      Boolean tensor, True where padded.
        Returns:
            A tuple of four tensors (logits_fx, logits_fy, logits_tx, logits_ty).
            Each tensor contains logits for one component of the predicted move.
            The shape of each output tensor is (batch_size, seq_len, num_classes_for_component).
            However, for next-move prediction, we are typically interested in the logits 
            for the *last relevant token* in the sequence, or a specific token position.
            For this GPT-style model predicting the next move, we'll output logits for all sequence positions.
            The training loop will handle selecting the correct logits for loss calculation (usually the last one).
        """
        seq_len = src.size(1)
        if seq_len > self.positional_encoding.num_embeddings:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds max_seq_len for positional embeddings ({self.positional_encoding.num_embeddings})")

        # 1. Get token embeddings
        src_emb = self.token_embedding(src) * math.sqrt(self.d_model) # Scale embedding (common practice)
        
        # 2. Add positional encodings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=src.device).unsqueeze(0) # (1, seq_len)
        pos_emb = self.positional_encoding(positions) # (1, seq_len, d_model)
        x = src_emb + pos_emb # (batch_size, seq_len, d_model)
        x = self.dropout(x)

        # 3. Pass through Transformer Encoder
        # Convert boolean src_padding_mask to float mask if it exists, to avoid warning
        # True (padded) becomes -inf, False (not padded) becomes 0.0.
        # The nn.TransformerEncoder expects src_key_padding_mask of shape (N, L)
        # and attention_mask (src_mask for self-attention) of shape (L,L) or (N*num_heads, L, L)
        
        # The causal mask (src_mask) should already be float with -inf for masked positions.
        # The padding mask (src_padding_mask) is boolean and needs to be handled correctly by the encoder.
        # PyTorch TransformerEncoder handles boolean src_key_padding_mask directly where True indicates a key to be ignored.
        # The warning might stem from internal conversions or specific conditions.
        # Let's ensure our causal mask (src_mask) is correctly passed to the `mask` argument
        # and boolean padding mask (src_padding_mask) to `src_key_padding_mask`.
        # The warning usually occurs if `mask` is boolean and `src_key_padding_mask` is float or vice-versa.
        # Our `generate_square_subsequent_mask` creates a float mask.
        # So, the boolean `src_padding_mask` is likely the one causing the type mismatch warning if it's not what one part of the layer expects.
        # However, nn.TransformerEncoder is documented to accept boolean for src_key_padding_mask.
        # Let's stick to passing them as is, as the warning might be a bit overzealous or refer to deeper parts.
        # If it persists and causes issues, explicit conversion of src_padding_mask to float -inf/0 can be done.
        # For now, the primary contract of the API seems to be met.

        transformer_output = self.transformer_encoder(
            x, 
            mask=src_mask, # This is our causal mask (float)
            src_key_padding_mask=src_padding_mask # This is our padding mask (bool)
        )
        # transformer_output shape: (batch_size, seq_len, d_model)

        # 4. Pass through output heads
        logits_fx = self.head_fx(transformer_output)
        logits_fy = self.head_fy(transformer_output)
        logits_tx = self.head_tx(transformer_output)
        logits_ty = self.head_ty(transformer_output)
        
        # logits shapes: (batch_size, seq_len, num_component_classes)
        return logits_fx, logits_fy, logits_tx, logits_ty


# Helper function to generate a causal mask (square subsequent mask)
def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    """Generates a square causal mask for attention.
    The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

if __name__ == '__main__':
    # Example Usage and Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 4
    seq_len = 21 # e.g., <start> + 5 moves * 4 tokens/move
    
    # Create a dummy model instance
    model = ElephantFormerGPT(
        vocab_size=constants.UNIFIED_VOCAB_SIZE,
        d_model=64, # Smaller for test
        nhead=4,    # Smaller for test
        num_encoder_layers=2, # Smaller for test
        dim_feedforward=128, # Smaller for test
        max_seq_len=100 # Should be >= seq_len
    ).to(device)

    # Create dummy input data
    # src shape: (batch_size, seq_len)
    dummy_src = torch.randint(0, constants.UNIFIED_VOCAB_SIZE, (batch_size, seq_len), device=device)
    
    # Create causal mask for the source sequence
    # src_mask shape: (seq_len, seq_len)
    src_mask = generate_square_subsequent_mask(seq_len, device)

    # Create padding mask (e.g., last 3 tokens of last sequence are padding)
    # src_padding_mask shape: (batch_size, seq_len), True where padded
    src_padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    if seq_len > 3 and batch_size > 0:
         src_padding_mask[batch_size-1, -3:] = True # Last sequence, last 3 tokens are pad
    
    print(f"\nModel: {model}")
    print(f"Input src shape: {dummy_src.shape}")
    print(f"Source mask shape: {src_mask.shape}")
    print(f"Source padding mask shape: {src_padding_mask.shape}")

    # Forward pass
    try:
        logits_fx, logits_fy, logits_tx, logits_ty = model(dummy_src, src_mask=src_mask, src_padding_mask=src_padding_mask)
        print("\nOutput Logits Shapes:")
        print(f"  Fx: {logits_fx.shape}") # Expected: (batch_size, seq_len, NUM_FROM_X_CLASSES)
        print(f"  Fy: {logits_fy.shape}") # Expected: (batch_size, seq_len, NUM_FROM_Y_CLASSES)
        print(f"  Tx: {logits_tx.shape}") # Expected: (batch_size, seq_len, NUM_TO_X_CLASSES)
        print(f"  Ty: {logits_ty.shape}") # Expected: (batch_size, seq_len, NUM_TO_Y_CLASSES)
        
        # Verify output dimensions against constants
        assert logits_fx.shape == (batch_size, seq_len, constants.NUM_FROM_X_CLASSES)
        assert logits_fy.shape == (batch_size, seq_len, constants.NUM_FROM_Y_CLASSES)
        assert logits_tx.shape == (batch_size, seq_len, constants.NUM_TO_X_CLASSES)
        assert logits_ty.shape == (batch_size, seq_len, constants.NUM_TO_Y_CLASSES)
        print("\nOutput shapes are correct.")

    except Exception as e:
        print(f"\nError during forward pass: {e}")
        import traceback
        traceback.print_exc() 