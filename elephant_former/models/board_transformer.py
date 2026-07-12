"""Encoder-only board-state transformer for ElephantFormer (Phase 0).

The model reads a position as 91 tokens (90 square tokens + 1 global ``[CLS]``
token that also carries side-to-move) and produces:

* a **policy** over ``90 x 90 = 8100`` from-square -> to-square moves via an
  attention-style bilinear head, and
* a **value** (loss/draw/win from the side-to-move's perspective) from the CLS
  representation.

See :mod:`elephant_former.data_utils.board_features` for the feature layout and
the flat square/move index conventions.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from elephant_former.data_utils import board_features as bf

NEG_INF = float("-inf")


class BoardTransformer(nn.Module):
    """Encoder-only transformer over board-state tokens with policy + value heads."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        policy_head_dim: int = 256,
        num_value_classes: int = 3,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.policy_head_dim = policy_head_dim
        self.num_value_classes = num_value_classes

        # Per-square token components (summed).
        self.piece_embedding = nn.Embedding(bf.NUM_PIECE_CLASSES, d_model)
        self.file_embedding = nn.Embedding(bf.BOARD_WIDTH, d_model)
        self.rank_embedding = nn.Embedding(bf.BOARD_HEIGHT, d_model)
        # One small learned embedding per binary flag (index 0/1).
        self.flag_embeddings = nn.ModuleList(
            [nn.Embedding(2, d_model) for _ in range(bf.NUM_FLAGS)]
        )

        # Global CLS token: its own embedding plus a side-to-move embedding.
        self.cls_base = nn.Parameter(torch.zeros(d_model))
        self.side_to_move_embedding = nn.Embedding(2, d_model)

        # Fixed per-square file/rank indices (buffers, not learned).
        squares = torch.arange(bf.NUM_SQUARES)
        self.register_buffer("square_files", squares % bf.BOARD_WIDTH, persistent=False)
        self.register_buffer("square_ranks", squares // bf.BOARD_WIDTH, persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-LN
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False
        )

        # Attention-style from->to policy head.
        self.policy_q = nn.Linear(d_model, policy_head_dim, bias=False)
        self.policy_k = nn.Linear(d_model, policy_head_dim, bias=False)

        # Value head: CLS -> MLP -> {loss, draw, win}.
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_value_classes),
        )

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        initrange = 0.1
        for emb in (
            self.piece_embedding,
            self.file_embedding,
            self.rank_embedding,
            self.side_to_move_embedding,
            *self.flag_embeddings,
        ):
            emb.weight.data.uniform_(-initrange, initrange)
        nn.init.uniform_(self.cls_base, -initrange, initrange)

    def _embed_tokens(
        self, piece_ids: torch.Tensor, flags: torch.Tensor, side_to_move: torch.Tensor
    ) -> torch.Tensor:
        """Build the ``(batch, 91, d_model)`` token sequence from features."""
        batch_size = piece_ids.size(0)

        square_tokens = (
            self.piece_embedding(piece_ids)
            + self.file_embedding(self.square_files).unsqueeze(0)
            + self.rank_embedding(self.square_ranks).unsqueeze(0)
        )
        for i, flag_emb in enumerate(self.flag_embeddings):
            square_tokens = square_tokens + flag_emb(flags[:, :, i])

        cls_token = self.cls_base.unsqueeze(0) + self.side_to_move_embedding(side_to_move)
        cls_token = cls_token.unsqueeze(1)  # (batch, 1, d_model)

        return torch.cat([square_tokens, cls_token], dim=1)  # (batch, 91, d_model)

    def forward(
        self,
        piece_ids: torch.Tensor,
        flags: torch.Tensor,
        side_to_move: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            piece_ids: ``(batch, 90)`` long tensor of piece class ids.
            flags: ``(batch, 90, 4)`` long tensor of binary flags.
            side_to_move: ``(batch,)`` long tensor (0=RED, 1=BLACK).

        Returns:
            ``(policy_logits, value_logits)`` where ``policy_logits`` has shape
            ``(batch, 8100)`` (flattened ``from_sq * 90 + to_sq``) and
            ``value_logits`` has shape ``(batch, 3)``.
        """
        x = self._embed_tokens(piece_ids, flags, side_to_move)
        x = self.dropout(x)
        hidden = self.encoder(x)  # (batch, 91, d_model)

        square_hidden = hidden[:, : bf.NUM_SQUARES, :]  # (batch, 90, d_model)
        cls_hidden = hidden[:, bf.NUM_SQUARES, :]  # (batch, d_model)

        q = self.policy_q(square_hidden)  # (batch, 90, head_dim)
        k = self.policy_k(square_hidden)  # (batch, 90, head_dim)
        scores = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(self.policy_head_dim)
        policy_logits = scores.reshape(scores.size(0), bf.NUM_POLICY_MOVES)

        value_logits = self.value_head(cls_hidden)
        return policy_logits, value_logits


def build_legal_mask(
    legal_indices: Iterable[int],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an ``(8100,)`` additive mask: ``0`` for legal moves, ``-inf`` otherwise."""
    mask = torch.full((bf.NUM_POLICY_MOVES,), NEG_INF, device=device, dtype=dtype)
    idx = list(legal_indices)
    if idx:
        mask[torch.tensor(idx, dtype=torch.long, device=device)] = 0.0
    return mask


def mask_policy_logits(policy_logits: torch.Tensor, legal_indices: Iterable[int]) -> torch.Tensor:
    """Apply an additive legal-move mask to a single ``(8100,)`` logit vector."""
    mask = build_legal_mask(legal_indices, device=policy_logits.device, dtype=policy_logits.dtype)
    return policy_logits + mask


def select_move_index(
    policy_logits: torch.Tensor,
    legal_indices: List[int],
    temperature: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Pick a legal move's flat policy index from a single ``(8100,)`` logit vector.

    ``temperature <= 0`` selects the argmax; otherwise samples from the
    temperature-scaled softmax over legal moves only.
    """
    if not legal_indices:
        raise ValueError("No legal moves to select from.")

    masked = mask_policy_logits(policy_logits, legal_indices)
    if temperature <= 0.0:
        return int(torch.argmax(masked).item())

    probs = torch.softmax(masked / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BoardTransformer(d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128)
    batch = 3
    piece_ids = torch.randint(0, bf.NUM_PIECE_CLASSES, (batch, bf.NUM_SQUARES))
    flags = torch.randint(0, 2, (batch, bf.NUM_SQUARES, bf.NUM_FLAGS))
    stm = torch.randint(0, 2, (batch,))
    policy, value = model(piece_ids, flags, stm)
    print("policy", tuple(policy.shape), "value", tuple(value.shape))
    masked = mask_policy_logits(policy[0], [0, 10, 8099])
    print("finite entries after mask:", int(torch.isfinite(masked).sum().item()))
