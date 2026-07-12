"""PyTorch Lightning module for the board-state ElephantFormer (Phase 0).

Multi-task loss: ``CE(policy) + value_loss_weight * CE(value)`` where the value
loss ignores positions whose result is unknown (``ignore_index``).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn

from elephant_former.data_utils.board_dataset import VALUE_IGNORE_INDEX
from elephant_former.models.board_transformer import BoardTransformer


class BoardLightningModule(pl.LightningModule):
    """Lightning wrapper around :class:`BoardTransformer`."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        policy_head_dim: int = 256,
        num_value_classes: int = 3,
        value_loss_weight: float = 0.5,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = BoardTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            policy_head_dim=policy_head_dim,
            num_value_classes=num_value_classes,
        )
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.CrossEntropyLoss(ignore_index=VALUE_IGNORE_INDEX)

    def forward(
        self, piece_ids: torch.Tensor, flags: torch.Tensor, side_to_move: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(piece_ids, flags, side_to_move)

    def _step(self, batch: Tuple[torch.Tensor, ...], stage: str) -> torch.Tensor:
        piece_ids, flags, side_to_move, target_policy, target_value = batch
        policy_logits, value_logits = self.model(piece_ids, flags, side_to_move)

        policy_loss = self.policy_loss_fn(policy_logits, target_policy)
        value_loss = self.value_loss_fn(value_logits, target_value)
        total_loss = policy_loss + self.hparams.value_loss_weight * value_loss

        batch_size = piece_ids.size(0)
        policy_acc = (policy_logits.argmax(dim=-1) == target_policy).float().mean()

        value_mask = target_value != VALUE_IGNORE_INDEX
        if value_mask.any():
            value_acc = (
                value_logits[value_mask].argmax(dim=-1) == target_value[value_mask]
            ).float().mean()
        else:
            value_acc = torch.zeros((), device=self.device)

        on_step = stage == "train"
        log_kwargs = dict(on_step=on_step, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_loss", total_loss, **log_kwargs)
        self.log(f"{stage}_policy_loss", policy_loss, on_step=on_step, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}_value_loss", value_loss, on_step=on_step, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}_policy_acc", policy_acc, on_step=on_step, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_value_acc", value_acc, on_step=on_step, on_epoch=True, batch_size=batch_size)
        return total_loss

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        self._step(batch, "test")

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return {"optimizer": optimizer}


if __name__ == "__main__":
    module = BoardLightningModule(d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128)
    print("Instantiated BoardLightningModule.")
    print("Hyperparameters:", dict(module.hparams))
