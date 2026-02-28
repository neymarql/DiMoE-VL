from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class DiffusionLossOutput:
    loss: torch.Tensor
    token_count: int


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, ignore_index: int = -100) -> DiffusionLossOutput:
    # logits [B,L,V], labels [B,L], mask [B,L]
    cnt = int(mask.sum().item())
    if cnt <= 0:
        # Avoid NaN when all targets are ignore_index.
        return DiffusionLossOutput(loss=logits.new_zeros(()), token_count=0)

    active_labels = labels.clone()
    active_labels[~mask] = ignore_index
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), active_labels.view(-1), ignore_index=ignore_index)
    return DiffusionLossOutput(loss=loss, token_count=cnt)
