from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class ComplementaryMaskOutput:
    mask1: torch.Tensor
    mask2: torch.Tensor
    t1: torch.Tensor
    t2: torch.Tensor


class TimestepSampler:
    def __init__(self, mode: str = "uniform", late_bias: float = 0.0):
        self.mode = mode
        self.late_bias = float(late_bias)

    def sample(self, batch: int, device: torch.device) -> torch.Tensor:
        u = torch.rand(batch, device=device)
        if self.mode == "late_bias":
            gamma = max(1e-6, 1.0 + self.late_bias)
            return u.pow(gamma)
        if self.mode == "early_bias":
            gamma = max(1e-6, 1.0 + self.late_bias)
            return 1.0 - (1.0 - u).pow(gamma)
        return u


def _answer_positions(labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    return labels.ne(ignore_index)


def build_complementary_masks(
    labels: torch.Tensor,
    prompt_mask: torch.Tensor,
    include_prompt_ratio: float,
    timestep_sampler_1: TimestepSampler,
    timestep_sampler_2: TimestepSampler,
    force_answer_coverage: bool = True,
    ignore_index: int = -100,
) -> ComplementaryMaskOutput:
    """
    Build two disjoint masks with mandatory answer coverage.
    labels shape: [B, L] where answer tokens are != ignore_index.
    prompt_mask shape: [B, L] True for prompt tokens.
    """
    device = labels.device
    bsz, seqlen = labels.shape
    ans_mask = _answer_positions(labels, ignore_index=ignore_index)

    t1 = timestep_sampler_1.sample(bsz, device)
    t2 = timestep_sampler_2.sample(bsz, device)

    rand1 = torch.rand((bsz, seqlen), device=device)
    rand2 = torch.rand((bsz, seqlen), device=device)

    p1 = t1[:, None]
    p2 = t2[:, None]

    mask1 = ans_mask & (rand1 < p1)
    mask2 = ans_mask & (~mask1) & (rand2 < p2)

    # optional prompt masking for curriculum stage2/3
    if include_prompt_ratio > 0:
        prompt_pick = prompt_mask & (torch.rand((bsz, seqlen), device=device) < include_prompt_ratio)
        p_split = torch.rand((bsz, seqlen), device=device) < 0.5
        mask1 = mask1 | (prompt_pick & p_split)
        mask2 = mask2 | (prompt_pick & (~p_split) & (~mask1))

    if force_answer_coverage:
        uncovered = ans_mask & (~mask1) & (~mask2)
        if uncovered.any():
            # assign uncovered answer tokens to branch1 by default
            mask1 = mask1 | uncovered

    # enforce disjointness
    overlap = mask1 & mask2
    if overlap.any():
        mask2 = mask2 & (~overlap)

    return ComplementaryMaskOutput(mask1=mask1, mask2=mask2, t1=t1, t2=t2)


def apply_absorbing_mask(input_ids: torch.Tensor, mask: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    out = input_ids.clone()
    out[mask] = int(mask_token_id)
    return out


def estimate_no_supervision_ratio(answer_lengths: torch.Tensor) -> torch.Tensor:
    # E[(1-r)^n] with r~U(0,1) = 1/(n+1)
    n = answer_lengths.float()
    return 1.0 / (n + 1.0)
