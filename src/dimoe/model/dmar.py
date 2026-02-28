from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DMARLossOutput:
    prior_loss: torch.Tensor
    consistency_loss: torch.Tensor
    balance_loss: torch.Tensor
    entropy_loss: torch.Tensor
    stats: Dict[str, float]


class DMARRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_t_buckets: int = 8,
        num_token_types: int = 4,
        aux_dim: int = 64,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.num_t_buckets = int(num_t_buckets)
        self.num_token_types = int(num_token_types)

        self.t_embed = nn.Embedding(self.num_t_buckets, aux_dim)
        self.type_embed = nn.Embedding(self.num_token_types, aux_dim)
        self.router = nn.Linear(hidden_size + 2 * aux_dim, num_experts)

        # prior[layer, t_bucket, token_type, expert] is managed outside per layer.
        # for module-level usage in a single location, keep shared prior table.
        self.prior = nn.Parameter(torch.zeros(self.num_t_buckets, self.num_token_types, num_experts))

    def forward(
        self,
        hidden_states: torch.Tensor,
        t_bucket: torch.Tensor,
        token_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        hidden_states: [B, L, H]
        t_bucket: [B, L]
        token_type: [B, L]
        returns gate_probs: [B, L, E]
        """
        t_emb = self.t_embed(t_bucket)
        ty_emb = self.type_embed(token_type)
        x = torch.cat([hidden_states, t_emb, ty_emb], dim=-1)
        logits = self.router(x)
        return F.softmax(logits, dim=-1)

    def topk_dispatch(self, gate_probs: torch.Tensor, k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        vals, idx = torch.topk(gate_probs, k=k, dim=-1)
        return idx, vals


def _bucketed_mean_pi(
    gate_probs: torch.Tensor,
    t_bucket: torch.Tensor,
    token_type: torch.Tensor,
    num_t_buckets: int,
    num_token_types: int,
) -> torch.Tensor:
    # output [T, Ty, E]
    b, l, e = gate_probs.shape
    out = gate_probs.new_zeros((num_t_buckets, num_token_types, e))
    cnt = gate_probs.new_zeros((num_t_buckets, num_token_types, 1))
    for tb in range(num_t_buckets):
        for ty in range(num_token_types):
            m = (t_bucket == tb) & (token_type == ty)
            if m.any():
                p = gate_probs[m]
                out[tb, ty] = p.mean(dim=0)
                cnt[tb, ty] = float(p.shape[0])
    return out, cnt


def dmar_losses(
    gate_probs: torch.Tensor,
    gate_probs_other: Optional[torch.Tensor],
    prefix_mask: torch.Tensor,
    t_bucket: torch.Tensor,
    token_type: torch.Tensor,
    prior_table: torch.Tensor,
    lambda_prior: float,
    lambda_consistency: float,
    lambda_balance: float,
    lambda_entropy: float,
) -> DMARLossOutput:
    # prior_table: [T, Ty, E]
    eps = 1e-9
    t_buckets = prior_table.shape[0]
    token_types = prior_table.shape[1]
    experts = prior_table.shape[2]

    mean_pi, cnt = _bucketed_mean_pi(gate_probs, t_bucket, token_type, t_buckets, token_types)
    prior = F.softmax(prior_table, dim=-1)

    valid = cnt.squeeze(-1) > 0
    if valid.any():
        prior_loss = F.kl_div((mean_pi[valid] + eps).log(), prior[valid], reduction="batchmean")
    else:
        prior_loss = gate_probs.new_tensor(0.0)

    # per-bucket balance, default target = prior
    if valid.any():
        bal_loss = F.kl_div((mean_pi[valid] + eps).log(), prior[valid].detach(), reduction="batchmean")
    else:
        bal_loss = gate_probs.new_tensor(0.0)

    if gate_probs_other is not None and prefix_mask.any():
        p = gate_probs[prefix_mask]
        q = gate_probs_other[prefix_mask]
        m = 0.5 * (p + q)
        jsd = 0.5 * (F.kl_div((p + eps).log(), m, reduction="batchmean") + F.kl_div((q + eps).log(), m, reduction="batchmean"))
        consistency_loss = jsd
    else:
        consistency_loss = gate_probs.new_tensor(0.0)

    # encourage sufficient exploration early
    entropy = -(gate_probs * (gate_probs + eps).log()).sum(dim=-1).mean()
    entropy_loss = -entropy

    total_prior = lambda_prior * prior_loss
    total_cons = lambda_consistency * consistency_loss
    total_bal = lambda_balance * bal_loss
    total_ent = lambda_entropy * entropy_loss

    stats = {
        "dmar_prior": float(prior_loss.detach().cpu()),
        "dmar_consistency": float(consistency_loss.detach().cpu()),
        "dmar_balance": float(bal_loss.detach().cpu()),
        "dmar_entropy": float(entropy.detach().cpu()),
    }

    return DMARLossOutput(
        prior_loss=total_prior,
        consistency_loss=total_cons,
        balance_loss=total_bal,
        entropy_loss=total_ent,
        stats=stats,
    )
