from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class PrefixRouteCacheEntry:
    experts: torch.Tensor
    probs: torch.Tensor


class PrefixRoutingCache:
    def __init__(self, jsd_threshold: float = 0.05):
        self.jsd_threshold = float(jsd_threshold)
        self._store: Dict[Tuple[str, int, int], PrefixRouteCacheEntry] = {}

    @staticmethod
    def _jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        m = 0.5 * (p + q)
        kl1 = (p * ((p + eps).log() - (m + eps).log())).sum(dim=-1)
        kl2 = (q * ((q + eps).log() - (m + eps).log())).sum(dim=-1)
        return 0.5 * (kl1 + kl2)

    def get(
        self,
        sample_uid: str,
        layer_id: int,
        token_pos: int,
        cur_probs: torch.Tensor,
    ) -> Optional[PrefixRouteCacheEntry]:
        key = (sample_uid, int(layer_id), int(token_pos))
        ent = self._store.get(key)
        if ent is None:
            return None
        jsd = self._jsd(cur_probs, ent.probs)
        if float(jsd.detach().cpu()) <= self.jsd_threshold:
            return ent
        return None

    def put(self, sample_uid: str, layer_id: int, token_pos: int, experts: torch.Tensor, probs: torch.Tensor) -> None:
        key = (sample_uid, int(layer_id), int(token_pos))
        self._store[key] = PrefixRouteCacheEntry(experts=experts.detach().clone(), probs=probs.detach().clone())

    def clear(self) -> None:
        self._store.clear()
