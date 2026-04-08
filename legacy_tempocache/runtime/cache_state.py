from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CacheState:
    entry: Optional[torch.Tensor] = None
    valid: bool = False
    age: int = 0
    last_signature: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.entry = None
        self.valid = False
        self.age = 0
        self.last_signature = None

    def update_full(self, entry: torch.Tensor, signature: Optional[torch.Tensor] = None) -> None:
        self.entry = entry.detach()
        self.valid = True
        self.age = 0
        self.last_signature = None if signature is None else signature.detach()

    def update_collapse(self, entry: torch.Tensor, signature: Optional[torch.Tensor] = None) -> None:
        self.entry = entry.detach()
        self.valid = True
        self.age = 0
        self.last_signature = None if signature is None else signature.detach()

    def update_reuse(self) -> None:
        self.age += 1

    def clone_shallow(self) -> "CacheState":
        return CacheState(
            entry=self.entry,
            valid=self.valid,
            age=self.age,
            last_signature=self.last_signature,
        )