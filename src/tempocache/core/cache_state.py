"""Cache state container for HDO outputs and routing metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class CacheState:
    """Per-module cache state with per-sample validity/age/signature."""

    cache: torch.Tensor | None = None
    signature: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    age: torch.Tensor | None = None
    batch_size: int = 0
    device: torch.device | None = None
    history: list[dict] = field(default_factory=list)

    def ensure_batch(self, batch_size: int, device: torch.device) -> None:
        if self.valid is None or self.age is None or self.batch_size != batch_size or self.device != device:
            self.batch_size = batch_size
            self.device = device
            self.valid = torch.zeros((batch_size,), dtype=torch.bool, device=device)
            self.age = torch.zeros((batch_size,), dtype=torch.long, device=device)
            self.signature = None
            self.cache = None

    def reset(self) -> None:
        if self.valid is not None:
            self.valid.zero_()
        if self.age is not None:
            self.age.zero_()
        self.cache = None
        self.signature = None
        self.history.clear()

    def resize(self, batch_size: int, device: torch.device) -> None:
        old_valid = self.valid
        old_age = self.age
        old_cache = self.cache
        old_sig = self.signature
        old_batch = self.batch_size
        self.ensure_batch(batch_size=batch_size, device=device)
        if old_valid is None or old_age is None or old_batch == 0:
            return
        n = min(old_batch, batch_size)
        self.valid[:n] = old_valid[:n].to(device=device)
        self.age[:n] = old_age[:n].to(device=device)
        if old_cache is not None and self.cache is None and old_cache.shape[0] >= n:
            # Preserve old cache if shape-compatible with current batch size.
            if old_cache.shape[0] == batch_size:
                self.cache = old_cache.to(device=device)
            else:
                self.cache = old_cache[:n].to(device=device)
        if old_sig is not None and self.signature is None and old_sig.shape[0] >= n:
            if old_sig.shape[0] == batch_size:
                self.signature = old_sig.to(device=device)
            else:
                self.signature = old_sig[:n].to(device=device)

    def _ensure_cache_tensor(self, values: torch.Tensor) -> None:
        if self.cache is None or self.cache.shape[1:] != values.shape[1:] or self.cache.shape[0] != self.batch_size:
            self.cache = torch.zeros(
                (self.batch_size, *values.shape[1:]), dtype=values.dtype, device=values.device
            )

    def update_drive(self, indices: torch.Tensor, values: torch.Tensor) -> None:
        if indices.numel() == 0:
            return
        self._ensure_cache_tensor(values)
        assert self.valid is not None and self.age is not None and self.cache is not None
        self.cache[indices] = values
        self.valid[indices] = True
        self.age[indices] = 0

    def update_signature(self, indices: torch.Tensor, signatures: torch.Tensor) -> None:
        if indices.numel() == 0:
            return
        if self.signature is None or self.signature.shape[0] != self.batch_size or self.signature.shape[1] != signatures.shape[1]:
            self.signature = torch.zeros((self.batch_size, signatures.shape[1]), dtype=signatures.dtype, device=signatures.device)
        self.signature[indices] = signatures

    def invalidate(self, mask: torch.Tensor) -> None:
        if mask.numel() == 0:
            return
        assert self.valid is not None and self.age is not None
        self.valid[mask] = False
        self.age[mask] = 0

    def increment_age(self, mask: torch.Tensor | None = None) -> None:
        if self.age is None:
            return
        if mask is None:
            self.age += 1
        elif mask.numel() > 0:
            self.age[mask] += 1

