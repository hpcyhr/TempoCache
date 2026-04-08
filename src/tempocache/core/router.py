"""Adaptive routing logic for FULL/COLLAPSE/REUSE selection."""

from __future__ import annotations

import torch

from ..config import RuntimeConfig, ThresholdConfig
from ..enums import RuntimeMode
from ..typing import RouterOutput

MODE_FULL = 0
MODE_COLLAPSE = 1
MODE_REUSE = 2


class TempoRouter:
    """Unified router for all HDO families."""

    def __init__(self, runtime_config: RuntimeConfig) -> None:
        self.runtime_config = runtime_config

    def _fixed_modes(self, mode: RuntimeMode, batch_size: int, device: torch.device) -> torch.Tensor:
        if mode == RuntimeMode.FULL:
            return torch.full((batch_size,), MODE_FULL, dtype=torch.long, device=device)
        if mode == RuntimeMode.FIXED_COLLAPSE:
            return torch.full((batch_size,), MODE_COLLAPSE, dtype=torch.long, device=device)
        if mode == RuntimeMode.FIXED_REUSE:
            return torch.full((batch_size,), MODE_REUSE, dtype=torch.long, device=device)
        raise ValueError(f"Unsupported fixed mode: {mode}")

    def route(
        self,
        *,
        window_index: int,
        d_inter: torch.Tensor,
        v_temp: torch.Tensor,
        cache_valid: torch.Tensor,
        cache_age: torch.Tensor,
        thresholds: ThresholdConfig,
    ) -> RouterOutput:
        mode = self.runtime_config.resolved_mode()
        batch_size = int(d_inter.numel())
        device = d_inter.device

        if mode != RuntimeMode.ADAPTIVE:
            fixed = self._fixed_modes(mode, batch_size, device)
            if mode == RuntimeMode.FIXED_REUSE:
                fixed = torch.where(cache_valid, fixed, torch.full_like(fixed, MODE_FULL))
            zero_mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)
            return RouterOutput(modes=fixed, hard_invalidation_mask=zero_mask, forced_refresh_mask=zero_mask)

        modes = torch.full((batch_size,), MODE_FULL, dtype=torch.long, device=device)
        hard_mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        forced_mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)

        if window_index < thresholds.warmup_full_windows:
            return RouterOutput(modes=modes, hard_invalidation_mask=hard_mask, forced_refresh_mask=forced_mask)

        invalid_cache = ~cache_valid
        modes[invalid_cache] = MODE_FULL

        hard = cache_valid & (d_inter >= thresholds.tau_reuse_hard)
        modes[hard] = MODE_FULL
        hard_mask |= hard

        forced = cache_valid & (cache_age >= thresholds.max_reuse)
        modes[forced] = MODE_FULL
        forced_mask |= forced

        can_adapt = cache_valid & ~hard & ~forced
        reuse = can_adapt & (d_inter <= thresholds.tau_reuse) & (v_temp <= thresholds.tau_stable)
        collapse = can_adapt & ~reuse & (v_temp <= thresholds.tau_collapse)

        modes[reuse] = MODE_REUSE
        modes[collapse] = MODE_COLLAPSE

        return RouterOutput(modes=modes, hard_invalidation_mask=hard_mask, forced_refresh_mask=forced_mask)


def mode_id_to_name(mode_id: int) -> str:
    if mode_id == MODE_FULL:
        return "FULL"
    if mode_id == MODE_COLLAPSE:
        return "COLLAPSE"
    if mode_id == MODE_REUSE:
        return "REUSE"
    return "UNKNOWN"

