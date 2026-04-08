"""Diagnostics collection for Tempo modules."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .router import MODE_COLLAPSE, MODE_FULL, MODE_REUSE


@dataclass
class ModuleDiagnostics:
    """Per-module counters and optional per-window history."""

    module_name: str
    operator_family: str
    keep_window_history: bool = True
    history_max_windows: int = 256
    total_windows: int = 0
    total_samples: int = 0
    full_count: int = 0
    collapse_count: int = 0
    reuse_count: int = 0
    cache_hit_count: int = 0
    hard_invalidation_count: int = 0
    forced_refresh_count: int = 0
    _sum_d_inter: float = 0.0
    _sum_v_temp: float = 0.0
    window_history: list[dict] = field(default_factory=list)

    def record(
        self,
        modes: torch.Tensor,
        d_inter: torch.Tensor,
        v_temp: torch.Tensor,
        hard_invalidation_mask: torch.Tensor,
        forced_refresh_mask: torch.Tensor,
    ) -> None:
        self.total_windows += 1
        batch_size = int(modes.numel())
        self.total_samples += batch_size
        self.full_count += int((modes == MODE_FULL).sum().item())
        self.collapse_count += int((modes == MODE_COLLAPSE).sum().item())
        self.reuse_count += int((modes == MODE_REUSE).sum().item())
        self.cache_hit_count += int((modes == MODE_REUSE).sum().item())
        self.hard_invalidation_count += int(hard_invalidation_mask.sum().item())
        self.forced_refresh_count += int(forced_refresh_mask.sum().item())
        self._sum_d_inter += float(d_inter.mean().item()) if d_inter.numel() > 0 else 0.0
        self._sum_v_temp += float(v_temp.mean().item()) if v_temp.numel() > 0 else 0.0

        if self.keep_window_history:
            self.window_history.append(
                {
                    "window_index": self.total_windows - 1,
                    "full": int((modes == MODE_FULL).sum().item()),
                    "collapse": int((modes == MODE_COLLAPSE).sum().item()),
                    "reuse": int((modes == MODE_REUSE).sum().item()),
                    "mean_d_inter": float(d_inter.mean().item()) if d_inter.numel() > 0 else 0.0,
                    "mean_v_temp": float(v_temp.mean().item()) if v_temp.numel() > 0 else 0.0,
                }
            )
            if len(self.window_history) > self.history_max_windows:
                self.window_history = self.window_history[-self.history_max_windows :]

    @property
    def mean_d_inter(self) -> float:
        if self.total_windows == 0:
            return 0.0
        return self._sum_d_inter / self.total_windows

    @property
    def mean_v_temp(self) -> float:
        if self.total_windows == 0:
            return 0.0
        return self._sum_v_temp / self.total_windows

    def to_dict(self) -> dict:
        payload = {
            "module_name": self.module_name,
            "operator_family": self.operator_family,
            "total_windows": self.total_windows,
            "total_samples": self.total_samples,
            "full_count": self.full_count,
            "collapse_count": self.collapse_count,
            "reuse_count": self.reuse_count,
            "cache_hit_count": self.cache_hit_count,
            "hard_invalidation_count": self.hard_invalidation_count,
            "forced_refresh_count": self.forced_refresh_count,
            "mean_d_inter": self.mean_d_inter,
            "mean_v_temp": self.mean_v_temp,
        }
        if self.keep_window_history:
            payload["window_history"] = list(self.window_history)
        return payload

