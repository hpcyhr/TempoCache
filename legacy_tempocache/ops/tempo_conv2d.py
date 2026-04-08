"""
tempocache/ops/tempo_conv2d.py

TempoConv2d – similarity-driven block-level temporal inference runtime wrapper
for nn.Conv2d.

Enhanced from the validated MVP with:
  - profiler hook integration
  - layer_name for per-layer diagnostics
  - contiguous output guarantee

Core semantics (Full / Collapse / Reuse) are preserved exactly.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from tempocache.config.default_config import RouterConfig
from tempocache.runtime.cache_state import CacheState
from tempocache.runtime.profiler import Profiler
from tempocache.runtime.router import route
from tempocache.runtime.signature_extractor import (
    build_signature,
    compute_temporal_variation,
)


class TempoConv2d(nn.Module):
    """Temporal-caching wrapper around nn.Conv2d.

    Parameters
    ----------
    conv        : nn.Conv2d to wrap (weights are shared, not copied).
    config      : RouterConfig with threshold values.
    grid_size   : coarse-grid resolution G for spatial signature.
    forced_mode : bypass the router with a fixed mode.
    layer_name  : human-readable name used for profiler reporting.
    profiler    : shared Profiler instance (optional).
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        config: Optional[RouterConfig] = None,
        grid_size: int = 2,
        forced_mode: Optional[str] = None,
        layer_name: str = "",
        profiler: Optional[Profiler] = None,
    ):
        super().__init__()
        self.conv = conv
        self.config = config or RouterConfig()
        self.grid_size = grid_size
        self.forced_mode = forced_mode
        self.layer_name = layer_name
        self.profiler = profiler
        self.cache = CacheState()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset_cache(self) -> None:
        self.cache.reset()

    # ------------------------------------------------------------------
    # Raw conv call – bypasses SpikingJelly step_mode wrapper
    # ------------------------------------------------------------------
    def _conv_forward(self, x: Tensor) -> Tensor:
        """Call the underlying nn.Conv2d.forward directly.

        SpikingJelly's layer.Conv2d in step_mode='m' expects 5D input and
        applies its own temporal unrolling.  Since TempoConv2d handles the
        temporal dimension itself, we bypass the SJ wrapper and call the
        base nn.Conv2d.forward which accepts standard 4D [N, C, H, W].
        """
        return nn.Conv2d.forward(self.conv, x)

    # ------------------------------------------------------------------
    # Execution modes  (semantics preserved from validated MVP)
    # ------------------------------------------------------------------
    def _exec_full(self, x_window: Tensor) -> Tuple[Tensor, Tensor]:
        K = x_window.shape[0]
        ys = []
        for t in range(K):
            ys.append(self._conv_forward(x_window[t]))
        y_window = torch.stack(ys, dim=0)
        representative = y_window.mean(dim=0)
        return y_window, representative

    def _exec_collapse(self, x_window: Tensor) -> Tuple[Tensor, Tensor]:
        K = x_window.shape[0]
        x_mean = x_window.mean(dim=0)
        y_agg = self._conv_forward(x_mean)
        y_window = y_agg.unsqueeze(0).expand(K, -1, -1, -1, -1).contiguous()
        return y_window, y_agg

    def _exec_reuse(self, x_window: Tensor) -> Tensor:
        K = x_window.shape[0]
        assert self.cache.entry is not None, "Reuse called with empty cache"
        y_window = self.cache.entry.unsqueeze(0).expand(K, -1, -1, -1, -1).contiguous()
        return y_window

    # ------------------------------------------------------------------
    # Core logic (shared by forward and forward_with_diag)
    # ------------------------------------------------------------------
    def _step(self, x_window: Tensor) -> Tuple[Tensor, str, Dict]:
        K = x_window.shape[0]
        assert K in (2, 4), f"Window length K must be 2 or 4, got {K}"

        sig_cur = build_signature(x_window, self.grid_size)
        var_intra = compute_temporal_variation(x_window).item()

        # --- routing ---
        if self.forced_mode is not None:
            mode = self.forced_mode
            diag: Dict[str, object] = {
                "mode": mode,
                "d_inter": 0.0,
                "var_intra": var_intra,
                "cache_valid_before": self.cache.valid,
                "cache_valid_after": self.cache.valid,
                "cache_age_before": self.cache.age,
                "signature_norm": sig_cur.abs().sum().item(),
            }
            if mode == "reuse" and not self.cache.valid:
                mode = "full"
                diag["mode"] = "full"
                diag["forced_fallback"] = True
        else:
            mode, _, diag = route(
                sig_cur=sig_cur,
                sig_prev=self.cache.last_signature,
                cache_state=self.cache,
                var_intra=var_intra,
                config=self.config,
            )

        # --- execute ---
        cache_updated = False
        if mode == "full":
            y_window, representative = self._exec_full(x_window)
            self.cache.update_full(representative, sig_cur)
            cache_updated = True
        elif mode == "collapse":
            y_window, aggregated = self._exec_collapse(x_window)
            self.cache.update_collapse(aggregated, sig_cur)
            cache_updated = True
        elif mode == "reuse":
            y_window = self._exec_reuse(x_window)
            self.cache.update_reuse(sig_cur)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        diag["cache_updated"] = cache_updated
        diag["cache_age_after"] = self.cache.age
        diag["cache_valid_after"] = self.cache.valid

        # --- profiler hook ---
        if self.profiler is not None:
            self.profiler.record(self.layer_name, diag)

        # --- store last diagnostics for external inspection ---
        self.last_mode = mode
        self.last_diag = diag

        return y_window, mode, diag

    # ------------------------------------------------------------------
    # Forward – returns Tensor only (compatible with nn.Sequential)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, x_window: Tensor) -> Tensor:
        """Model-compatible forward: returns y_window [K, B, Cout, Hout, Wout].

        Diagnostics are stored in self.last_mode and self.last_diag,
        and recorded by the profiler if one is attached.
        """
        y_window, _, _ = self._step(x_window)
        return y_window

    # ------------------------------------------------------------------
    # Forward with diagnostics – for standalone / benchmark use
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_with_diag(
        self, x_window: Tensor
    ) -> Tuple[Tensor, str, CacheState, Dict]:
        """Explicit diagnostic interface (backward-compatible with MVP).

        Returns (y_window, mode, cache, diag).
        """
        y_window, mode, diag = self._step(x_window)
        return y_window, mode, self.cache, diag