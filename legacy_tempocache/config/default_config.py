"""
tempocache/config/default_config.py

Centralised dataclass configuration for the TempoCache runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RouterConfig:
    """Threshold knobs for the TempoCache router."""

    tau_reuse: float = 0.05
    tau_reuse_hard: float = 0.50
    tau_stable: float = 0.02
    tau_collapse: float = 0.10
    max_reuse: int = 4


@dataclass
class TempoConfig:
    """Top-level configuration for the TempoCache runtime."""

    # --- windowing ---
    window_size: int = 4          # K, must be 2 or 4
    grid_size: int = 2            # G for spatial signature

    # --- router thresholds ---
    router: RouterConfig = field(default_factory=RouterConfig)

    # --- mode override ---
    forced_mode: Optional[str] = None  # "full", "collapse", "reuse", or None

    # --- layer selection ---
    # If non-empty, only wrap modules whose name matches a prefix in this list.
    # An empty list means "wrap all eligible Conv2d layers".
    wrap_whitelist: List[str] = field(default_factory=list)
    # Modules whose name matches a prefix here are never wrapped.
    wrap_blacklist: List[str] = field(default_factory=list)
    # If > 0, wrap at most this many Conv2d layers (in module-tree order).
    wrap_max_layers: int = 0

    # --- diagnostics ---
    record_per_window_trace: bool = False

    # --- output ---
    export_dir: str = "tempocache_artifacts"

    # --- reproducibility ---
    seed: int = 42

    def __post_init__(self) -> None:
        assert self.window_size in (2, 4), f"window_size must be 2 or 4, got {self.window_size}"