"""
tempocache/runtime/profiler.py

Collects per-layer runtime statistics for TempoCache.
Each TempoConv2d wrapper reports diagnostics to a shared Profiler instance.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LayerStats:
    """Accumulated statistics for a single wrapped layer."""

    full_count: int = 0
    collapse_count: int = 0
    reuse_count: int = 0
    cache_refresh_count: int = 0
    cache_invalidation_count: int = 0
    d_inter_sum: float = 0.0
    var_intra_sum: float = 0.0
    window_count: int = 0
    traces: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, diag: Dict[str, Any], record_trace: bool = False) -> None:
        mode = diag["mode"]
        if mode == "full":
            self.full_count += 1
        elif mode == "collapse":
            self.collapse_count += 1
        elif mode == "reuse":
            self.reuse_count += 1

        if diag.get("cache_updated", False):
            self.cache_refresh_count += 1
        if diag.get("cache_valid_before", True) and not diag.get("cache_valid_after", True):
            self.cache_invalidation_count += 1

        d = diag.get("d_inter", 0.0)
        if d != float("inf"):
            self.d_inter_sum += d
        self.var_intra_sum += diag.get("var_intra", 0.0)
        self.window_count += 1

        if record_trace:
            self.traces.append(diag)

    @property
    def mean_d_inter(self) -> float:
        finite = self.window_count - (1 if self.window_count > 0 else 0)
        return self.d_inter_sum / max(finite, 1)

    @property
    def mean_var_intra(self) -> float:
        return self.var_intra_sum / max(self.window_count, 1)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "full": self.full_count,
            "collapse": self.collapse_count,
            "reuse": self.reuse_count,
            "cache_refresh": self.cache_refresh_count,
            "cache_invalidation": self.cache_invalidation_count,
            "mean_d_inter": round(self.mean_d_inter, 6),
            "mean_var_intra": round(self.mean_var_intra, 6),
            "window_count": self.window_count,
        }


class Profiler:
    """Global profiling collector, shared across all wrapped layers."""

    def __init__(self, record_traces: bool = False) -> None:
        self.record_traces = record_traces
        self._layers: Dict[str, LayerStats] = {}

    def get_or_create(self, layer_name: str) -> LayerStats:
        if layer_name not in self._layers:
            self._layers[layer_name] = LayerStats()
        return self._layers[layer_name]

    def record(self, layer_name: str, diag: Dict[str, Any]) -> None:
        stats = self.get_or_create(layer_name)
        stats.record(diag, record_trace=self.record_traces)

    def reset(self) -> None:
        self._layers.clear()

    @property
    def layer_names(self) -> List[str]:
        return list(self._layers.keys())

    def layer_summary(self, layer_name: str) -> Dict[str, Any]:
        if layer_name not in self._layers:
            return {}
        return self._layers[layer_name].summary_dict()

    def global_summary(self) -> Dict[str, Any]:
        total = LayerStats()
        for ls in self._layers.values():
            total.full_count += ls.full_count
            total.collapse_count += ls.collapse_count
            total.reuse_count += ls.reuse_count
            total.cache_refresh_count += ls.cache_refresh_count
            total.cache_invalidation_count += ls.cache_invalidation_count
            total.d_inter_sum += ls.d_inter_sum
            total.var_intra_sum += ls.var_intra_sum
            total.window_count += ls.window_count
        s = total.summary_dict()
        s["num_layers"] = len(self._layers)
        return s

    def all_layer_summaries(self) -> Dict[str, Dict[str, Any]]:
        return {name: stats.summary_dict() for name, stats in self._layers.items()}

    def all_traces(self) -> Dict[str, List[Dict[str, Any]]]:
        return {name: stats.traces for name, stats in self._layers.items() if stats.traces}