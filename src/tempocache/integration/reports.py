"""Reporting helpers for patching, FX rewriting, and diagnostics export."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch.nn as nn

from ..utils.export import dump_csv_rows, dump_json


@dataclass
class PatchReport:
    """Report produced by module-based patching."""

    hdo_replaced: list[dict] = field(default_factory=list)
    recognized_non_hdo: list[dict] = field(default_factory=list)
    stateful_preserved: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "hdo_replaced": list(self.hdo_replaced),
            "recognized_non_hdo": list(self.recognized_non_hdo),
            "stateful_preserved": list(self.stateful_preserved),
        }


@dataclass
class FXRewriteReport:
    """Report produced by FX graph rewriter."""

    hdo_nodes_replaced: list[dict] = field(default_factory=list)
    non_hdo_recognized: list[dict] = field(default_factory=list)
    stateful_nodes_preserved: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "hdo_nodes_replaced": list(self.hdo_nodes_replaced),
            "non_hdo_recognized": list(self.non_hdo_recognized),
            "stateful_nodes_preserved": list(self.stateful_nodes_preserved),
        }


def collect_module_diagnostics(model: nn.Module) -> list[dict]:
    rows: list[dict] = []
    for name, module in model.named_modules():
        if hasattr(module, "get_diagnostics"):
            diag = module.get_diagnostics()
            if isinstance(diag, dict):
                row = {"module_path": name}
                row.update(diag)
                rows.append(row)
    return rows


def summarize_diagnostics(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    summary = {
        "module_count": len(rows),
        "total_windows": 0,
        "total_samples": 0,
        "full_count": 0,
        "collapse_count": 0,
        "reuse_count": 0,
        "cache_hit_count": 0,
    }
    for row in rows:
        for key in ("total_windows", "total_samples", "full_count", "collapse_count", "reuse_count", "cache_hit_count"):
            if key in row and isinstance(row[key], (int, float)):
                summary[key] += row[key]
    return summary


def export_diagnostics(model: nn.Module, export_dir: str, prefix: str = "tempocache_diag") -> dict:
    rows = collect_module_diagnostics(model)
    summary = summarize_diagnostics(rows)
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{prefix}_modules.json"
    csv_path = out_dir / f"{prefix}_modules.csv"
    summary_path = out_dir / f"{prefix}_summary.json"
    dump_json(json_path, {"modules": rows})
    dump_csv_rows(csv_path, rows)
    dump_json(summary_path, summary)
    return {
        "modules_json": str(json_path),
        "modules_csv": str(csv_path),
        "summary_json": str(summary_path),
        "summary": summary,
    }

