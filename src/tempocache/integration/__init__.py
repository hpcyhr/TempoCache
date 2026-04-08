"""Integration modules (patcher, fx rewriter, reports)."""

from .fx_rewriter import TempoFXRewriter
from .patcher import TempoPatcher, patch_model
from .reports import FXRewriteReport, PatchReport, collect_module_diagnostics, export_diagnostics

__all__ = [
    "TempoPatcher",
    "TempoFXRewriter",
    "PatchReport",
    "FXRewriteReport",
    "patch_model",
    "collect_module_diagnostics",
    "export_diagnostics",
]

