"""Attention executor for cached attention core stages."""

from __future__ import annotations

import torch

from ..core.cache_state import CacheState
from ..core.executor_base import ExecutionContext, ExecutorBase
from ..core.router import MODE_COLLAPSE, MODE_FULL, MODE_REUSE
from ..utils.tensor_ops import collapse_window


class AttentionExecutor(ExecutorBase):
    """
    Binary grouped executor used by attention sub-stages (qk score, av product).
    """

    def execute(
        self,
        *,
        modes: torch.Tensor,
        cache_state: CacheState,
        context: ExecutionContext,
        op,
        inputs: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        a_win, b_win = inputs
        k, bsz = a_win.shape[:2]
        device = a_win.device

        full_idx = torch.where(modes == MODE_FULL)[0]
        collapse_idx = torch.where(modes == MODE_COLLAPSE)[0]
        reuse_idx = torch.where(modes == MODE_REUSE)[0]
        outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        if full_idx.numel() > 0:
            y_full = op(a_win[:, full_idx], b_win[:, full_idx])
            outputs["full"] = (full_idx, y_full)
            cache_state.update_drive(full_idx, y_full.mean(dim=0))

        if collapse_idx.numel() > 0:
            a_col = collapse_window(a_win[:, collapse_idx], reduction=context.reduction)
            b_col = collapse_window(b_win[:, collapse_idx], reduction=context.reduction)
            y_col = op(a_col, b_col)
            y_win = y_col.unsqueeze(0).expand(k, *y_col.shape)
            outputs["collapse"] = (collapse_idx, y_win)
            cache_state.update_drive(collapse_idx, y_col)

        if reuse_idx.numel() > 0:
            if cache_state.cache is None:
                y_reuse = op(a_win[:, reuse_idx], b_win[:, reuse_idx])
                cache_state.update_drive(reuse_idx, y_reuse.mean(dim=0))
            else:
                y_reuse = cache_state.cache[reuse_idx].unsqueeze(0).expand(k, *cache_state.cache[reuse_idx].shape)
            outputs["reuse"] = (reuse_idx, y_reuse)

        first = next(iter(outputs.values()))[1]
        y_win = torch.empty((k, bsz, *first.shape[2:]), device=device, dtype=first.dtype)
        for idx, y_part in outputs.values():
            y_win[:, idx] = y_part
        return y_win

