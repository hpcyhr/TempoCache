"""Conv family executor."""

from __future__ import annotations

import torch

from ..core.cache_state import CacheState
from ..core.executor_base import ExecutionContext, ExecutorBase
from ..core.router import MODE_COLLAPSE, MODE_FULL, MODE_REUSE
from ..utils.tensor_ops import collapse_window


class ConvExecutor(ExecutorBase):
    """Grouped FULL/COLLAPSE/REUSE execution for Conv1d/2d/3d."""

    def execute(
        self,
        *,
        modes: torch.Tensor,
        cache_state: CacheState,
        context: ExecutionContext,
        op,
        inputs: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        (x_win,) = inputs
        k, bsz = x_win.shape[:2]
        device = x_win.device

        full_idx = torch.where(modes == MODE_FULL)[0]
        collapse_idx = torch.where(modes == MODE_COLLAPSE)[0]
        reuse_idx = torch.where(modes == MODE_REUSE)[0]
        outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        if full_idx.numel() > 0:
            x_full = x_win[:, full_idx]
            x_flat = x_full.reshape(k * full_idx.numel(), *x_full.shape[2:])
            y_flat = op(x_flat)
            y_full = y_flat.reshape(k, full_idx.numel(), *y_flat.shape[1:])
            outputs["full"] = (full_idx, y_full)
            cache_state.update_drive(full_idx, y_full.mean(dim=0))

        if collapse_idx.numel() > 0:
            x_col = collapse_window(x_win[:, collapse_idx], reduction=context.reduction)
            y_col = op(x_col)
            y_win = y_col.unsqueeze(0).expand(k, *y_col.shape)
            outputs["collapse"] = (collapse_idx, y_win)
            cache_state.update_drive(collapse_idx, y_col)

        if reuse_idx.numel() > 0:
            if cache_state.cache is None:
                x_full = x_win[:, reuse_idx]
                x_flat = x_full.reshape(k * reuse_idx.numel(), *x_full.shape[2:])
                y_flat = op(x_flat)
                y_reuse = y_flat.reshape(k, reuse_idx.numel(), *y_flat.shape[1:])
                cache_state.update_drive(reuse_idx, y_reuse.mean(dim=0))
            else:
                y_reuse = cache_state.cache[reuse_idx].unsqueeze(0).expand(k, *cache_state.cache[reuse_idx].shape)
            outputs["reuse"] = (reuse_idx, y_reuse)

        first = next(iter(outputs.values()))[1]
        y_win = torch.empty((k, bsz, *first.shape[2:]), device=device, dtype=first.dtype)
        for idx, y_part in outputs.values():
            y_win[:, idx] = y_part
        return y_win

