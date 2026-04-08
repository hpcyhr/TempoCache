"""Base classes for TempoCache HDO operators."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn

from ..config import RuntimeConfig
from ..utils.tensor_ops import ensure_time_batch
from .cache_state import CacheState
from .diagnostics import ModuleDiagnostics
from .distance import inter_signature_distance
from .executor_base import ExecutionContext, ExecutorBase
from .router import MODE_REUSE, TempoRouter


class TempoBaseHDO(nn.Module):
    """
    Base class for unary HDO modules with temporal window routing.
    """

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig,
        operator_family: str,
        module_name: str,
        signature_extractor: Any,
        executor: ExecutorBase,
    ) -> None:
        super().__init__()
        self.runtime_config = runtime_config
        self.operator_family = operator_family
        self.module_name = module_name
        self.signature_extractor = signature_extractor
        self.executor = executor
        self.router = TempoRouter(runtime_config)
        self.cache_state = CacheState()
        self.diagnostics = ModuleDiagnostics(
            module_name=module_name,
            operator_family=operator_family,
            keep_window_history=runtime_config.keep_window_history,
            history_max_windows=runtime_config.history_max_windows,
        )

    @abstractmethod
    def forward_single_step(self, x: torch.Tensor) -> torch.Tensor:
        """Forward for one logical timestep (batched samples)."""
        raise NotImplementedError

    def collapse_window(self, x_win: torch.Tensor) -> torch.Tensor:
        """Default temporal collapse for input window."""
        return x_win.mean(dim=0)

    def _compute_signature(self, x_win: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.signature_extractor.extract(x_win)

    def update_cache(self, indices: torch.Tensor, drive: torch.Tensor) -> None:
        self.cache_state.update_drive(indices, drive)

    def get_diagnostics(self) -> dict:
        return self.diagnostics.to_dict()

    def reset_cache(self) -> None:
        self.cache_state.reset()

    def forward_window(self, x_win: torch.Tensor, *, window_index: int) -> torch.Tensor:
        ensure_time_batch(x_win)
        bsz = x_win.shape[1]
        device = x_win.device
        self.cache_state.ensure_batch(bsz, device=device)
        assert self.cache_state.valid is not None and self.cache_state.age is not None

        sig_cur, v_temp = self._compute_signature(x_win)
        if self.cache_state.signature is None or self.cache_state.signature.shape != sig_cur.shape:
            sig_prev = torch.zeros_like(sig_cur)
        else:
            sig_prev = self.cache_state.signature
        d_inter = inter_signature_distance(sig_cur, sig_prev)

        thresholds = self.runtime_config.thresholds_for(self.operator_family, self.module_name)
        routing = self.router.route(
            window_index=window_index,
            d_inter=d_inter,
            v_temp=v_temp,
            cache_valid=self.cache_state.valid,
            cache_age=self.cache_state.age,
            thresholds=thresholds,
        )

        if routing.hard_invalidation_mask.any():
            self.cache_state.invalidate(routing.hard_invalidation_mask)

        y_win = self.executor.execute(
            modes=routing.modes,
            cache_state=self.cache_state,
            context=ExecutionContext(reduction=self.runtime_config.collapse_reduction),
            op=self.forward_single_step,
            inputs=(x_win,),
        )

        refresh_mask = routing.modes != MODE_REUSE
        reuse_mask = routing.modes == MODE_REUSE
        refresh_idx = torch.where(refresh_mask)[0]
        if refresh_idx.numel() > 0:
            self.cache_state.update_signature(refresh_idx, sig_cur[refresh_idx])
        self.cache_state.increment_age(mask=reuse_mask)

        if self.runtime_config.enable_diag:
            self.diagnostics.record(
                modes=routing.modes,
                d_inter=d_inter,
                v_temp=v_temp,
                hard_invalidation_mask=routing.hard_invalidation_mask,
                forced_refresh_mask=routing.forced_refresh_mask,
            )
        return y_win

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ensure_time_batch(x)
        t_total = x.shape[0]
        k = self.runtime_config.window_size
        outputs = []
        window_index = 0
        for start in range(0, t_total, k):
            end = min(t_total, start + k)
            x_win = x[start:end]
            outputs.append(self.forward_window(x_win, window_index=window_index))
            window_index += 1
        return torch.cat(outputs, dim=0)


class TempoBinaryBaseHDO(nn.Module):
    """
    Base class for binary HDO modules (MatMul/BMM/Attention core pieces).
    """

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig,
        operator_family: str,
        module_name: str,
        signature_extractor: Any,
        executor: ExecutorBase,
    ) -> None:
        super().__init__()
        self.runtime_config = runtime_config
        self.operator_family = operator_family
        self.module_name = module_name
        self.signature_extractor = signature_extractor
        self.executor = executor
        self.router = TempoRouter(runtime_config)
        self.cache_state = CacheState()
        self.diagnostics = ModuleDiagnostics(
            module_name=module_name,
            operator_family=operator_family,
            keep_window_history=runtime_config.keep_window_history,
            history_max_windows=runtime_config.history_max_windows,
        )

    @abstractmethod
    def forward_single_step(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def collapse_window(self, a_win: torch.Tensor, b_win: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return a_win.mean(dim=0), b_win.mean(dim=0)

    def _compute_signature(self, a_win: torch.Tensor, b_win: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.signature_extractor.extract(a_win, b_win)

    def update_cache(self, indices: torch.Tensor, drive: torch.Tensor) -> None:
        self.cache_state.update_drive(indices, drive)

    def get_diagnostics(self) -> dict:
        return self.diagnostics.to_dict()

    def reset_cache(self) -> None:
        self.cache_state.reset()

    def forward_window(self, a_win: torch.Tensor, b_win: torch.Tensor, *, window_index: int) -> torch.Tensor:
        ensure_time_batch(a_win)
        ensure_time_batch(b_win)
        if a_win.shape[:2] != b_win.shape[:2]:
            raise ValueError(f"A/B window mismatch: {a_win.shape[:2]} vs {b_win.shape[:2]}")

        bsz = a_win.shape[1]
        device = a_win.device
        self.cache_state.ensure_batch(bsz, device=device)
        assert self.cache_state.valid is not None and self.cache_state.age is not None

        sig_cur, v_temp = self._compute_signature(a_win, b_win)
        if self.cache_state.signature is None or self.cache_state.signature.shape != sig_cur.shape:
            sig_prev = torch.zeros_like(sig_cur)
        else:
            sig_prev = self.cache_state.signature
        d_inter = inter_signature_distance(sig_cur, sig_prev)

        thresholds = self.runtime_config.thresholds_for(self.operator_family, self.module_name)
        routing = self.router.route(
            window_index=window_index,
            d_inter=d_inter,
            v_temp=v_temp,
            cache_valid=self.cache_state.valid,
            cache_age=self.cache_state.age,
            thresholds=thresholds,
        )

        if routing.hard_invalidation_mask.any():
            self.cache_state.invalidate(routing.hard_invalidation_mask)

        y_win = self.executor.execute(
            modes=routing.modes,
            cache_state=self.cache_state,
            context=ExecutionContext(reduction=self.runtime_config.collapse_reduction),
            op=self.forward_single_step,
            inputs=(a_win, b_win),
        )

        refresh_mask = routing.modes != MODE_REUSE
        reuse_mask = routing.modes == MODE_REUSE
        refresh_idx = torch.where(refresh_mask)[0]
        if refresh_idx.numel() > 0:
            self.cache_state.update_signature(refresh_idx, sig_cur[refresh_idx])
        self.cache_state.increment_age(mask=reuse_mask)

        if self.runtime_config.enable_diag:
            self.diagnostics.record(
                modes=routing.modes,
                d_inter=d_inter,
                v_temp=v_temp,
                hard_invalidation_mask=routing.hard_invalidation_mask,
                forced_refresh_mask=routing.forced_refresh_mask,
            )
        return y_win

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ensure_time_batch(a)
        ensure_time_batch(b)
        t_total = a.shape[0]
        if b.shape[0] != t_total:
            raise ValueError(f"Time dimension mismatch: A={a.shape[0]}, B={b.shape[0]}")
        k = self.runtime_config.window_size
        outputs = []
        window_index = 0
        for start in range(0, t_total, k):
            end = min(t_total, start + k)
            outputs.append(self.forward_window(a[start:end], b[start:end], window_index=window_index))
            window_index += 1
        return torch.cat(outputs, dim=0)

