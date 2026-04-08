"""Integration wrapper helpers."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import RuntimeConfig
from ..ops.tempo_bmm import TempoBMM
from ..ops.tempo_matmul import TempoMatMul


class TempoMatMulWrapper(nn.Module):
    """Module wrapper for matmul nodes used by FX rewriting."""

    def __init__(self, runtime_config: RuntimeConfig, module_name: str) -> None:
        super().__init__()
        self.op = TempoMatMul(runtime_config=runtime_config, module_name=module_name)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.op(a, b)

    def get_diagnostics(self) -> dict:
        return self.op.get_diagnostics()


class TempoBMMWrapper(nn.Module):
    """Module wrapper for bmm nodes used by FX rewriting."""

    def __init__(self, runtime_config: RuntimeConfig, module_name: str) -> None:
        super().__init__()
        self.op = TempoBMM(runtime_config=runtime_config, module_name=module_name)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.op(a, b)

    def get_diagnostics(self) -> dict:
        return self.op.get_diagnostics()

