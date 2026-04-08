"""TempoLinear operator."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import RuntimeConfig
from ..core.base import TempoBaseHDO
from ..core.signatures.vector import VectorSignatureExtractor
from ..executors.linear import LinearExecutor


class TempoLinear(TempoBaseHDO):
    """Tempo-cached Linear HDO."""

    def __init__(self, linear: nn.Linear, runtime_config: RuntimeConfig, module_name: str = "tempo_linear") -> None:
        super().__init__(
            runtime_config=runtime_config,
            operator_family="linear",
            module_name=module_name,
            signature_extractor=VectorSignatureExtractor(),
            executor=LinearExecutor(),
        )
        self.linear = linear

    @classmethod
    def from_linear(cls, linear: nn.Linear, runtime_config: RuntimeConfig, module_name: str = "tempo_linear") -> "TempoLinear":
        return cls(linear=linear, runtime_config=runtime_config, module_name=module_name)

    def forward_single_step(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

