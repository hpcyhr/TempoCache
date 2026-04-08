"""TempoConv1d operator."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import RuntimeConfig
from ..core.base import TempoBaseHDO
from ..core.signatures.spatial import SpatialTensorSignatureExtractor
from ..executors.conv import ConvExecutor


class TempoConv1d(TempoBaseHDO):
    """Tempo-cached Conv1d HDO."""

    def __init__(self, conv: nn.Conv1d, runtime_config: RuntimeConfig, module_name: str = "tempo_conv1d") -> None:
        super().__init__(
            runtime_config=runtime_config,
            operator_family="conv1d",
            module_name=module_name,
            signature_extractor=SpatialTensorSignatureExtractor(),
            executor=ConvExecutor(),
        )
        self.conv = conv

    @classmethod
    def from_conv(cls, conv: nn.Conv1d, runtime_config: RuntimeConfig, module_name: str = "tempo_conv1d") -> "TempoConv1d":
        return cls(conv=conv, runtime_config=runtime_config, module_name=module_name)

    def forward_single_step(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

