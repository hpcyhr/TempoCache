"""TempoMatMul operator."""

from __future__ import annotations

import torch

from ..config import RuntimeConfig
from ..core.base import TempoBinaryBaseHDO
from ..core.signatures.pairwise import PairwiseTensorSignatureExtractor
from ..executors.pairwise import MatMulExecutor


class TempoMatMul(TempoBinaryBaseHDO):
    """Tempo-cached torch.matmul HDO."""

    def __init__(self, runtime_config: RuntimeConfig, module_name: str = "tempo_matmul") -> None:
        super().__init__(
            runtime_config=runtime_config,
            operator_family="matmul",
            module_name=module_name,
            signature_extractor=PairwiseTensorSignatureExtractor(),
            executor=MatMulExecutor(),
        )

    def forward_single_step(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)

