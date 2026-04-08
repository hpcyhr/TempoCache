"""TempoBMM operator."""

from __future__ import annotations

import torch

from ..config import RuntimeConfig
from ..core.base import TempoBinaryBaseHDO
from ..core.signatures.pairwise import PairwiseTensorSignatureExtractor
from ..executors.pairwise import BMMExecutor


class TempoBMM(TempoBinaryBaseHDO):
    """Tempo-cached torch.bmm HDO."""

    def __init__(self, runtime_config: RuntimeConfig, module_name: str = "tempo_bmm") -> None:
        super().__init__(
            runtime_config=runtime_config,
            operator_family="bmm",
            module_name=module_name,
            signature_extractor=PairwiseTensorSignatureExtractor(),
            executor=BMMExecutor(),
        )

    def forward_single_step(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a, b)

