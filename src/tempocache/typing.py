"""Typing helpers for TempoCache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

Tensor = torch.Tensor
TensorMap = Dict[str, Tensor]
MaybeTensor = Optional[Tensor]
Shape = Tuple[int, ...]
Number = float | int
StrMap = Mapping[str, Any]
MutableStrMap = MutableMapping[str, Any]
StrList = List[str]
StrSeq = Sequence[str]


@dataclass
class RouterOutput:
    """Structured router output."""

    modes: Tensor
    hard_invalidation_mask: Tensor
    forced_refresh_mask: Tensor

