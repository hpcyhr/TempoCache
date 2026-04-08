"""Executor base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch

from .cache_state import CacheState


@dataclass
class ExecutionContext:
    reduction: str = "mean"


class ExecutorBase(ABC):
    """Base class for per-family grouped execution."""

    @abstractmethod
    def execute(
        self,
        *,
        modes: torch.Tensor,
        cache_state: CacheState,
        context: ExecutionContext,
        op: Callable,
        inputs: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """
        Execute a single window with grouped FULL/COLLAPSE/REUSE paths.
        Returns output in shape [K,B,...].
        """
        raise NotImplementedError

