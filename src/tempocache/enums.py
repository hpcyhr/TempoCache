"""Core enums used by TempoCache."""

from __future__ import annotations

from enum import Enum


class RoutingMode(str, Enum):
    """Per-window execution mode for a sample."""

    FULL = "FULL"
    COLLAPSE = "COLLAPSE"
    REUSE = "REUSE"


class RuntimeMode(str, Enum):
    """Top-level runtime strategy."""

    FULL = "full"
    FIXED_COLLAPSE = "fixed_collapse"
    FIXED_REUSE = "fixed_reuse"
    ADAPTIVE = "adaptive"

    @classmethod
    def from_value(cls, value: str | "RuntimeMode") -> "RuntimeMode":
        if isinstance(value, cls):
            return value
        return cls(value)


class OperatorFamily(str, Enum):
    """Recognized operator families."""

    CONV1D = "conv1d"
    CONV2D = "conv2d"
    CONV3D = "conv3d"
    LINEAR = "linear"
    MATMUL = "matmul"
    BMM = "bmm"
    ATTENTION = "attention"
    NORM = "norm"
    POOL = "pool"
    FLATTEN = "flatten"
    IDENTITY = "identity"
    RESIDUAL = "residual"
    RESHAPE = "reshape"
    STATEFUL = "stateful"
    OTHER = "other"

