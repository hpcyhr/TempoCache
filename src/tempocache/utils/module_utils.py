"""Module classification utilities for patching and reporting."""

from __future__ import annotations

import fnmatch
from typing import Iterable

import torch.nn as nn

from ..enums import OperatorFamily


NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
    nn.GroupNorm,
)

POOL_TYPES = (
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
)


def _match_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def should_patch_module(module_name: str, include: list[str], exclude: list[str]) -> bool:
    if exclude and _match_any(module_name, exclude):
        return False
    if include and not _match_any(module_name, include):
        return False
    return True


def is_norm_module(module: nn.Module) -> bool:
    return isinstance(module, NORM_TYPES)


def is_pool_module(module: nn.Module) -> bool:
    return isinstance(module, POOL_TYPES)


def is_stateful_temporal_module(module: nn.Module) -> bool:
    if getattr(module, "is_stateful_temporal", False):
        return True
    name = module.__class__.__name__.lower()
    stateful_tokens = ("ifnode", "lifnode", "parametriclif", "delay", "recurrent", "memory")
    return any(tok in name for tok in stateful_tokens)


def is_attention_like_module(module: nn.Module) -> bool:
    attrs = ("q_proj", "k_proj", "v_proj", "out_proj")
    return all(hasattr(module, attr) for attr in attrs)


def operator_family_of(module: nn.Module) -> OperatorFamily:
    if isinstance(module, nn.Conv1d):
        return OperatorFamily.CONV1D
    if isinstance(module, nn.Conv2d):
        return OperatorFamily.CONV2D
    if isinstance(module, nn.Conv3d):
        return OperatorFamily.CONV3D
    if isinstance(module, nn.Linear):
        return OperatorFamily.LINEAR
    if isinstance(module, nn.Flatten):
        return OperatorFamily.FLATTEN
    if isinstance(module, nn.Identity):
        return OperatorFamily.IDENTITY
    if is_norm_module(module):
        return OperatorFamily.NORM
    if is_pool_module(module):
        return OperatorFamily.POOL
    if is_stateful_temporal_module(module):
        return OperatorFamily.STATEFUL
    return OperatorFamily.OTHER

