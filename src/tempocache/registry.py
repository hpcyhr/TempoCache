"""Registries for extensibility hooks."""

from __future__ import annotations

from typing import Callable, Dict, Type

import torch.nn as nn

HDO_REGISTRY: Dict[str, Type[nn.Module]] = {}
STATEFUL_REGISTRY: Dict[str, Type[nn.Module]] = {}
ATTENTION_VARIANT_REGISTRY: Dict[str, Callable] = {}


def register_hdo(name: str, module_cls: Type[nn.Module]) -> None:
    HDO_REGISTRY[name] = module_cls


def register_stateful(name: str, module_cls: Type[nn.Module]) -> None:
    STATEFUL_REGISTRY[name] = module_cls


def register_attention_variant(name: str, fn: Callable) -> None:
    ATTENTION_VARIANT_REGISTRY[name] = fn

