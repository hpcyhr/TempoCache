"""Helpers around torch.fx graph manipulation."""

from __future__ import annotations

import operator
from typing import Iterable

import torch


MATMUL_FUNCS = {torch.matmul, torch.bmm}
ADD_FUNCS = {operator.add, torch.add}
CAT_FUNCS = {torch.cat}
RESHAPE_METHODS = {"reshape", "view", "permute", "transpose"}


def is_matmul_target(target: object) -> bool:
    return target in MATMUL_FUNCS


def is_add_target(target: object) -> bool:
    return target in ADD_FUNCS


def is_cat_target(target: object) -> bool:
    return target in CAT_FUNCS


def is_reshape_method(name: str) -> bool:
    return name in RESHAPE_METHODS


def ensure_tuple(value: object) -> tuple:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def readable_target(target: object) -> str:
    if hasattr(target, "__name__"):
        return str(target.__name__)
    return str(target)

