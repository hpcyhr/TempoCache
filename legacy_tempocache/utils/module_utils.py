"""tempocache/utils/module_utils.py – model introspection helpers."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch.nn as nn


def find_conv2d_modules(model: nn.Module) -> List[Tuple[str, nn.Conv2d]]:
    """Return (name, module) pairs for all nn.Conv2d layers in *model*.

    This also catches subclasses of nn.Conv2d (e.g. spikingjelly layer.Conv2d).
    """
    results = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            results.append((name, mod))
    return results


def reset_all_caches(model: nn.Module) -> int:
    """Call reset_cache() on every TempoConv2d in *model*.  Returns count."""
    from tempocache.ops.tempo_conv2d import TempoConv2d

    count = 0
    for mod in model.modules():
        if isinstance(mod, TempoConv2d):
            mod.reset_cache()
            count += 1
    return count


def set_forced_mode(model: nn.Module, mode: str | None) -> int:
    """Set forced_mode on every TempoConv2d.  Returns count."""
    from tempocache.ops.tempo_conv2d import TempoConv2d

    count = 0
    for mod in model.modules():
        if isinstance(mod, TempoConv2d):
            mod.forced_mode = mode
            count += 1
    return count