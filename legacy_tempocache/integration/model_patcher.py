"""
tempocache/integration/model_patcher.py

Generic model patching utility.

Walks the module tree of any nn.Module, finds eligible nn.Conv2d layers
(including SpikingJelly's layer.Conv2d), and replaces them in-place with
TempoConv2d wrappers.

Supports whitelist, blacklist, and max-layer selection.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch.nn as nn

from tempocache.config.default_config import RouterConfig, TempoConfig
from tempocache.ops.tempo_conv2d import TempoConv2d
from tempocache.runtime.profiler import Profiler
from tempocache.utils.module_utils import find_conv2d_modules


def _is_eligible(
    name: str,
    whitelist: List[str],
    blacklist: List[str],
) -> bool:
    """Check if a layer name passes whitelist / blacklist filters."""
    if blacklist:
        for prefix in blacklist:
            if name.startswith(prefix) or name == prefix:
                return False
    if whitelist:
        for prefix in whitelist:
            if name.startswith(prefix) or name == prefix:
                return True
        return False  # whitelist is non-empty but no match
    return True


def _set_submodule(model: nn.Module, target: str, new_module: nn.Module) -> None:
    """Replace a nested submodule by dotted name, e.g. 'features.0'."""
    parts = target.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]  # type: ignore[index]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module  # type: ignore[index]
    else:
        setattr(parent, last, new_module)


def patch_model(
    model: nn.Module,
    config: Optional[TempoConfig] = None,
    profiler: Optional[Profiler] = None,
) -> Tuple[nn.Module, List[str]]:
    """Patch *model* in-place, replacing Conv2d layers with TempoConv2d.

    Parameters
    ----------
    model    : the target model
    config   : TempoConfig (uses defaults if None)
    profiler : shared Profiler (created internally if None)

    Returns
    -------
    model          : the patched model (same object, modified in-place)
    patched_names  : list of module names that were wrapped
    """
    cfg = config or TempoConfig()
    if profiler is None:
        profiler = Profiler(record_traces=cfg.record_per_window_trace)

    conv_layers = find_conv2d_modules(model)
    patched_names: List[str] = []
    count = 0

    for name, conv_mod in conv_layers:
        # Already wrapped?
        if isinstance(conv_mod, TempoConv2d):
            continue
        if not _is_eligible(name, cfg.wrap_whitelist, cfg.wrap_blacklist):
            continue
        if cfg.wrap_max_layers > 0 and count >= cfg.wrap_max_layers:
            break

        wrapper = TempoConv2d(
            conv=conv_mod,
            config=cfg.router,
            grid_size=cfg.grid_size,
            forced_mode=cfg.forced_mode,
            layer_name=name,
            profiler=profiler,
        )
        _set_submodule(model, name, wrapper)
        patched_names.append(name)
        count += 1

    return model, patched_names


def print_patch_summary(model: nn.Module, patched_names: List[str]) -> None:
    """Print a human-readable patching summary."""
    total_conv = len(find_conv2d_modules(model))
    print(f"  Patched {len(patched_names)} / {total_conv + len(patched_names)} Conv2d layers")
    for n in patched_names:
        print(f"    ✓ {n}")