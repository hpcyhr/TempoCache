"""
tempocache/integration/spikingjelly_adapter.py

SpikingJelly-specific helpers for TempoCache integration.

Handles:
  - Discovering available built-in model families
  - Constructing models with correct neuron type
  - Running patched models with windowed temporal input
  - Resetting neuron states between runs
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# =====================================================================
# Model registry – maps short names to (module_path, factory_func_name)
# =====================================================================
_MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    "spiking_vgg11_bn":     ("spikingjelly.activation_based.model.spiking_vgg",    "spiking_vgg11_bn"),
    "spiking_vgg13_bn":     ("spikingjelly.activation_based.model.spiking_vgg",    "spiking_vgg13_bn"),
    "spiking_vgg16_bn":     ("spikingjelly.activation_based.model.spiking_vgg",    "spiking_vgg16_bn"),
    "spiking_resnet18":     ("spikingjelly.activation_based.model.spiking_resnet", "spiking_resnet18"),
    "spiking_resnet34":     ("spikingjelly.activation_based.model.spiking_resnet", "spiking_resnet34"),
    "sew_resnet18":         ("spikingjelly.activation_based.model.sew_resnet",     "sew_resnet18"),
    "sew_resnet34":         ("spikingjelly.activation_based.model.sew_resnet",     "sew_resnet34"),
}


def list_available_models() -> List[str]:
    """Return model short-names that are actually importable."""
    import importlib

    available = []
    for short_name, (mod_path, func_name) in _MODEL_REGISTRY.items():
        try:
            m = importlib.import_module(mod_path)
            if hasattr(m, func_name):
                available.append(short_name)
        except ImportError:
            pass
    return available


def build_spikingjelly_model(
    name: str,
    num_classes: int = 10,
    neuron_type: str = "IFNode",
    **kwargs: Any,
) -> nn.Module:
    """Construct a SpikingJelly model in multi-step mode.

    Parameters
    ----------
    name        : short name from the registry (e.g. "spiking_vgg11_bn")
    num_classes : output classes
    neuron_type : "IFNode" or "LIFNode"
    **kwargs    : extra keyword arguments forwarded to the factory

    Returns
    -------
    model in step_mode='m', eval mode
    """
    import importlib
    from spikingjelly.activation_based import functional, neuron as sj_neuron

    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}")

    mod_path, func_name = _MODEL_REGISTRY[name]
    mod = importlib.import_module(mod_path)
    factory = getattr(mod, func_name)

    neuron_cls = getattr(sj_neuron, neuron_type)

    # SEW-ResNet requires 'cnf' argument
    extra = {}
    if "sew_resnet" in name:
        extra["cnf"] = kwargs.pop("cnf", "ADD")

    model = factory(
        pretrained=False,
        spiking_neuron=neuron_cls,
        num_classes=num_classes,
        **extra,
        **kwargs,
    )

    functional.set_step_mode(model, "m")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def reset_spikingjelly_states(model: nn.Module) -> None:
    """Reset all neuron membrane states in a SpikingJelly model."""
    from spikingjelly.activation_based import functional
    functional.reset_net(model)


def run_patched_model(
    model: nn.Module,
    x: Tensor,
    window_size: int = 4,
) -> Tensor:
    """Run a patched SpikingJelly model on temporal input.

    The model is in step_mode='m' and expects [T, N, C, H, W].
    TempoConv2d wrappers inside it expect [K, B, C, H, W] windows.

    This function handles the windowing externally:
      1. Split T into windows of size K
      2. For each window, run model forward (SpikingJelly internal loop
         handles the multi-step unrolling for non-patched layers;
         TempoConv2d wrappers intercept their own input).

    IMPORTANT DESIGN NOTE:
    In step_mode='m', SpikingJelly's layer.Conv2d calls
    seq_to_ann_forward(x, super().forward) which reshapes [T,N]→[T*N]
    and applies Conv2d.  Since we *replaced* these layer.Conv2d modules
    with TempoConv2d, our forward() receives the [T,N,C,H,W] tensor
    directly.  TempoConv2d treats T as the window dimension K.

    So we feed the model one window at a time (T=K per call).
    """
    from tempocache.utils.module_utils import reset_all_caches

    T, N, C, H, W = x.shape
    K = window_size
    assert T >= K, f"T={T} must be >= window_size={K}"

    reset_spikingjelly_states(model)
    reset_all_caches(model)

    outputs = []
    for t0 in range(0, T - K + 1, K):
        x_win = x[t0 : t0 + K]  # [K, N, C, H, W]
        y_win = model(x_win)     # [K, N, num_classes]
        outputs.append(y_win)

    # Handle remaining timesteps (tail < K) with Full mode
    remainder = T % K
    if remainder > 0:
        # Pad to K by repeating the last frame
        x_tail = x[T - remainder:]  # [remainder, N, C, H, W]
        pad = x_tail[-1:].expand(K - remainder, -1, -1, -1, -1)
        x_padded = torch.cat([x_tail, pad], dim=0)  # [K, ...]
        y_padded = model(x_padded)
        outputs.append(y_padded[:remainder])  # only keep valid timesteps

    return torch.cat(outputs, dim=0)  # [T, N, num_classes]