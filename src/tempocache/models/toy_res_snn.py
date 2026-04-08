"""Toy ResNet-like SNN with residual merge."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import RuntimeConfig
from ..ops.tempo_conv2d import TempoConv2d
from ..ops.tempo_linear import TempoLinear
from .neurons import LIFNode, ParametricLIFNode


def _maybe_tempo_conv(conv: nn.Conv2d, use_tempo: bool, runtime: RuntimeConfig, name: str) -> nn.Module:
    return TempoConv2d.from_conv(conv, runtime, module_name=name) if use_tempo else conv


class ResidualSNNBlock(nn.Module):
    def __init__(self, channels: int, use_tempo: bool, runtime_config: RuntimeConfig, module_prefix: str) -> None:
        super().__init__()
        self.conv1 = _maybe_tempo_conv(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            use_tempo,
            runtime_config,
            name=f"{module_prefix}.conv1",
        )
        self.conv2 = _maybe_tempo_conv(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            use_tempo,
            runtime_config,
            name=f"{module_prefix}.conv2",
        )
        self.neuron1 = LIFNode()
        self.neuron2 = ParametricLIFNode()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.neuron1(out)
        out = self.conv2(out)
        out = out + residual
        out = self.neuron2(out)
        return out


class ToyResSNN(nn.Module):
    """Residual branch + conv blocks + neuron toy SNN."""

    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 16,
        num_classes: int = 10,
        use_tempo: bool = False,
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        super().__init__()
        runtime_config = runtime_config or RuntimeConfig(mode="full")
        self.stem = _maybe_tempo_conv(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            use_tempo,
            runtime_config,
            name="toy_res.stem",
        )
        self.stem_neuron = LIFNode()
        self.block1 = ResidualSNNBlock(channels, use_tempo=use_tempo, runtime_config=runtime_config, module_prefix="toy_res.block1")
        self.block2 = ResidualSNNBlock(channels, use_tempo=use_tempo, runtime_config=runtime_config, module_prefix="toy_res.block2")
        head = nn.Linear(channels, num_classes)
        self.head = TempoLinear.from_linear(head, runtime_config, module_name="toy_res.head") if use_tempo else head
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def reset_state(self) -> None:
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "reset_state"):
                module.reset_state()
            if hasattr(module, "reset_cache"):
                module.reset_cache()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"ToyResSNN expects [T,B,C,H,W], got {tuple(x.shape)}")
        x = self.stem(x)
        x = self.stem_neuron(x)
        x = self.block1(x)
        x = self.block2(x)
        t, b, c, h, w = x.shape
        x = self.pool(x.reshape(t * b, c, h, w)).reshape(t, b, c)
        return self.head(x)
