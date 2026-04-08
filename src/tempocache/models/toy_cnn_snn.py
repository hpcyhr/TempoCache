"""Toy CNN-like SNN for integration and benchmark."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import RuntimeConfig
from ..ops.tempo_conv2d import TempoConv2d
from ..ops.tempo_linear import TempoLinear
from .neurons import IFNode, LIFNode


class ToyCNNSNN(nn.Module):
    """Conv + neuron + conv + neuron toy SNN."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 16,
        num_classes: int = 10,
        use_tempo: bool = False,
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        super().__init__()
        runtime_config = runtime_config or RuntimeConfig(mode="full")
        conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        head = nn.Linear(hidden_channels, num_classes)

        self.conv1: nn.Module = TempoConv2d.from_conv(conv1, runtime_config, module_name="toy_cnn.conv1") if use_tempo else conv1
        self.conv2: nn.Module = TempoConv2d.from_conv(conv2, runtime_config, module_name="toy_cnn.conv2") if use_tempo else conv2
        self.head: nn.Module = TempoLinear.from_linear(head, runtime_config, module_name="toy_cnn.head") if use_tempo else head

        self.neuron1 = LIFNode()
        self.neuron2 = IFNode()
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
            raise ValueError(f"ToyCNNSNN expects [T,B,C,H,W], got {tuple(x.shape)}")
        x = self.conv1(x)
        x = self.neuron1(x)
        x = self.conv2(x)
        x = self.neuron2(x)

        t, b, c, h, w = x.shape
        pooled = self.pool(x.reshape(t * b, c, h, w)).reshape(t, b, c)
        logits = self.head(pooled)
        return logits
