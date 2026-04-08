import copy

import torch
import torch.nn as nn

from tempocache.config import RuntimeConfig
from tempocache.ops import TempoConv1d, TempoConv2d, TempoConv3d


def _step_ref(op: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return torch.stack([op(x[t]) for t in range(x.shape[0])], dim=0)


def test_tempo_conv1d_full_equivalence():
    conv = nn.Conv1d(4, 6, kernel_size=3, padding=1, groups=2, dilation=1, bias=True)
    ref = copy.deepcopy(conv)
    tempo = TempoConv1d.from_conv(conv, RuntimeConfig(mode="full", window_size=3))
    x = torch.randn(7, 3, 4, 32)
    y_ref = _step_ref(ref, x)
    y = tempo(x)
    assert torch.allclose(y, y_ref, atol=1e-6, rtol=1e-5)


def test_tempo_conv2d_full_equivalence():
    conv = nn.Conv2d(4, 6, kernel_size=3, padding=1, groups=2, stride=1, bias=True)
    ref = copy.deepcopy(conv)
    tempo = TempoConv2d.from_conv(conv, RuntimeConfig(mode="full", window_size=2))
    x = torch.randn(6, 2, 4, 16, 16)
    y_ref = _step_ref(ref, x)
    y = tempo(x)
    assert torch.allclose(y, y_ref, atol=1e-6, rtol=1e-5)


def test_tempo_conv3d_full_equivalence():
    conv = nn.Conv3d(2, 4, kernel_size=3, padding=1, groups=1, stride=1, bias=True)
    ref = copy.deepcopy(conv)
    tempo = TempoConv3d.from_conv(conv, RuntimeConfig(mode="full", window_size=2))
    x = torch.randn(5, 2, 2, 6, 8, 8)
    y_ref = _step_ref(ref, x)
    y = tempo(x)
    assert torch.allclose(y, y_ref, atol=1e-6, rtol=1e-5)

