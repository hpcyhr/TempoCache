import copy

import torch
import torch.nn as nn

from tempocache.config import RuntimeConfig
from tempocache.ops import TempoLinear


def _step_ref(op: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return torch.stack([op(x[t]) for t in range(x.shape[0])], dim=0)


def test_tempo_linear_full_equivalence_3d_and_4d():
    lin = nn.Linear(8, 12)
    ref = copy.deepcopy(lin)
    tempo = TempoLinear.from_linear(lin, RuntimeConfig(mode="full", window_size=3))

    x3 = torch.randn(7, 4, 8)
    y3_ref = _step_ref(ref, x3)
    y3 = tempo(x3)
    assert torch.allclose(y3, y3_ref, atol=1e-6, rtol=1e-5)

    tempo.reset_cache()
    x4 = torch.randn(6, 3, 5, 8)
    y4_ref = _step_ref(ref, x4)
    y4 = tempo(x4)
    assert torch.allclose(y4, y4_ref, atol=1e-6, rtol=1e-5)


def test_tempo_linear_fixed_collapse_behavior():
    lin = nn.Linear(6, 4)
    tempo = TempoLinear.from_linear(lin, RuntimeConfig(mode="fixed_collapse", window_size=2))
    x = torch.randn(4, 3, 6)
    y = tempo(x)
    expected = []
    for start in (0, 2):
        win = x[start : start + 2]
        col = win.mean(dim=0)
        y_col = lin(col)
        expected.append(y_col.unsqueeze(0).expand(win.shape[0], *y_col.shape))
    y_ref = torch.cat(expected, dim=0)
    assert torch.allclose(y, y_ref, atol=1e-6, rtol=1e-5)


def test_tempo_linear_fixed_reuse_behavior():
    lin = nn.Linear(6, 4)
    tempo = TempoLinear.from_linear(lin, RuntimeConfig(mode="fixed_reuse", window_size=2))
    x = torch.randn(4, 2, 6)
    y = tempo(x)
    first_win = torch.stack([lin(x[t]) for t in range(2)], dim=0)
    cache = first_win.mean(dim=0)
    expected_second = cache.unsqueeze(0).expand(2, *cache.shape)
    assert torch.allclose(y[:2], first_win, atol=1e-6, rtol=1e-5)
    assert torch.allclose(y[2:], expected_second, atol=1e-6, rtol=1e-5)

