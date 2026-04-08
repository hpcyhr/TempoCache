import pytest
import torch

from tempocache.config import RuntimeConfig
from tempocache.models import ToyCNNSNN


@pytest.mark.parametrize("mode", ["full", "fixed_collapse", "fixed_reuse", "adaptive"])
def test_toy_cnn_runs_all_modes(mode: str):
    runtime = RuntimeConfig(mode=mode, window_size=2, warmup_full_windows=1)
    model = ToyCNNSNN(use_tempo=True, runtime_config=runtime)
    x = torch.randn(6, 2, 3, 16, 16)
    y = model(x)
    assert y.shape == (6, 2, 10)
    assert torch.isfinite(y).all()

