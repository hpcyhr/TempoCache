import pytest
import torch

from tempocache.config import AttentionConfig, RuntimeConfig
from tempocache.models import ToySpikeTransformer


@pytest.mark.parametrize("mode", ["full", "fixed_collapse", "fixed_reuse", "adaptive"])
def test_toy_spike_transformer_runs_all_modes(mode: str):
    runtime = RuntimeConfig(mode=mode, window_size=2, warmup_full_windows=1)
    model = ToySpikeTransformer(
        input_dim=16,
        embed_dim=32,
        num_heads=4,
        use_tempo=True,
        runtime_config=runtime,
        attention_config=AttentionConfig(),
    )
    x = torch.randn(6, 2, 12, 16)
    y = model(x)
    assert y.shape == (6, 2, 10)
    assert torch.isfinite(y).all()

