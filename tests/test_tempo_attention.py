import torch

from tempocache.config import AttentionConfig, RuntimeConfig, ThresholdConfig
from tempocache.ops import TempoAttentionCore, TempoSpikeSelfAttention


def _sum_reuse(diag: dict) -> int:
    total = 0
    if isinstance(diag, dict):
        if "reuse_count" in diag and isinstance(diag["reuse_count"], (int, float)):
            total += int(diag["reuse_count"])
        for v in diag.values():
            if isinstance(v, dict):
                total += _sum_reuse(v)
    return total


def test_tempo_attention_projection_and_output_shapes():
    runtime = RuntimeConfig(mode="full", window_size=2)
    attn = TempoSpikeSelfAttention(embed_dim=32, num_heads=4, runtime_config=runtime, attention_config=AttentionConfig())
    x = torch.randn(6, 2, 10, 32)
    y = attn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_tempo_attention_core_qk_av_paths():
    runtime = RuntimeConfig(mode="full", window_size=2)
    core = TempoAttentionCore(runtime_config=runtime, attention_config=AttentionConfig())
    q = torch.randn(4, 2, 3, 5, 8)
    k = torch.randn(4, 2, 3, 5, 8)
    v = torch.randn(4, 2, 3, 5, 8)
    context, scores, attn = core(q, k, v)
    assert context.shape == q.shape
    assert scores.shape == (4, 2, 3, 5, 5)
    assert attn.shape == (4, 2, 3, 5, 5)


def test_tempo_attention_adaptive_triggers_reuse_and_has_diagnostics():
    th = ThresholdConfig(
        tau_reuse=1.0,
        tau_reuse_hard=10.0,
        tau_stable=1.0,
        tau_collapse=1.0,
        max_reuse=10,
        warmup_full_windows=1,
    )
    runtime = RuntimeConfig(
        mode="adaptive",
        window_size=2,
        warmup_full_windows=1,
        global_thresholds=th,
    )
    attn = TempoSpikeSelfAttention(embed_dim=16, num_heads=4, runtime_config=runtime, attention_config=AttentionConfig())
    x = torch.ones(6, 2, 8, 16)
    _ = attn(x)
    diag = attn.get_diagnostics()
    assert "module_name" in diag and "operator_family" in diag
    assert _sum_reuse(diag) > 0

