"""Smoke runner for quick end-to-end validation."""

from __future__ import annotations

import torch

from ..config import AttentionConfig, RuntimeConfig
from ..integration.reports import collect_module_diagnostics
from ..models.toy_cnn_snn import ToyCNNSNN
from ..models.toy_spike_transformer import ToySpikeTransformer
from ..utils.seed import set_seed


def run_smoke(device: str = "cpu", dtype: str = "float32") -> dict:
    set_seed(7)
    torch_dtype = getattr(torch, dtype)

    runtime = RuntimeConfig(mode="adaptive", window_size=4, warmup_full_windows=1)
    model = ToyCNNSNN(use_tempo=True, runtime_config=runtime).to(device=device, dtype=torch_dtype)
    x = torch.randn(8, 2, 3, 16, 16, device=device, dtype=torch_dtype)
    with torch.no_grad():
        y = model(x)

    attn_model = ToySpikeTransformer(
        input_dim=16,
        embed_dim=32,
        num_heads=4,
        use_tempo=True,
        runtime_config=runtime,
        attention_config=AttentionConfig(),
    ).to(device=device, dtype=torch_dtype)
    tx = torch.randn(8, 2, 12, 16, device=device, dtype=torch_dtype)
    with torch.no_grad():
        ty = attn_model(tx)

    return {
        "cnn_output_shape": list(y.shape),
        "attn_output_shape": list(ty.shape),
        "cnn_diag_modules": len(collect_module_diagnostics(model)),
        "attn_diag_modules": len(collect_module_diagnostics(attn_model)),
    }

