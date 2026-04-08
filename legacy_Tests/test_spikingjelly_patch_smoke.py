"""Tests/test_spikingjelly_patch_smoke.py – model patching smoke test."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from tempocache.config import TempoConfig, RouterConfig
from tempocache.integration.model_patcher import patch_model
from tempocache.integration.spikingjelly_adapter import (
    build_spikingjelly_model,
    list_available_models,
    reset_spikingjelly_states,
)
from tempocache.ops.tempo_conv2d import TempoConv2d
from tempocache.runtime.profiler import Profiler
from tempocache.utils.module_utils import reset_all_caches


def test_model_patches():
    available = list_available_models()
    assert len(available) > 0, "No SpikingJelly models available"

    name = available[0]
    model = build_spikingjelly_model(name, num_classes=10)

    cfg = TempoConfig(window_size=4, forced_mode="full")
    profiler = Profiler()
    model, patched = patch_model(model, cfg, profiler)

    assert len(patched) > 0, "No layers were patched"
    print(f"[PASS] Patched {len(patched)} layers in {name}")


def test_patched_forward():
    available = list_available_models()
    name = available[0]

    model = build_spikingjelly_model(name, num_classes=10)
    cfg = TempoConfig(window_size=4, forced_mode="full")
    profiler = Profiler()
    model, _ = patch_model(model, cfg, profiler)

    x = torch.randn(4, 1, 3, 32, 32)
    reset_spikingjelly_states(model)
    reset_all_caches(model)

    with torch.no_grad():
        y = model(x)

    assert y is not None
    assert y.dim() >= 2
    print(f"[PASS] Patched forward: output shape = {y.shape}")


def test_whitelist_filter():
    available = list_available_models()
    name = available[0]

    model = build_spikingjelly_model(name, num_classes=10)
    cfg = TempoConfig(window_size=4, forced_mode="full",
                      wrap_whitelist=["features.0"])
    profiler = Profiler()
    model, patched = patch_model(model, cfg, profiler)

    assert len(patched) == 1
    assert patched[0] == "features.0"
    print(f"[PASS] Whitelist filter: only {patched}")


def test_blacklist_filter():
    available = list_available_models()
    name = available[0]

    model = build_spikingjelly_model(name, num_classes=10)
    cfg = TempoConfig(window_size=4, forced_mode="full",
                      wrap_blacklist=["features.0"])
    profiler = Profiler()
    model, patched = patch_model(model, cfg, profiler)

    assert "features.0" not in patched
    assert len(patched) > 0
    print(f"[PASS] Blacklist filter: excluded features.0, patched {len(patched)}")


def test_max_layers():
    available = list_available_models()
    name = available[0]

    model = build_spikingjelly_model(name, num_classes=10)
    cfg = TempoConfig(window_size=4, forced_mode="full", wrap_max_layers=2)
    profiler = Profiler()
    model, patched = patch_model(model, cfg, profiler)

    assert len(patched) == 2
    print(f"[PASS] max_layers=2: patched {patched}")


if __name__ == "__main__":
    test_model_patches()
    test_patched_forward()
    test_whitelist_filter()
    test_blacklist_filter()
    test_max_layers()
    print("\nAll SpikingJelly patch smoke tests passed.")