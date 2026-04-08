"""Tests/test_tempo_conv2d.py – TempoConv2d forward semantics."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from tempocache.ops.tempo_conv2d import TempoConv2d
from tempocache.config import RouterConfig


def _make():
    conv = nn.Conv2d(8, 16, 3, padding=1, bias=True)
    conv.eval()
    for p in conv.parameters(): p.requires_grad_(False)
    return conv


def test_full_output_shape():
    conv = _make()
    tc = TempoConv2d(conv, forced_mode="full")
    x = torch.randn(4, 2, 8, 8, 8)
    y, mode, _, diag = tc.forward_with_diag(x)
    assert y.shape == (4, 2, 16, 8, 8), f"Wrong shape: {y.shape}"
    assert mode == "full"
    print("[PASS] full output shape")


def test_collapse_output_shape():
    conv = _make()
    tc = TempoConv2d(conv, forced_mode="collapse")
    x = torch.randn(4, 2, 8, 8, 8)
    y, mode, _, diag = tc.forward_with_diag(x)
    assert y.shape == (4, 2, 16, 8, 8)
    assert mode == "collapse"
    print("[PASS] collapse output shape")


def test_reuse_after_full():
    conv = _make()
    tc = TempoConv2d(conv, forced_mode="full")
    x = torch.randn(4, 2, 8, 8, 8)
    tc(x)  # populate cache
    tc.forced_mode = "reuse"
    y, mode, _, _ = tc.forward_with_diag(x)
    assert mode == "reuse"
    assert y.shape == (4, 2, 16, 8, 8)
    print("[PASS] reuse after full")


def test_reuse_fallback_without_cache():
    conv = _make()
    tc = TempoConv2d(conv, forced_mode="reuse")
    x = torch.randn(4, 2, 8, 8, 8)
    y, mode, _, diag = tc.forward_with_diag(x)
    assert mode == "full"  # should fallback
    assert diag.get("forced_fallback") is True
    print("[PASS] reuse fallback without cache")


def test_adaptive_first_window_is_full():
    conv = _make()
    tc = TempoConv2d(conv, config=RouterConfig())
    x = torch.randn(4, 2, 8, 8, 8)
    y, mode, _, _ = tc.forward_with_diag(x)
    assert mode == "full"
    print("[PASS] adaptive first window → full")


def test_diagnostics_keys():
    conv = _make()
    tc = TempoConv2d(conv, forced_mode="full")
    x = torch.randn(4, 2, 8, 8, 8)
    _, _, _, diag = tc.forward_with_diag(x)
    required = ["mode", "d_inter", "var_intra", "cache_valid_before",
                "cache_valid_after", "cache_age_before", "cache_updated",
                "cache_age_after", "signature_norm"]
    for k in required:
        assert k in diag, f"Missing diag key: {k}"
    print("[PASS] diagnostics keys present")


def test_k2_supported():
    conv = _make()
    tc = TempoConv2d(conv, forced_mode="full")
    x = torch.randn(2, 2, 8, 8, 8)
    y, mode, _, _ = tc.forward_with_diag(x)
    assert y.shape == (2, 2, 16, 8, 8)
    print("[PASS] K=2 supported")


if __name__ == "__main__":
    test_full_output_shape()
    test_collapse_output_shape()
    test_reuse_after_full()
    test_reuse_fallback_without_cache()
    test_adaptive_first_window_is_full()
    test_diagnostics_keys()
    test_k2_supported()
    print("\nAll tempo_conv2d tests passed.")