"""Tests/test_full_equivalence.py – Full mode must match plain Conv2d."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from tempocache.ops.tempo_conv2d import TempoConv2d


def test_full_equivalence():
    """TempoConv2d(forced_mode='full') must produce the same output as
    running nn.Conv2d on each timestep manually."""
    torch.manual_seed(123)

    conv = nn.Conv2d(8, 16, 3, padding=1, bias=True)
    conv.eval()
    for p in conv.parameters(): p.requires_grad_(False)

    tc = TempoConv2d(conv, forced_mode="full")
    x = torch.randn(4, 2, 8, 8, 8)

    # Reference: manual per-timestep conv
    with torch.no_grad():
        ref = torch.stack([conv(x[t]) for t in range(4)], dim=0)

    # TempoConv2d Full
    y, mode, _, _ = tc.forward_with_diag(x)

    diff = (ref - y).abs()
    assert diff.max().item() < 1e-6, f"Full-mode diverged: max diff = {diff.max().item()}"
    print(f"[PASS] Full equivalence: max diff = {diff.max().item():.2e}")


def test_full_equivalence_k2():
    """Same test with K=2."""
    torch.manual_seed(456)
    conv = nn.Conv2d(4, 8, 3, padding=1)
    conv.eval()
    for p in conv.parameters(): p.requires_grad_(False)

    tc = TempoConv2d(conv, forced_mode="full")
    x = torch.randn(2, 3, 4, 6, 6)

    with torch.no_grad():
        ref = torch.stack([conv(x[t]) for t in range(2)], dim=0)

    y, _, _, _ = tc.forward_with_diag(x)
    diff = (ref - y).abs()
    assert diff.max().item() < 1e-6
    print(f"[PASS] Full equivalence K=2: max diff = {diff.max().item():.2e}")


if __name__ == "__main__":
    test_full_equivalence()
    test_full_equivalence_k2()
    print("\nAll full-equivalence tests passed.")