"""Tests/test_router.py – router decision tree verification."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from tempocache.config import RouterConfig
from tempocache.runtime.cache_state import CacheState
from tempocache.runtime.router import route


def _sig(val=1.0, dim=13):
    return torch.full((dim,), val)


def test_first_window_forces_full():
    """No previous signature → mode must be Full."""
    cs = CacheState()
    mode, _, diag = route(_sig(), None, cs, var_intra=0.0, config=RouterConfig())
    assert mode == "full", f"Expected full, got {mode}"
    print("[PASS] first window → full")


def test_hard_invalidation():
    """d_inter >= tau_reuse_hard → cache invalidated → Full."""
    cs = CacheState(valid=True, age=0, last_signature=_sig(1.0))
    cfg = RouterConfig(tau_reuse_hard=0.5)
    sig_far = _sig(100.0)  # very different
    mode, _, diag = route(sig_far, cs.last_signature, cs, var_intra=0.0, config=cfg)
    assert mode == "full"
    assert not cs.valid
    print("[PASS] hard invalidation → full")


def test_reuse_triggered():
    """Small d_inter + small var_intra + valid cache → Reuse."""
    sig_prev = _sig(1.0)
    sig_cur = _sig(1.001)  # very close
    cs = CacheState(valid=True, age=0, last_signature=sig_prev,
                    entry=torch.zeros(1))
    cfg = RouterConfig(tau_reuse=0.05, tau_stable=0.02)
    mode, _, diag = route(sig_cur, sig_prev, cs, var_intra=0.001, config=cfg)
    assert mode == "reuse", f"Expected reuse, got {mode}"
    print("[PASS] small d_inter + small var → reuse")


def test_collapse_triggered():
    """d_inter too large for reuse, but var_intra <= tau_collapse → Collapse."""
    sig_prev = _sig(1.0)
    sig_cur = _sig(1.04)  # d_inter ~ 0.04, just below tau_reuse_hard but above tau_reuse
    cs = CacheState(valid=True, age=0, last_signature=sig_prev,
                    entry=torch.zeros(1))
    cfg = RouterConfig(tau_reuse=0.01, tau_collapse=0.10, tau_stable=0.001)
    mode, _, diag = route(sig_cur, sig_prev, cs, var_intra=0.05, config=cfg)
    assert mode == "collapse", f"Expected collapse, got {mode}"
    print("[PASS] moderate d_inter + moderate var → collapse")


def test_max_reuse_forces_full():
    """cache_age >= MAX_REUSE → Full."""
    cs = CacheState(valid=True, age=4, last_signature=_sig(1.0),
                    entry=torch.zeros(1))
    cfg = RouterConfig(max_reuse=4)
    mode, _, _ = route(_sig(1.0), cs.last_signature, cs, var_intra=0.0, config=cfg)
    assert mode == "full"
    print("[PASS] max_reuse exceeded → full")


def test_fallback_to_full():
    """High d_inter + high var_intra → Full (catch-all)."""
    sig_prev = _sig(1.0)
    sig_cur = _sig(1.03)
    cs = CacheState(valid=True, age=0, last_signature=sig_prev,
                    entry=torch.zeros(1))
    cfg = RouterConfig(tau_reuse=0.01, tau_collapse=0.02, tau_stable=0.001)
    mode, _, _ = route(sig_cur, sig_prev, cs, var_intra=0.5, config=cfg)
    assert mode == "full"
    print("[PASS] high var → full fallback")


if __name__ == "__main__":
    test_first_window_forces_full()
    test_hard_invalidation()
    test_reuse_triggered()
    test_collapse_triggered()
    test_max_reuse_forces_full()
    test_fallback_to_full()
    print("\nAll router tests passed.")