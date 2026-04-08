"""Tests/test_cache_state.py – CacheState update semantics."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from tempocache.runtime.cache_state import CacheState


def test_initial_state():
    cs = CacheState()
    assert cs.entry is None
    assert cs.valid is False
    assert cs.age == 0
    assert cs.last_signature is None
    print("[PASS] initial state")


def test_update_full():
    cs = CacheState()
    rep = torch.randn(2, 16, 8, 8)
    sig = torch.randn(13)
    cs.update_full(rep, sig)
    assert cs.valid is True
    assert cs.age == 0
    assert cs.entry is not None
    assert cs.entry.shape == (2, 16, 8, 8)
    assert cs.last_signature is not None
    print("[PASS] update_full")


def test_update_collapse():
    cs = CacheState()
    agg = torch.randn(2, 16, 8, 8)
    sig = torch.randn(13)
    cs.update_collapse(agg, sig)
    assert cs.valid is True
    assert cs.age == 0
    print("[PASS] update_collapse")


def test_update_reuse_increments_age():
    cs = CacheState(valid=True, age=2, entry=torch.zeros(1))
    sig = torch.randn(13)
    cs.update_reuse(sig)
    assert cs.age == 3
    assert cs.entry is not None  # entry unchanged
    print("[PASS] update_reuse increments age")


def test_reset():
    cs = CacheState(valid=True, age=5, entry=torch.zeros(1),
                    last_signature=torch.zeros(13))
    cs.reset()
    assert cs.valid is False
    assert cs.age == 0
    assert cs.entry is None
    assert cs.last_signature is None
    print("[PASS] reset")


def test_clone_shallow():
    cs = CacheState(valid=True, age=3, entry=torch.randn(2, 16, 8, 8))
    clone = cs.clone_shallow()
    assert clone.valid == cs.valid
    assert clone.age == cs.age
    assert clone.entry is cs.entry  # same tensor, not a deep copy
    print("[PASS] clone_shallow")


if __name__ == "__main__":
    test_initial_state()
    test_update_full()
    test_update_collapse()
    test_update_reuse_increments_age()
    test_reset()
    test_clone_shallow()
    print("\nAll cache_state tests passed.")