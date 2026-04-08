import torch

from tempocache.config import RuntimeConfig, ThresholdConfig
from tempocache.core.router import MODE_COLLAPSE, MODE_FULL, MODE_REUSE, TempoRouter


def _router_and_thresholds():
    cfg = RuntimeConfig(mode="adaptive", warmup_full_windows=0)
    router = TempoRouter(cfg)
    th = ThresholdConfig(
        tau_reuse=0.1,
        tau_reuse_hard=0.8,
        tau_stable=0.05,
        tau_collapse=0.12,
        max_reuse=3,
        warmup_full_windows=0,
    )
    return router, th


def test_router_invalid_cache_is_full():
    router, th = _router_and_thresholds()
    out = router.route(
        window_index=1,
        d_inter=torch.tensor([0.01, 0.01]),
        v_temp=torch.tensor([0.0, 0.0]),
        cache_valid=torch.tensor([False, False]),
        cache_age=torch.tensor([0, 0]),
        thresholds=th,
    )
    assert torch.all(out.modes == MODE_FULL)


def test_router_hard_invalidation_is_full():
    router, th = _router_and_thresholds()
    out = router.route(
        window_index=1,
        d_inter=torch.tensor([0.9]),
        v_temp=torch.tensor([0.0]),
        cache_valid=torch.tensor([True]),
        cache_age=torch.tensor([0]),
        thresholds=th,
    )
    assert int(out.modes.item()) == MODE_FULL
    assert bool(out.hard_invalidation_mask.item())


def test_router_max_reuse_forces_full():
    router, th = _router_and_thresholds()
    out = router.route(
        window_index=1,
        d_inter=torch.tensor([0.01]),
        v_temp=torch.tensor([0.0]),
        cache_valid=torch.tensor([True]),
        cache_age=torch.tensor([th.max_reuse]),
        thresholds=th,
    )
    assert int(out.modes.item()) == MODE_FULL
    assert bool(out.forced_refresh_mask.item())


def test_router_reuse_condition_triggers_reuse():
    router, th = _router_and_thresholds()
    out = router.route(
        window_index=1,
        d_inter=torch.tensor([0.02]),
        v_temp=torch.tensor([0.01]),
        cache_valid=torch.tensor([True]),
        cache_age=torch.tensor([0]),
        thresholds=th,
    )
    assert int(out.modes.item()) == MODE_REUSE


def test_router_collapse_condition_triggers_collapse():
    router, th = _router_and_thresholds()
    out = router.route(
        window_index=1,
        d_inter=torch.tensor([0.5]),
        v_temp=torch.tensor([0.08]),
        cache_valid=torch.tensor([True]),
        cache_age=torch.tensor([0]),
        thresholds=th,
    )
    assert int(out.modes.item()) == MODE_COLLAPSE

