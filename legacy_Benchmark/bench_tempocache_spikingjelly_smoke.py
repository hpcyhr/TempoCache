#!/usr/bin/env python3
"""
Benchmark/bench_tempocache_spikingjelly_smoke.py

Model-level smoke test for TempoCache on SpikingJelly models.

Zero-copy approach: builds ONE model, runs unpatched baseline, patches
in-place, then switches policies via set_forced_mode().
"""

from __future__ import annotations

import gc
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tempocache.config import TempoConfig, RouterConfig
from tempocache.integration.model_patcher import patch_model, print_patch_summary
from tempocache.integration.spikingjelly_adapter import (
    build_spikingjelly_model,
    list_available_models,
    reset_spikingjelly_states,
)
from tempocache.ops.tempo_conv2d import TempoConv2d
from tempocache.runtime.profiler import Profiler
from tempocache.utils.io import export_json, export_csv
from tempocache.utils.module_utils import reset_all_caches, set_forced_mode
from tempocache.utils.seed import set_seed
from tempocache.utils.timing import Timer


def _reset_and_run(model, x):
    reset_spikingjelly_states(model)
    reset_all_caches(model)
    with torch.no_grad():
        return model(x)


def _attach_profiler(model, profiler):
    for mod in model.modules():
        if isinstance(mod, TempoConv2d):
            mod.profiler = profiler


def main(
    model_name: str = "auto",
    export_dir: str = "tempocache_artifacts/smoke",
) -> bool:
    set_seed(42)

    available = list_available_models()
    print(f"Available SpikingJelly models: {available}")
    if not available:
        print("ERROR: No SpikingJelly models found.")
        return False

    if model_name == "auto":
        for pref in ["spiking_vgg11_bn", "spiking_resnet18", "sew_resnet18"]:
            if pref in available:
                model_name = pref
                break
        else:
            model_name = available[0]

    print(f"\nUsing model: {model_name}")

    T, N, C, H, W = 4, 1, 3, 32, 32
    K = 4
    num_classes = 10
    x = torch.randn(T, N, C, H, W)

    # =================================================================
    # Step 1: Unpatched baseline (before patching)
    # =================================================================
    print("\n[1/6] Building model & running unpatched baseline …")
    model = build_spikingjelly_model(model_name, num_classes=num_classes)

    with Timer() as t_orig:
        y_orig = _reset_and_run(model, x)
    print(f"  output: {y_orig.shape}  ({t_orig.elapsed*1000:.1f} ms)")

    # =================================================================
    # Step 2: Patch the SAME model in-place
    # =================================================================
    print("\n[2/6] Patching model in-place …")
    cfg = TempoConfig(window_size=K, forced_mode="full", router=RouterConfig())
    profiler = Profiler(record_traces=True)
    model, patched_names = patch_model(model, cfg, profiler)
    print_patch_summary(model, patched_names)

    # =================================================================
    # Step 3: Forced Full (should match unpatched exactly)
    # =================================================================
    print("\n[3/6] Forced Full …")
    prof_full = Profiler(record_traces=True)
    set_forced_mode(model, "full")
    _attach_profiler(model, prof_full)

    with Timer() as t_full:
        y_full = _reset_and_run(model, x)

    diff_full = (y_orig - y_full).abs()
    mae_full = diff_full.mean().item()
    max_full = diff_full.max().item()
    full_equiv = mae_full < 1e-4
    print(f"  Full vs Unpatched: MAE={mae_full:.6e}  MaxAE={max_full:.6e}  ({t_full.elapsed*1000:.1f} ms)")

    # =================================================================
    # Step 4: Adaptive
    # =================================================================
    print("\n[4/6] Adaptive TempoCache …")
    prof_adapt = Profiler(record_traces=True)
    set_forced_mode(model, None)
    _attach_profiler(model, prof_adapt)

    with Timer() as t_adapt:
        y_adapt = _reset_and_run(model, x)

    mae_adapt = (y_orig - y_adapt).abs().mean().item()
    print(f"  Adaptive vs Unpatched: MAE={mae_adapt:.6e}  ({t_adapt.elapsed*1000:.1f} ms)")

    # =================================================================
    # Step 5: Fixed Collapse
    # =================================================================
    print("\n[5/6] Fixed Collapse …")
    prof_coll = Profiler(record_traces=True)
    set_forced_mode(model, "collapse")
    _attach_profiler(model, prof_coll)

    with Timer() as t_coll:
        y_coll = _reset_and_run(model, x)
    print(f"  Collapse: {y_coll.shape}  ({t_coll.elapsed*1000:.1f} ms)")

    # =================================================================
    # Step 6: Fixed Reuse
    # =================================================================
    print("\n[6/6] Fixed Reuse …")
    prof_reuse = Profiler(record_traces=True)
    set_forced_mode(model, "reuse")
    _attach_profiler(model, prof_reuse)

    with Timer() as t_reuse:
        y_reuse = _reset_and_run(model, x)
    print(f"  Reuse: {y_reuse.shape}  ({t_reuse.elapsed*1000:.1f} ms)")

    # =================================================================
    # Per-layer stats (Adaptive)
    # =================================================================
    print(f"\n{'='*60}")
    print("  PER-LAYER STATS (Adaptive)")
    print(f"{'='*60}")
    layer_rows = []
    for lname in prof_adapt.layer_names:
        s = prof_adapt.layer_summary(lname)
        layer_rows.append({"layer": lname, **s})
        print(f"  {lname}: {s}")
    gs = prof_adapt.global_summary()
    print(f"\n  Global: {gs}")

    # =================================================================
    # Verification
    # =================================================================
    print(f"\n{'='*60}")
    print("  SMOKE TEST VERIFICATION")
    print(f"{'='*60}")

    checks = [
        ("Model patches without crash", len(patched_names) > 0),
        ("Full mode matches unpatched (MAE < 1e-4)", full_equiv),
        ("Adaptive mode runs without crash", y_adapt is not None),
        ("Collapse mode runs without crash", y_coll is not None),
        ("Reuse mode runs without crash", y_reuse is not None),
        ("Diagnostics produced", len(prof_adapt.layer_names) > 0),
    ]

    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")

    all_pass = all(ok for _, ok in checks)
    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"{'='*60}")

    # =================================================================
    # Export
    # =================================================================
    os.makedirs(export_dir, exist_ok=True)
    summary = {
        "model": model_name,
        "patched_layers": patched_names,
        "input_shape": [T, N, C, H, W],
        "full_vs_unpatched_mae": mae_full,
        "full_vs_unpatched_max_ae": max_full,
        "adaptive_vs_unpatched_mae": mae_adapt,
        "timings_ms": {
            "unpatched": round(t_orig.elapsed * 1000, 2),
            "full": round(t_full.elapsed * 1000, 2),
            "adaptive": round(t_adapt.elapsed * 1000, 2),
            "collapse": round(t_coll.elapsed * 1000, 2),
            "reuse": round(t_reuse.elapsed * 1000, 2),
        },
        "adaptive_global_stats": gs,
    }
    export_json(summary, os.path.join(export_dir, "smoke_summary.json"))
    if layer_rows:
        export_csv(layer_rows, os.path.join(export_dir, "smoke_layer_stats.csv"))

    return all_pass


if __name__ == "__main__":
    model_arg = sys.argv[1] if len(sys.argv) > 1 else "auto"
    success = main(model_name=model_arg)
    sys.exit(0 if success else 1)