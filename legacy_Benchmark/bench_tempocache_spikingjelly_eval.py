#!/usr/bin/env python3
"""
Benchmark/bench_tempocache_spikingjelly_eval.py

Multi-policy evaluation on SpikingJelly models.
In-place patching: ONE model, switch policies via set_forced_mode().

Usage:
  python bench_tempocache_spikingjelly_eval.py --smoke
  python bench_tempocache_spikingjelly_eval.py --dataset dvs128_gesture --data-root ./data
"""

from __future__ import annotations

import argparse
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


def try_load_dataset(name, root="./data", T=4):
    try:
        if name == "dvs128_gesture":
            from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
            ds = DVS128Gesture(root=root, train=False, data_type='frame', frames_number=T)
            return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False), \
                   {"name": name, "num_classes": 11}
        elif name == "cifar10_dvs":
            from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
            ds = CIFAR10DVS(root=root, data_type='frame', frames_number=T)
            return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False), \
                   {"name": name, "num_classes": 10}
        return None, f"Unknown: {name}"
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--export-dir", default="tempocache_artifacts/eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    available = list_available_models()
    if not available:
        print("No SpikingJelly models available."); return

    model_name = args.model
    if model_name == "auto":
        for p in ["spiking_vgg11_bn", "spiking_resnet18"]:
            if p in available: model_name = p; break
        else: model_name = available[0]

    num_classes = 10
    C, H, W = 3, 32, 32
    use_dataset = False
    loader = None

    if args.dataset and not args.smoke:
        loader, info = try_load_dataset(args.dataset, args.data_root, args.T)
        if loader is not None:
            use_dataset = True
            num_classes = info["num_classes"]
            print(f"Dataset: {info['name']}  (classes={num_classes})")
        else:
            print(f"Dataset unavailable ({info}), using random input.")

    print(f"Model: {model_name}  |  T={args.T}  K={args.K}  |  {'dataset' if use_dataset else 'random'}")

    # --- Prepare input batches ---
    x_batches = []
    for _ in range(args.num_batches):
        if use_dataset:
            try:
                xd, _ = next(iter(loader))
                if xd.dim() == 5 and xd.shape[1] == args.T:
                    x_batches.append(xd.permute(1, 0, 2, 3, 4).float())
                else:
                    x_batches.append(xd.float())
            except StopIteration:
                break
        else:
            x_batches.append(torch.randn(args.T, 1, C, H, W))

    # --- Build model ---
    model = build_spikingjelly_model(model_name, num_classes=num_classes)

    # --- Unpatched baseline ---
    print("\n--- Unpatched baseline ---")
    with Timer() as t_up:
        for xb in x_batches:
            _reset_and_run(model, xb)
    avg_up = t_up.elapsed / max(len(x_batches), 1) * 1000
    print(f"  avg latency: {avg_up:.2f} ms  ({len(x_batches)} batches)")

    results = {"unpatched": {"policy": "unpatched", "batches": len(x_batches),
                             "avg_latency_ms": round(avg_up, 2), "patched_layers": 0}}

    # --- Patch in-place ---
    router_cfg = RouterConfig()
    cfg = TempoConfig(window_size=args.K, forced_mode="full", router=router_cfg)
    dummy = Profiler()
    model, patched_names = patch_model(model, cfg, dummy)
    print_patch_summary(model, patched_names)

    # --- Run each policy ---
    policies = [("full", "full"), ("collapse", "collapse"),
                ("reuse", "reuse"), ("adaptive", None)]

    for pname, fmode in policies:
        print(f"\n--- Policy: {pname} ---")
        prof = Profiler()
        set_forced_mode(model, fmode)
        _attach_profiler(model, prof)

        with Timer() as t_pol:
            for bi, xb in enumerate(x_batches):
                if pname == "reuse" and bi == 0:
                    set_forced_mode(model, "full")
                    _attach_profiler(model, Profiler())
                    _reset_and_run(model, xb)
                    set_forced_mode(model, "reuse")
                    _attach_profiler(model, prof)
                    continue
                _reset_and_run(model, xb)

        nb = len(x_batches) - (1 if pname == "reuse" else 0)
        avg = t_pol.elapsed / max(nb, 1) * 1000

        entry = {"policy": pname, "batches": nb,
                 "avg_latency_ms": round(avg, 2), "patched_layers": len(patched_names)}
        entry.update(prof.global_summary())
        results[pname] = entry
        print(f"  avg latency: {avg:.2f} ms  |  {prof.global_summary()}")

    # --- Summary table ---
    print(f"\n{'='*60}")
    print("  POLICY COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Policy':<12} {'Latency':>10} {'Full':>6} {'Coll':>6} {'Reuse':>6}")
    print("  " + "-" * 45)
    for r in results.values():
        lat = f"{r['avg_latency_ms']:.1f} ms"
        print(f"  {r['policy']:<12} {lat:>10} {str(r.get('full','-')):>6} "
              f"{str(r.get('collapse','-')):>6} {str(r.get('reuse','-')):>6}")
    print(f"{'='*60}")

    # --- Export ---
    os.makedirs(args.export_dir, exist_ok=True)
    export_json(results, os.path.join(args.export_dir, "eval_summary.json"))
    export_csv(list(results.values()), os.path.join(args.export_dir, "eval_comparison.csv"))
    print("\nDone.")


if __name__ == "__main__":
    main()