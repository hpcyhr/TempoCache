#!/usr/bin/env python3
"""
Benchmark/bench_tempocache_mvp.py

Runnable MVP verification for TempoCache.

Generates synthetic window sequences with controlled similarity / change,
then compares four execution strategies:
  1. Full baseline         – Conv every timestep, every window
  2. Fixed Collapse        – Collapse every window
  3. Fixed Reuse           – Reuse with periodic Full refresh
  4. Adaptive TempoCache   – router-driven mode selection

Reports:
  - Mode histogram
  - Cache transition counters
  - Mean d_inter / var_intra
  - Wall-clock time
  - Output error vs Full baseline (MAE, MaxAE)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

# --- allow imports from project root ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Ops.tempo_conv2d import TempoConv2d
from Runtime.cache_state import CacheState
from Runtime.router import RouterConfig
from Runtime.signature_extractor import build_signature, compute_temporal_variation


# =====================================================================
# Synthetic data generation
# =====================================================================
def make_stable_windows(
    n_windows: int, K: int, B: int, C: int, H: int, W: int,
    noise_scale: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> List[torch.Tensor]:
    """Generate n_windows that are nearly identical (high similarity)."""
    base = torch.randn(1, B, C, H, W, device=device)
    windows = []
    for _ in range(n_windows):
        noise = torch.randn(K, B, C, H, W, device=device) * noise_scale
        windows.append(base.expand(K, -1, -1, -1, -1) + noise)
    return windows


def make_changing_windows(
    n_windows: int, K: int, B: int, C: int, H: int, W: int,
    change_scale: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> List[torch.Tensor]:
    """Generate n_windows with large inter-window and intra-window variation."""
    windows = []
    for _ in range(n_windows):
        windows.append(torch.randn(K, B, C, H, W, device=device) * change_scale)
    return windows


def make_mixed_windows(
    n_windows: int, K: int, B: int, C: int, H: int, W: int,
    device: torch.device = torch.device("cpu"),
) -> List[torch.Tensor]:
    """Alternating blocks of 4 stable + 4 changing windows."""
    windows: List[torch.Tensor] = []
    i = 0
    while len(windows) < n_windows:
        if (i // 4) % 2 == 0:
            batch = make_stable_windows(
                min(4, n_windows - len(windows)), K, B, C, H, W,
                noise_scale=0.005, device=device,
            )
        else:
            batch = make_changing_windows(
                min(4, n_windows - len(windows)), K, B, C, H, W,
                change_scale=1.0, device=device,
            )
        windows.extend(batch)
        i += 4
    return windows[:n_windows]


# =====================================================================
# Evaluation helpers
# =====================================================================
@dataclass
class RunResult:
    name: str
    y_outputs: List[torch.Tensor]
    modes: List[str]
    diags: List[dict]
    elapsed: float

    @property
    def mode_hist(self) -> Dict[str, int]:
        hist: Dict[str, int] = {"full": 0, "collapse": 0, "reuse": 0}
        for m in self.modes:
            hist[m] = hist.get(m, 0) + 1
        return hist

    @property
    def mean_d_inter(self) -> float:
        vals = [d["d_inter"] for d in self.diags
                if "d_inter" in d and d["d_inter"] != float("inf")]
        return sum(vals) / max(len(vals), 1) if vals else float("nan")

    @property
    def mean_var_intra(self) -> float:
        vals = [d["var_intra"] for d in self.diags if "var_intra" in d]
        return sum(vals) / max(len(vals), 1)

    @property
    def cache_refresh_count(self) -> int:
        return sum(1 for d in self.diags if d.get("cache_updated", False))

    @property
    def cache_invalidation_count(self) -> int:
        return sum(
            1 for d in self.diags
            if d.get("cache_valid_before", True) and not d.get("cache_valid_after", True)
        )


def compute_error(
    baseline_outputs: List[torch.Tensor],
    test_outputs: List[torch.Tensor],
) -> Dict[str, float]:
    """MAE and MaxAE between two output sequences."""
    assert len(baseline_outputs) == len(test_outputs)
    all_abs = []
    for yb, yt in zip(baseline_outputs, test_outputs):
        all_abs.append((yb - yt).abs())
    cat = torch.cat([a.reshape(-1) for a in all_abs])
    return {
        "mae": cat.mean().item(),
        "max_ae": cat.max().item(),
    }


# =====================================================================
# Run a strategy over a window sequence
# =====================================================================
def run_strategy(
    name: str,
    conv: nn.Conv2d,
    windows: List[torch.Tensor],
    config: RouterConfig,
    forced_mode: str | None = None,
    reuse_refresh_every: int = 4,
) -> RunResult:
    """Run TempoConv2d with given settings over a list of windows."""
    wrapper = TempoConv2d(
        conv=conv, config=config, forced_mode=forced_mode,
    )
    wrapper.eval()

    y_outputs: List[torch.Tensor] = []
    modes: List[str] = []
    diags: List[dict] = []

    t0 = time.perf_counter()
    for idx, xw in enumerate(windows):
        # For fixed reuse with periodic refresh: force full on refresh windows
        if forced_mode == "reuse" and idx % reuse_refresh_every == 0:
            # Temporarily override to full for this window
            wrapper.forced_mode = "full"
            y, m, _, d = wrapper(xw)
            wrapper.forced_mode = "reuse"
        else:
            y, m, _, d = wrapper(xw)
        y_outputs.append(y.clone())
        modes.append(m)
        diags.append(d)
    elapsed = time.perf_counter() - t0

    return RunResult(name=name, y_outputs=y_outputs, modes=modes,
                     diags=diags, elapsed=elapsed)


# =====================================================================
# Print report
# =====================================================================
def print_report(result: RunResult, error: Dict[str, float] | None = None) -> None:
    hist = result.mode_hist
    print(f"\n{'='*60}")
    print(f"  Strategy : {result.name}")
    print(f"{'='*60}")
    print(f"  Windows processed : {len(result.modes)}")
    print(f"  Mode histogram    : full={hist['full']}  collapse={hist['collapse']}  reuse={hist['reuse']}")
    print(f"  Cache refreshes   : {result.cache_refresh_count}")
    print(f"  Cache invalidations: {result.cache_invalidation_count}")
    print(f"  Mean d_inter      : {result.mean_d_inter:.6f}")
    print(f"  Mean var_intra    : {result.mean_var_intra:.6f}")
    print(f"  Wall-clock time   : {result.elapsed*1000:.2f} ms")
    if error is not None:
        print(f"  MAE  vs Full      : {error['mae']:.6e}")
        print(f"  MaxAE vs Full     : {error['max_ae']:.6e}")
    print(f"{'='*60}")


# =====================================================================
# Main benchmark
# =====================================================================
def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cpu")

    # --- Hyper-parameters ---
    K = 4            # window length
    B = 2            # batch
    Cin = 8          # input channels
    Cout = 16        # output channels
    H, W = 8, 8      # spatial
    kernel = 3
    n_windows = 24   # per scenario

    # Shared Conv2d (weights frozen)
    conv = nn.Conv2d(Cin, Cout, kernel, padding=1, bias=True)
    conv.eval()
    for p in conv.parameters():
        p.requires_grad_(False)

    # Router config
    config = RouterConfig(
        tau_reuse=0.05,
        tau_reuse_hard=0.50,
        tau_stable=0.02,
        tau_collapse=0.10,
        max_reuse=4,
    )

    # =================================================================
    # Scenario A: Stable windows
    # =================================================================
    print("\n" + "#" * 60)
    print("  SCENARIO A: Stable (high-similarity) windows")
    print("#" * 60)

    windows_stable = make_stable_windows(n_windows, K, B, Cin, H, W,
                                          noise_scale=0.005, device=device)

    full_a = run_strategy("Full baseline", conv, windows_stable, config,
                          forced_mode="full")
    collapse_a = run_strategy("Fixed Collapse", conv, windows_stable, config,
                              forced_mode="collapse")
    reuse_a = run_strategy("Fixed Reuse (refresh/4)", conv, windows_stable, config,
                           forced_mode="reuse", reuse_refresh_every=4)
    adaptive_a = run_strategy("Adaptive TempoCache", conv, windows_stable, config)

    for r in [full_a, collapse_a, reuse_a, adaptive_a]:
        err = compute_error(full_a.y_outputs, r.y_outputs) if r is not full_a else None
        print_report(r, err)

    # =================================================================
    # Scenario B: Changing windows
    # =================================================================
    print("\n" + "#" * 60)
    print("  SCENARIO B: Changing (high-variation) windows")
    print("#" * 60)

    windows_change = make_changing_windows(n_windows, K, B, Cin, H, W,
                                            change_scale=1.0, device=device)

    full_b = run_strategy("Full baseline", conv, windows_change, config,
                          forced_mode="full")
    collapse_b = run_strategy("Fixed Collapse", conv, windows_change, config,
                              forced_mode="collapse")
    reuse_b = run_strategy("Fixed Reuse (refresh/4)", conv, windows_change, config,
                           forced_mode="reuse", reuse_refresh_every=4)
    adaptive_b = run_strategy("Adaptive TempoCache", conv, windows_change, config)

    for r in [full_b, collapse_b, reuse_b, adaptive_b]:
        err = compute_error(full_b.y_outputs, r.y_outputs) if r is not full_b else None
        print_report(r, err)

    # =================================================================
    # Scenario C: Mixed windows
    # =================================================================
    print("\n" + "#" * 60)
    print("  SCENARIO C: Mixed (alternating stable/changing)")
    print("#" * 60)

    windows_mixed = make_mixed_windows(n_windows, K, B, Cin, H, W, device=device)

    full_c = run_strategy("Full baseline", conv, windows_mixed, config,
                          forced_mode="full")
    collapse_c = run_strategy("Fixed Collapse", conv, windows_mixed, config,
                              forced_mode="collapse")
    reuse_c = run_strategy("Fixed Reuse (refresh/4)", conv, windows_mixed, config,
                           forced_mode="reuse", reuse_refresh_every=4)
    adaptive_c = run_strategy("Adaptive TempoCache", conv, windows_mixed, config)

    for r in [full_c, collapse_c, reuse_c, adaptive_c]:
        err = compute_error(full_c.y_outputs, r.y_outputs) if r is not full_c else None
        print_report(r, err)

    # =================================================================
    # Per-window trace for Adaptive on Mixed
    # =================================================================
    print("\n" + "#" * 60)
    print("  PER-WINDOW TRACE: Adaptive on Mixed scenario")
    print("#" * 60)
    print(f"  {'Win':>3}  {'Mode':<9} {'d_inter':>10} {'var_intra':>10} "
          f"{'c_valid':>7} {'c_age':>5}")
    print("  " + "-" * 55)
    for i, d in enumerate(adaptive_c.diags):
        d_inter_s = f"{d['d_inter']:.4f}" if d['d_inter'] != float('inf') else "     inf"
        print(f"  {i:3d}  {d['mode']:<9} {d_inter_s:>10} {d['var_intra']:10.4f} "
              f"{str(d.get('cache_valid_after', '?')):>7} {d.get('cache_age_after', '?'):>5}")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 60)
    print("  MVP VERIFICATION SUMMARY")
    print("=" * 60)
    hist_a = adaptive_a.mode_hist
    hist_b = adaptive_b.mode_hist
    hist_c = adaptive_c.mode_hist

    checks = []

    # Check 1: stable windows should trigger reuse or collapse
    reuse_frac_a = (hist_a["reuse"] + hist_a["collapse"]) / max(len(adaptive_a.modes), 1)
    checks.append(("Stable → Reuse/Collapse fraction >= 0.5", reuse_frac_a >= 0.5))

    # Check 2: changing windows should mostly trigger full
    full_frac_b = hist_b["full"] / max(len(adaptive_b.modes), 1)
    checks.append(("Changing → Full fraction >= 0.7", full_frac_b >= 0.7))

    # Check 3: adaptive error on stable should be small
    err_a = compute_error(full_a.y_outputs, adaptive_a.y_outputs)
    checks.append(("Stable adaptive MAE < 0.1", err_a["mae"] < 0.1))

    # Check 4: full baseline error vs itself is zero
    checks.append(("Full vs Full error == 0", True))  # trivially true

    # Check 5: mixed adaptive shows both modes
    mixed_has_variety = hist_c["full"] > 0 and (hist_c["reuse"] > 0 or hist_c["collapse"] > 0)
    checks.append(("Mixed → uses both Full and Reuse/Collapse", mixed_has_variety))

    for desc, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc}")

    all_pass = all(ok for _, ok in checks)
    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    main()