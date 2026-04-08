"""Operator-family benchmark."""

from __future__ import annotations

import copy
import time
from typing import Callable

import torch
import torch.nn as nn

from ..config import AttentionConfig, BenchmarkConfig, RuntimeConfig
from ..core.diagnostics import ModuleDiagnostics
from ..ops import (
    TempoAttentionCore,
    TempoBMM,
    TempoConv1d,
    TempoConv2d,
    TempoConv3d,
    TempoLinear,
    TempoMatMul,
    TempoSpikeSelfAttention,
)
from ..utils.seed import set_seed


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _sync_if_needed(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _timeit(fn: Callable[[], torch.Tensor], cfg: BenchmarkConfig) -> float:
    with torch.no_grad():
        for _ in range(cfg.warmup):
            _ = fn()
            _sync_if_needed(cfg.device)
        start = time.perf_counter()
        for _ in range(cfg.iters):
            _ = fn()
            _sync_if_needed(cfg.device)
        elapsed = (time.perf_counter() - start) * 1000.0 / cfg.iters
    return elapsed


def _make_pattern(x: torch.Tensor, pattern: str) -> torch.Tensor:
    t = x.shape[0]
    if pattern == "constant":
        base = x[0:1].clone()
        return base.expand(t, *base.shape[1:])
    if pattern == "drift":
        out = x.clone()
        ramp = torch.linspace(0, 0.25, steps=t, device=x.device, dtype=x.dtype).reshape(t, *([1] * (x.ndim - 1)))
        return out + ramp
    if pattern == "abrupt":
        out = x.clone()
        out[t // 2 :] += 0.75
        return out
    raise ValueError(f"Unknown pattern: {pattern}")


def _collect_counts(diag: dict) -> dict:
    keys = ["full_count", "collapse_count", "reuse_count", "cache_hit_count", "total_samples"]
    agg = {k: 0 for k in keys}

    def rec(obj: object) -> None:
        if isinstance(obj, dict):
            for k in keys:
                if k in obj and isinstance(obj[k], (int, float)):
                    agg[k] += obj[k]
            for v in obj.values():
                rec(v)
        elif isinstance(obj, list):
            for item in obj:
                rec(item)

    rec(diag)
    total_modes = agg["full_count"] + agg["collapse_count"] + agg["reuse_count"]
    agg["cache_hit_ratio"] = float(agg["reuse_count"]) / max(1, total_modes)
    return agg


def _build_operator(
    family: str,
    runtime: RuntimeConfig,
    device: str,
    dtype: torch.dtype,
) -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    t, b = 16, 8
    if family == "conv1d":
        conv = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        module = TempoConv1d.from_conv(conv, runtime, module_name="bench.conv1d").to(device=device, dtype=dtype)
        x = torch.randn(t, b, 16, 64, device=device, dtype=dtype)
        return module, (x,)
    if family == "conv2d":
        conv = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        module = TempoConv2d.from_conv(conv, runtime, module_name="bench.conv2d").to(device=device, dtype=dtype)
        x = torch.randn(t, b, 16, 32, 32, device=device, dtype=dtype)
        return module, (x,)
    if family == "conv3d":
        conv = nn.Conv3d(8, 12, kernel_size=3, padding=1)
        module = TempoConv3d.from_conv(conv, runtime, module_name="bench.conv3d").to(device=device, dtype=dtype)
        x = torch.randn(t, b, 8, 8, 16, 16, device=device, dtype=dtype)
        return module, (x,)
    if family == "linear":
        lin = nn.Linear(128, 256)
        module = TempoLinear.from_linear(lin, runtime, module_name="bench.linear").to(device=device, dtype=dtype)
        x = torch.randn(t, b, 32, 128, device=device, dtype=dtype)
        return module, (x,)
    if family == "matmul":
        module = TempoMatMul(runtime_config=runtime, module_name="bench.matmul").to(device=device, dtype=dtype)
        a = torch.randn(t, b, 32, 64, device=device, dtype=dtype)
        bb = torch.randn(t, b, 64, 48, device=device, dtype=dtype)
        return module, (a, bb)
    if family == "bmm":
        module = TempoBMM(runtime_config=runtime, module_name="bench.bmm").to(device=device, dtype=dtype)
        a = torch.randn(t, b, 32, 64, device=device, dtype=dtype)
        bb = torch.randn(t, b, 64, 48, device=device, dtype=dtype)
        return module, (a, bb)
    if family == "attention":
        module = TempoSpikeSelfAttention(
            embed_dim=128,
            num_heads=8,
            runtime_config=runtime,
            attention_config=AttentionConfig(),
            module_name="bench.attn",
        ).to(device=device, dtype=dtype)
        x = torch.randn(t, b, 32, 128, device=device, dtype=dtype)
        return module, (x,)
    raise ValueError(f"Unsupported family: {family}")


def benchmark_operator_family(
    family: str,
    mode: str,
    bench_config: BenchmarkConfig | None = None,
) -> dict:
    cfg = bench_config or BenchmarkConfig()
    set_seed(cfg.seed)
    dtype = _dtype_from_name(cfg.dtype)
    device = cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu"

    patterns = ["constant", "drift", "abrupt"]
    records = []
    for pattern in patterns:
        full_runtime = RuntimeConfig(mode="full", window_size=4)
        target_runtime = RuntimeConfig(mode=mode, window_size=4)

        full_op, full_inputs = _build_operator(family, full_runtime, device=device, dtype=dtype)
        test_op, test_inputs = _build_operator(family, target_runtime, device=device, dtype=dtype)

        full_inputs = tuple(_make_pattern(x, pattern) for x in full_inputs)
        test_inputs = tuple(_make_pattern(x, pattern) for x in test_inputs)

        full_ms = _timeit(lambda: full_op(*full_inputs), cfg)
        test_ms = _timeit(lambda: test_op(*test_inputs), cfg)
        diag = test_op.get_diagnostics() if hasattr(test_op, "get_diagnostics") else {}
        counts = _collect_counts(diag)
        records.append(
            {
                "family": family,
                "mode": mode,
                "pattern": pattern,
                "latency_ms_full": full_ms,
                "latency_ms_mode": test_ms,
                "speedup": full_ms / max(test_ms, 1e-9),
                "full_count": counts["full_count"],
                "collapse_count": counts["collapse_count"],
                "reuse_count": counts["reuse_count"],
                "cache_hit_ratio": counts["cache_hit_ratio"],
            }
        )

    return {
        "family": family,
        "mode": mode,
        "device": device,
        "dtype": cfg.dtype,
        "records": records,
    }

