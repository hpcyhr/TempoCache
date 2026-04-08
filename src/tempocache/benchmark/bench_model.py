"""Model-level benchmark for toy SNN models."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable

import torch

from ..config import AttentionConfig, BenchmarkConfig, RuntimeConfig
from ..integration.reports import collect_module_diagnostics
from ..models import ToyCNNSNN, ToyResSNN, ToySpikeTransformer
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


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _timeit(fn: Callable[[], torch.Tensor], cfg: BenchmarkConfig) -> float:
    with torch.no_grad():
        for _ in range(cfg.warmup):
            _ = fn()
            _sync(cfg.device)
        start = time.perf_counter()
        for _ in range(cfg.iters):
            _ = fn()
            _sync(cfg.device)
        return (time.perf_counter() - start) * 1000.0 / cfg.iters


def _build_model(model_type: str, runtime: RuntimeConfig) -> tuple[torch.nn.Module, tuple[int, ...]]:
    if model_type == "cnn":
        return ToyCNNSNN(use_tempo=True, runtime_config=runtime), (runtime.window_size * 4, 8, 3, 32, 32)
    if model_type == "res":
        return ToyResSNN(use_tempo=True, runtime_config=runtime), (runtime.window_size * 4, 8, 3, 32, 32)
    if model_type == "transformer":
        return (
            ToySpikeTransformer(
                input_dim=32,
                embed_dim=64,
                num_heads=4,
                use_tempo=True,
                runtime_config=runtime,
                attention_config=AttentionConfig(),
            ),
            (runtime.window_size * 4, 8, 32, 32),
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def _module_mode_ratio(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        full_count = float(row.get("full_count", 0))
        collapse_count = float(row.get("collapse_count", 0))
        reuse_count = float(row.get("reuse_count", 0))
        denom = max(1.0, full_count + collapse_count + reuse_count)
        out.append(
            {
                "module_path": row.get("module_path", ""),
                "operator_family": row.get("operator_family", "unknown"),
                "full_ratio": full_count / denom,
                "collapse_ratio": collapse_count / denom,
                "reuse_ratio": reuse_count / denom,
            }
        )
    return out


def _family_summary(rows: list[dict]) -> list[dict]:
    agg = defaultdict(lambda: {"full_count": 0.0, "collapse_count": 0.0, "reuse_count": 0.0})
    for row in rows:
        fam = row.get("operator_family", "unknown")
        agg[fam]["full_count"] += float(row.get("full_count", 0))
        agg[fam]["collapse_count"] += float(row.get("collapse_count", 0))
        agg[fam]["reuse_count"] += float(row.get("reuse_count", 0))
    result = []
    for fam, val in agg.items():
        total = max(1.0, val["full_count"] + val["collapse_count"] + val["reuse_count"])
        result.append(
            {
                "operator_family": fam,
                "full_ratio": val["full_count"] / total,
                "collapse_ratio": val["collapse_count"] / total,
                "reuse_ratio": val["reuse_count"] / total,
            }
        )
    return result


def benchmark_model(model_type: str, mode: str, bench_config: BenchmarkConfig | None = None) -> dict:
    cfg = bench_config or BenchmarkConfig()
    set_seed(cfg.seed)
    dtype = _dtype_from_name(cfg.dtype)
    device = cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu"

    runtime = RuntimeConfig(mode=mode, window_size=4, warmup_full_windows=1)
    model, shape = _build_model(model_type, runtime)
    model = model.to(device=device, dtype=dtype)

    x = torch.randn(*shape, device=device, dtype=dtype)
    latency_ms = _timeit(lambda: model(x), cfg)
    throughput = (shape[0] * shape[1]) / max(latency_ms / 1000.0, 1e-9)

    rows = collect_module_diagnostics(model)
    return {
        "model_type": model_type,
        "mode": mode,
        "device": device,
        "dtype": cfg.dtype,
        "e2e_latency_ms": latency_ms,
        "throughput_samples_per_s": throughput,
        "module_mode_ratio": _module_mode_ratio(rows),
        "per_family_summary": _family_summary(rows),
    }

