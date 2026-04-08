"""Attention-specific benchmark."""

from __future__ import annotations

import time
from dataclasses import replace

import torch

from ..config import AttentionConfig, BenchmarkConfig, RuntimeConfig
from ..ops.tempo_attention import TempoSpikeSelfAttention
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


def _timeit(fn, cfg: BenchmarkConfig) -> float:
    with torch.no_grad():
        for _ in range(cfg.warmup):
            _ = fn()
            _sync(cfg.device)
        start = time.perf_counter()
        for _ in range(cfg.iters):
            _ = fn()
            _sync(cfg.device)
        return (time.perf_counter() - start) * 1000.0 / cfg.iters


def _scenario_configs() -> dict[str, AttentionConfig]:
    base = AttentionConfig()
    return {
        "qkv_only_cached": replace(
            base,
            cache_q_proj=True,
            cache_k_proj=True,
            cache_v_proj=True,
            cache_qk_score=False,
            cache_av_product=False,
            cache_out_proj=False,
        ),
        "qk_only_cached": replace(
            base,
            cache_q_proj=False,
            cache_k_proj=False,
            cache_v_proj=False,
            cache_qk_score=True,
            cache_av_product=False,
            cache_out_proj=False,
        ),
        "av_only_cached": replace(
            base,
            cache_q_proj=False,
            cache_k_proj=False,
            cache_v_proj=False,
            cache_qk_score=False,
            cache_av_product=True,
            cache_out_proj=False,
        ),
        "all_substages_cached": replace(
            base,
            cache_q_proj=True,
            cache_k_proj=True,
            cache_v_proj=True,
            cache_qk_score=True,
            cache_av_product=True,
            cache_out_proj=True,
        ),
    }


def benchmark_attention_configs(mode: str, bench_config: BenchmarkConfig | None = None) -> dict:
    cfg = bench_config or BenchmarkConfig()
    set_seed(cfg.seed)
    dtype = _dtype_from_name(cfg.dtype)
    device = cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu"

    x = torch.randn(cfg.T, cfg.batch_size, 32, 64, device=device, dtype=dtype)
    results = []
    for name, attn_cfg in _scenario_configs().items():
        runtime_full = RuntimeConfig(mode="full", window_size=4)
        runtime_mode = RuntimeConfig(mode=mode, window_size=4)
        attn_full = TempoSpikeSelfAttention(
            embed_dim=64,
            num_heads=4,
            runtime_config=runtime_full,
            attention_config=attn_cfg,
            module_name=f"bench_attn.{name}.full",
        ).to(device=device, dtype=dtype)
        attn_mode = TempoSpikeSelfAttention(
            embed_dim=64,
            num_heads=4,
            runtime_config=runtime_mode,
            attention_config=attn_cfg,
            module_name=f"bench_attn.{name}.mode",
        ).to(device=device, dtype=dtype)
        attn_mode.load_state_dict(attn_full.state_dict())

        with torch.no_grad():
            y_ref = attn_full(x)
            y_mode = attn_mode(x)

        full_ms = _timeit(lambda: attn_full(x), cfg)
        mode_ms = _timeit(lambda: attn_mode(x), cfg)
        mse = float(torch.mean((y_ref - y_mode) ** 2).item())
        rel = float(torch.norm(y_ref - y_mode).item() / (torch.norm(y_ref).item() + 1e-8))

        results.append(
            {
                "scenario": name,
                "latency_ms_full": full_ms,
                "latency_ms_mode": mode_ms,
                "speedup": full_ms / max(mode_ms, 1e-9),
                "mse_vs_full": mse,
                "relative_error": rel,
            }
        )

    return {
        "mode": mode,
        "device": device,
        "dtype": cfg.dtype,
        "results": results,
    }

