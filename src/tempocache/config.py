"""Configuration dataclasses for TempoCache."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .enums import RuntimeMode


@dataclass
class ThresholdConfig:
    """Router thresholds for a module/family/global scope."""

    tau_reuse: float = 0.08
    tau_reuse_hard: float = 0.35
    tau_stable: float = 0.03
    tau_collapse: float = 0.08
    max_reuse: int = 4
    warmup_full_windows: int = 1

    def merged(self, override: "ThresholdConfig | None") -> "ThresholdConfig":
        if override is None:
            return ThresholdConfig(**self.__dict__)
        base = self.__dict__.copy()
        for key, value in override.__dict__.items():
            base[key] = value
        return ThresholdConfig(**base)


@dataclass
class RuntimeConfig:
    """Top-level runtime settings for all Tempo operators."""

    mode: RuntimeMode | str = RuntimeMode.ADAPTIVE
    window_size: int = 4
    warmup_full_windows: int = 1
    global_thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    per_family_thresholds: Dict[str, ThresholdConfig] = field(default_factory=dict)
    per_module_overrides: Dict[str, ThresholdConfig] = field(default_factory=dict)
    collapse_reduction: str = "mean"
    enable_diag: bool = True
    export_json: bool = True
    export_csv: bool = True
    keep_window_history: bool = True
    history_max_windows: int = 256

    def resolved_mode(self) -> RuntimeMode:
        return RuntimeMode.from_value(self.mode)

    def thresholds_for(self, family: str, module_name: str) -> ThresholdConfig:
        global_t = self.global_thresholds
        family_t = self.per_family_thresholds.get(family)
        module_t = self.per_module_overrides.get(module_name)
        merged = global_t.merged(family_t)
        merged = merged.merged(module_t)
        merged.warmup_full_windows = self.warmup_full_windows
        return merged


@dataclass
class PatchConfig:
    """Model patching options."""

    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    replace_conv1d: bool = True
    replace_conv2d: bool = True
    replace_conv3d: bool = True
    replace_linear: bool = True
    replace_matmul: bool = True
    replace_bmm: bool = True
    replace_attention: bool = True
    use_fx_rewriter: bool = False
    verbose: bool = False


@dataclass
class AttentionConfig:
    """Attention-specific cache controls."""

    cache_q_proj: bool = True
    cache_k_proj: bool = True
    cache_v_proj: bool = True
    cache_qk_score: bool = True
    cache_av_product: bool = True
    cache_out_proj: bool = True
    token_group_size: int = 4
    attention_signature_mode: str = "qkv_drift"
    spike_form: bool = True
    scale_qk: bool = True


@dataclass
class BenchmarkConfig:
    """Benchmark runtime options."""

    device: str = "cpu"
    dtype: str = "float32"
    T: int = 16
    batch_size: int = 8
    warmup: int = 5
    iters: int = 20
    seed: int = 7
    export_dir: str = "benchmark_outputs"

