"""Attention-aware Tempo operators."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from ..config import AttentionConfig, RuntimeConfig
from ..core.signatures.attention import AttentionSignatureExtractor
from ..registry import ATTENTION_VARIANT_REGISTRY
from .tempo_linear import TempoLinear
from .tempo_matmul import TempoMatMul


class TempoAttentionCore(nn.Module):
    """
    Attention core wrapper for qk score generation and attn@v product.
    """

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        attention_config: AttentionConfig | None = None,
        module_name: str = "tempo_attention_core",
    ) -> None:
        super().__init__()
        self.runtime_config = runtime_config
        self.attention_config = attention_config or AttentionConfig()
        self.module_name = module_name
        self.signature_extractor = AttentionSignatureExtractor()

        self.qk_op = TempoMatMul(runtime_config=runtime_config, module_name=f"{module_name}.qk")
        self.av_op = TempoMatMul(runtime_config=runtime_config, module_name=f"{module_name}.av")

    def collapse_window(self, q_win: torch.Tensor, k_win: torch.Tensor, v_win: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return q_win.mean(dim=0), k_win.mean(dim=0), v_win.mean(dim=0)

    def forward_single_step(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k_t = k.transpose(-1, -2)
        scores = torch.matmul(q, k_t)
        if self.attention_config.scale_qk:
            scores = scores / math.sqrt(q.shape[-1])
        attn = self._apply_attention_form(scores)
        context = torch.matmul(attn, v)
        return context, scores

    def _apply_attention_form(self, scores: torch.Tensor) -> torch.Tensor:
        if self.attention_config.spike_form:
            act = torch.relu(scores)
            return act / (act.sum(dim=-1, keepdim=True) + 1e-6)
        return torch.softmax(scores, dim=-1)

    def forward_window(self, q_win: torch.Tensor, k_win: torch.Tensor, v_win: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_t = k_win.transpose(-1, -2)
        if self.attention_config.cache_qk_score:
            scores = self.qk_op(q_win, k_t)
        else:
            scores = torch.matmul(q_win, k_t)
        if self.attention_config.scale_qk:
            scores = scores / math.sqrt(q_win.shape[-1])

        attn = self._apply_attention_form(scores)
        if self.attention_config.cache_av_product:
            context = self.av_op(attn, v_win)
        else:
            context = torch.matmul(attn, v_win)
        return context, scores, attn

    def update_cache(self, *args: Any, **kwargs: Any) -> None:
        # Cache updates are delegated to sub-operators (TempoMatMul).
        return None

    def get_diagnostics(self) -> dict:
        return {
            "module_name": self.module_name,
            "operator_family": "attention",
            "qk": self.qk_op.get_diagnostics(),
            "av": self.av_op.get_diagnostics(),
        }

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if q.ndim < 5:
            raise ValueError(f"TempoAttentionCore expects [T,B,H,N,D], got {tuple(q.shape)}")
        return self.forward_window(q, k, v)


class TempoSpikeSelfAttention(nn.Module):
    """
    High-level attention wrapper:
    - q/k/v/out projections
    - qk^T
    - attn@v
    - optional spike-form attention
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        runtime_config: RuntimeConfig,
        attention_config: AttentionConfig | None = None,
        module_name: str = "tempo_spike_self_attention",
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.runtime_config = runtime_config
        self.attention_config = attention_config or AttentionConfig()
        self.module_name = module_name
        self.variant_name: str | None = None

        self.q_proj = self._build_proj(name="q_proj", enable=self.attention_config.cache_q_proj)
        self.k_proj = self._build_proj(name="k_proj", enable=self.attention_config.cache_k_proj)
        self.v_proj = self._build_proj(name="v_proj", enable=self.attention_config.cache_v_proj)
        self.out_proj = self._build_proj(name="out_proj", enable=self.attention_config.cache_out_proj)
        self.attn_core = TempoAttentionCore(
            runtime_config=runtime_config,
            attention_config=self.attention_config,
            module_name=f"{module_name}.core",
        )

    def _build_proj(self, name: str, enable: bool) -> nn.Module:
        linear = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        if enable:
            return TempoLinear.from_linear(
                linear=linear,
                runtime_config=self.runtime_config,
                module_name=f"{self.module_name}.{name}",
            )
        return linear

    @classmethod
    def from_existing(
        cls,
        module: nn.Module,
        runtime_config: RuntimeConfig,
        attention_config: AttentionConfig | None = None,
        module_name: str = "tempo_spike_self_attention",
    ) -> "TempoSpikeSelfAttention":
        embed_dim = int(getattr(module, "embed_dim"))
        num_heads = int(getattr(module, "num_heads"))
        wrapped = cls(
            embed_dim=embed_dim,
            num_heads=num_heads,
            runtime_config=runtime_config,
            attention_config=attention_config,
            module_name=module_name,
        )
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            if not hasattr(module, name):
                continue
            src = getattr(module, name)
            dst = getattr(wrapped, name)
            if isinstance(dst, TempoLinear):
                dst.linear.load_state_dict(src.state_dict())
            elif isinstance(dst, nn.Linear):
                dst.load_state_dict(src.state_dict())
        return wrapped

    def set_variant(self, name: str) -> None:
        self.variant_name = name

    def _apply_variant(self, scores: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.variant_name is None:
            return scores
        if self.variant_name not in ATTENTION_VARIANT_REGISTRY:
            raise KeyError(f"Unknown attention variant: {self.variant_name}")
        return ATTENTION_VARIANT_REGISTRY[self.variant_name](scores=scores, q=q, k=k, v=v)

    def _project(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return layer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"TempoSpikeSelfAttention expects [T,B,N,D], got {tuple(x.shape)}")
        t, b, n, d = x.shape
        if d != self.embed_dim:
            raise ValueError(f"Input feature dim {d} != embed_dim {self.embed_dim}")

        q = self._project(self.q_proj, x)
        k = self._project(self.k_proj, x)
        v = self._project(self.v_proj, x)

        qh = q.reshape(t, b, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        kh = k.reshape(t, b, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        vh = v.reshape(t, b, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        context, scores, _ = self.attn_core(qh, kh, vh)
        scores = self._apply_variant(scores, qh, kh, vh)
        if self.attention_config.cache_av_product:
            # Keep behavior deterministic with the variant-adjusted scores.
            if self.attention_config.spike_form:
                attn = torch.relu(scores)
                attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
            else:
                attn = torch.softmax(scores, dim=-1)
            context = self.attn_core.av_op(attn, vh)

        out = context.permute(0, 1, 3, 2, 4).reshape(t, b, n, self.embed_dim)
        out = self._project(self.out_proj, out)
        return out

    def get_diagnostics(self) -> dict:
        payload = {
            "module_name": self.module_name,
            "operator_family": "attention",
            "core": self.attn_core.get_diagnostics(),
        }
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            layer = getattr(self, name)
            if hasattr(layer, "get_diagnostics"):
                payload[name] = layer.get_diagnostics()
        return payload

