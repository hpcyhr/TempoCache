"""Toy spiking transformer-like model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import AttentionConfig, RuntimeConfig
from ..ops.tempo_attention import TempoSpikeSelfAttention
from ..ops.tempo_linear import TempoLinear
from .neurons import IFNode, LIFNode


class PlainSpikeSelfAttention(nn.Module):
    """Baseline spike-friendly attention without TempoCache."""

    def __init__(self, embed_dim: int, num_heads: int, spike_form: bool = True) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.spike_form = spike_form
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, b, n, d = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        qh = q.reshape(t, b, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        kh = k.reshape(t, b, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        vh = v.reshape(t, b, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        scores = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if self.spike_form:
            attn = torch.relu(scores)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, vh)
        out = ctx.permute(0, 1, 3, 2, 4).reshape(t, b, n, d)
        return self.out_proj(out)


class ToySpikeTransformer(nn.Module):
    """
    Toy spiking transformer block:
    token projection -> self attention -> stateful neuron -> MLP.
    """

    def __init__(
        self,
        input_dim: int = 32,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_classes: int = 10,
        use_tempo: bool = False,
        runtime_config: RuntimeConfig | None = None,
        attention_config: AttentionConfig | None = None,
    ) -> None:
        super().__init__()
        runtime_config = runtime_config or RuntimeConfig(mode="full")
        attention_config = attention_config or AttentionConfig()

        tok = nn.Linear(input_dim, embed_dim)
        fc1 = nn.Linear(embed_dim, embed_dim * 2)
        fc2 = nn.Linear(embed_dim * 2, embed_dim)
        head = nn.Linear(embed_dim, num_classes)

        self.token_proj: nn.Module = TempoLinear.from_linear(tok, runtime_config, module_name="toy_transformer.token_proj") if use_tempo else tok
        self.attn: nn.Module = (
            TempoSpikeSelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                runtime_config=runtime_config,
                attention_config=attention_config,
                module_name="toy_transformer.attn",
            )
            if use_tempo
            else PlainSpikeSelfAttention(embed_dim=embed_dim, num_heads=num_heads, spike_form=attention_config.spike_form)
        )
        self.fc1: nn.Module = TempoLinear.from_linear(fc1, runtime_config, module_name="toy_transformer.fc1") if use_tempo else fc1
        self.fc2: nn.Module = TempoLinear.from_linear(fc2, runtime_config, module_name="toy_transformer.fc2") if use_tempo else fc2
        self.head: nn.Module = TempoLinear.from_linear(head, runtime_config, module_name="toy_transformer.head") if use_tempo else head

        self.neuron1 = LIFNode()
        self.neuron2 = IFNode()
        self.act = nn.GELU()

    def reset_state(self) -> None:
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "reset_state"):
                module.reset_state()
            if hasattr(module, "reset_cache"):
                module.reset_cache()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"ToySpikeTransformer expects [T,B,N,D], got {tuple(x.shape)}")
        x = self.token_proj(x)
        attn_out = self.attn(x)
        x = self.neuron1(attn_out + x)
        mlp = self.fc2(self.act(self.fc1(x)))
        x = self.neuron2(x + mlp)
        cls = x.mean(dim=2)
        return self.head(cls)
