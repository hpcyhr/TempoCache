"""Attention signatures for Q/K/V-based operators."""

from __future__ import annotations

import torch

from ...utils.tensor_ops import segment_summary, temporal_variation
from .vector import VectorSignatureExtractor


class AttentionSignatureExtractor:
    """Extract Q/K/V summaries plus qk drift and token variation."""

    def __init__(self, feature_segments: int = 8, token_segments: int = 4) -> None:
        self.vector = VectorSignatureExtractor(feature_segments=feature_segments, token_segments=token_segments)
        self.token_segments = token_segments

    def _flat(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            # [K,B,H,N,D] -> [K,B,N,H*D]
            x = x.permute(0, 1, 3, 2, 4).reshape(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * x.shape[4])
        return x

    def extract(self, q_win: torch.Tensor, k_win: torch.Tensor, v_win: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qf = self._flat(q_win)
        kf = self._flat(k_win)
        vf = self._flat(v_win)

        sig_q, v_q = self.vector.extract(qf)
        sig_k, v_k = self.vector.extract(kf)
        sig_v, v_v = self.vector.extract(vf)

        q_mean = qf.mean(dim=0).reshape(qf.shape[1], -1)
        k_mean = kf.mean(dim=0).reshape(kf.shape[1], -1)
        drift = (q_mean - k_mean).abs().mean(dim=1, keepdim=True)

        token_stat = qf.mean(dim=0).abs().mean(dim=-1) if qf.ndim == 4 else qf.mean(dim=0).abs()
        token_sig = segment_summary(token_stat.reshape(token_stat.shape[0], -1), segments=self.token_segments)

        v_temp = (v_q + v_k + v_v) / 3.0
        signature = torch.cat([sig_q, sig_k, sig_v, token_sig, drift, v_temp.unsqueeze(1)], dim=1)
        return signature, v_temp

