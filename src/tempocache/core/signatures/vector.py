"""Vector signature extractor for Linear-like operators."""

from __future__ import annotations

import torch

from ...utils.tensor_ops import segment_summary, temporal_variation


class VectorSignatureExtractor:
    """Extract vector and grouped feature signatures from [K,B,D] or [K,B,N,D]."""

    def __init__(self, feature_segments: int = 8, token_segments: int = 4) -> None:
        self.feature_segments = feature_segments
        self.token_segments = token_segments

    def extract(self, x_win: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x_win.ndim not in (3, 4):
            raise ValueError(f"Vector extractor expects [K,B,D] or [K,B,N,D], got {tuple(x_win.shape)}")
        x_mean = x_win.mean(dim=0)
        b = x_mean.shape[0]

        feat_flat = x_mean.abs().reshape(b, -1)
        feat_sig = segment_summary(feat_flat, segments=self.feature_segments)

        if x_mean.ndim == 4:
            token_stat = x_mean.abs().mean(dim=-1)  # [B,N]
            token_sig = segment_summary(token_stat, segments=self.token_segments)
        else:
            token_sig = torch.zeros((b, self.token_segments), device=x_mean.device, dtype=x_mean.dtype)

        v_temp = temporal_variation(x_win)
        signature = torch.cat([feat_sig, token_sig, v_temp.unsqueeze(1)], dim=1)
        return signature, v_temp

