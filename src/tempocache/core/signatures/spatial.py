"""Spatial tensor signatures for Conv1d/2d/3d families."""

from __future__ import annotations

import torch

from ...utils.tensor_ops import segment_summary, temporal_variation


class SpatialTensorSignatureExtractor:
    """Extract channel/grid/temporal signatures from [K,B,C,*spatial]."""

    def __init__(self, channel_segments: int = 8, spatial_segments: int = 8) -> None:
        self.channel_segments = channel_segments
        self.spatial_segments = spatial_segments

    def extract(self, x_win: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x_win.ndim < 4:
            raise ValueError(f"Spatial extractor expects [K,B,C,...], got {tuple(x_win.shape)}")
        k, b, c = x_win.shape[:3]
        x_mean = x_win.mean(dim=0)  # [B,C,*]

        spatial_dims = tuple(range(2, x_mean.ndim))
        channel_stat = x_mean.abs().mean(dim=spatial_dims)  # [B,C]
        ch_sig = segment_summary(channel_stat, segments=self.channel_segments)

        flat = x_mean.abs().reshape(b, -1)
        sp_sig = segment_summary(flat, segments=self.spatial_segments)

        v_temp = temporal_variation(x_win)
        signature = torch.cat([ch_sig, sp_sig, v_temp.unsqueeze(1)], dim=1)
        return signature, v_temp

