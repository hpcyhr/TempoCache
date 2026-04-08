"""Pairwise signatures for MatMul/BMM families."""

from __future__ import annotations

import torch

from ...utils.tensor_ops import segment_summary, temporal_variation
from .vector import VectorSignatureExtractor


class PairwiseTensorSignatureExtractor:
    """Extract paired signatures for A/B inputs and their interaction summary."""

    def __init__(self, feature_segments: int = 8) -> None:
        self.feature_segments = feature_segments
        self.vector_extractor = VectorSignatureExtractor(feature_segments=feature_segments, token_segments=4)

    def extract(self, a_win: torch.Tensor, b_win: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if a_win.shape[0] != b_win.shape[0] or a_win.shape[1] != b_win.shape[1]:
            raise ValueError(f"Pairwise extractor expects same [K,B,...], got {a_win.shape}, {b_win.shape}")
        sig_a, v_a = self.vector_extractor.extract(a_win.reshape(a_win.shape[0], a_win.shape[1], -1))
        sig_b, v_b = self.vector_extractor.extract(b_win.reshape(b_win.shape[0], b_win.shape[1], -1))

        a_mean = a_win.mean(dim=0).reshape(a_win.shape[1], -1).abs()
        b_mean = b_win.mean(dim=0).reshape(b_win.shape[1], -1).abs()
        pair_scalar = (a_mean.mean(dim=1, keepdim=True) * b_mean.mean(dim=1, keepdim=True))
        pair_sig = segment_summary(pair_scalar, segments=1)

        pair_v_temp = 0.5 * (v_a + v_b)
        signature = torch.cat([sig_a, sig_b, pair_sig, pair_v_temp.unsqueeze(1)], dim=1)
        return signature, pair_v_temp

