"""Distance metrics for signature routing."""

from __future__ import annotations

import torch


def inter_signature_distance(sig_cur: torch.Tensor, sig_prev: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    d_inter = ||sig_cur - sig_prev||_1 / (||sig_prev||_1 + eps), computed per sample.
    Both signatures are expected in shape [B, S].
    """
    if sig_cur.shape != sig_prev.shape:
        raise ValueError(f"Signature shape mismatch: {sig_cur.shape} vs {sig_prev.shape}")
    numer = (sig_cur - sig_prev).abs().sum(dim=1)
    denom = sig_prev.abs().sum(dim=1) + eps
    return numer / denom

