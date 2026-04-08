"""
tempocache/runtime/signature_extractor.py

Lightweight signature extraction from the target block's input tensor
x_window [K, B, C, H, W].

Signature vector:  S = concat(s_ch, s_grid, [v_temp])
Inter-window distance: d_inter = ||S_cur - S_prev||_1 / (||S_prev||_1 + eps)

Semantics preserved from the validated MVP.
"""

from __future__ import annotations

import torch
from torch import Tensor


def extract_channel_summary(x_window: Tensor) -> Tensor:
    """Per-channel scalar summary.  s_ch[c] = sum_{k,b,h,w} x[k,b,c,h,w]."""
    return x_window.sum(dim=(0, 1, 3, 4))  # [C]


def extract_grid_summary(x_window: Tensor, grid_size: int = 2) -> Tensor:
    """Partition H×W into G×G coarse grid and sum each cell → [G*G]."""
    K, B, C, H, W = x_window.shape
    G = grid_size
    h_edges = torch.linspace(0, H, G + 1, dtype=torch.long)
    w_edges = torch.linspace(0, W, G + 1, dtype=torch.long)

    cells = []
    for gi in range(G):
        for gj in range(G):
            h0, h1 = int(h_edges[gi].item()), int(h_edges[gi + 1].item())
            w0, w1 = int(w_edges[gj].item()), int(w_edges[gj + 1].item())
            h1 = max(h1, h0 + 1)
            w1 = max(w1, w0 + 1)
            cells.append(x_window[:, :, :, h0:h1, w0:w1].sum())
    return torch.stack(cells)  # [G*G]


def compute_temporal_variation(x_window: Tensor) -> Tensor:
    """Average L1 difference between adjacent timesteps → scalar."""
    K = x_window.shape[0]
    if K <= 1:
        return torch.tensor(0.0, device=x_window.device, dtype=x_window.dtype)
    diffs = (x_window[1:] - x_window[:-1]).abs().mean(dim=(1, 2, 3, 4))  # [K-1]
    return diffs.mean()


def build_signature(x_window: Tensor, grid_size: int = 2) -> Tensor:
    """S = concat(s_ch, s_grid, [v_temp])  → [C + G*G + 1]."""
    s_ch = extract_channel_summary(x_window)
    s_grid = extract_grid_summary(x_window, grid_size)
    v_temp = compute_temporal_variation(x_window)
    return torch.cat([s_ch, s_grid, v_temp.unsqueeze(0)])


def compute_inter_window_distance(
    sig_cur: Tensor, sig_prev: Tensor, eps: float = 1e-6
) -> Tensor:
    """Normalised L1 distance → scalar."""
    return (sig_cur - sig_prev).abs().sum() / (sig_prev.abs().sum() + eps)