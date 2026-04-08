"""Tensor utilities used across executors/signatures."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch


def ensure_time_batch(x: torch.Tensor) -> None:
    if x.ndim < 2:
        raise ValueError(f"Expected at least 2 dims [T, B, ...], got shape={tuple(x.shape)}")


def segment_summary(x: torch.Tensor, segments: int = 8) -> torch.Tensor:
    """
    Reduce a [B, F] tensor into [B, segments] by chunked mean.
    """
    if x.ndim != 2:
        raise ValueError(f"segment_summary expects [B,F], got {x.shape}")
    bsz, feat = x.shape
    if feat == 0:
        return torch.zeros((bsz, segments), device=x.device, dtype=x.dtype)
    seg_size = max(1, feat // segments)
    summaries = []
    for idx in range(segments):
        start = idx * seg_size
        end = feat if idx == segments - 1 else min(feat, (idx + 1) * seg_size)
        if start >= feat:
            summaries.append(torch.zeros((bsz,), device=x.device, dtype=x.dtype))
        else:
            summaries.append(x[:, start:end].mean(dim=1))
    return torch.stack(summaries, dim=1)


def temporal_variation(x_win: torch.Tensor) -> torch.Tensor:
    """
    Return per-sample temporal variation score for window input [K,B,...].
    """
    ensure_time_batch(x_win)
    if x_win.shape[0] <= 1:
        return torch.zeros((x_win.shape[1],), device=x_win.device, dtype=x_win.dtype)
    delta = (x_win[1:] - x_win[:-1]).abs()
    reduce_dims = tuple(range(2, delta.ndim))
    return delta.mean(dim=reduce_dims).mean(dim=0)


def collapse_window(x_win: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Collapse temporal axis K -> 1 sample per batch."""
    if reduction == "mean":
        return x_win.mean(dim=0)
    if reduction == "max":
        return x_win.max(dim=0).values
    if reduction == "first":
        return x_win[0]
    raise ValueError(f"Unsupported collapse reduction: {reduction}")


def broadcast_time(x: torch.Tensor, k: int) -> torch.Tensor:
    """Broadcast [B,...] -> [K,B,...]."""
    return x.unsqueeze(0).expand(k, *x.shape)


def merge_time_batch(x_win: torch.Tensor) -> torch.Tensor:
    """[K,B,...] -> [K*B,...]."""
    k, b = x_win.shape[:2]
    return x_win.reshape(k * b, *x_win.shape[2:])


def split_time_batch(x: torch.Tensor, k: int, b: int) -> torch.Tensor:
    """[K*B,...] -> [K,B,...]."""
    return x.reshape(k, b, *x.shape[1:])


def as_long_index(indices: Sequence[int] | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(indices, torch.Tensor):
        return indices.to(device=device, dtype=torch.long)
    return torch.tensor(list(indices), device=device, dtype=torch.long)


def safe_stack(tensors: Iterable[torch.Tensor], dim: int = 0) -> torch.Tensor:
    tensors = list(tensors)
    if not tensors:
        raise ValueError("Cannot stack an empty iterable")
    return torch.stack(tensors, dim=dim)

