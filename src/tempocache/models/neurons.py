"""Stateful temporal neuron fallbacks with optional SpikingJelly compatibility."""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn

try:
    from spikingjelly.activation_based import neuron as sj_neuron  # type: ignore

    _HAS_SPIKINGJELLY = True
except Exception:
    sj_neuron = None
    _HAS_SPIKINGJELLY = False


class _StatefulNeuronBase(nn.Module):
    is_stateful_temporal = True

    def __init__(self) -> None:
        super().__init__()
        self.v: torch.Tensor | None = None

    def reset_state(self) -> None:
        self.v = None

    def _init_state(self, x_step: torch.Tensor) -> None:
        if self.v is None or self.v.shape != x_step.shape or self.v.device != x_step.device:
            self.v = torch.zeros_like(x_step)

    def step(self, x_step: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"Neuron expects [T,B,...], got {tuple(x.shape)}")
        outs = []
        for t in range(x.shape[0]):
            outs.append(self.step(x[t]))
        return torch.stack(outs, dim=0)


class IFNode(_StatefulNeuronBase):
    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__()
        self.threshold = threshold
        self._sj = sj_neuron.IFNode(v_threshold=threshold, step_mode="s") if _HAS_SPIKINGJELLY else None

    def reset_state(self) -> None:
        super().reset_state()
        if self._sj is not None:
            self._sj.reset()

    def step(self, x_step: torch.Tensor) -> torch.Tensor:
        if self._sj is not None:
            return self._sj(x_step)
        self._init_state(x_step)
        assert self.v is not None
        self.v = self.v + x_step
        spike = (self.v >= self.threshold).to(x_step.dtype)
        self.v = self.v * (1.0 - spike)
        return spike


class LIFNode(_StatefulNeuronBase):
    def __init__(self, threshold: float = 1.0, decay: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self._sj = sj_neuron.LIFNode(v_threshold=threshold, tau=max(1e-4, 1.0 / (1.0 - decay)), step_mode="s") if _HAS_SPIKINGJELLY else None

    def reset_state(self) -> None:
        super().reset_state()
        if self._sj is not None:
            self._sj.reset()

    def step(self, x_step: torch.Tensor) -> torch.Tensor:
        if self._sj is not None:
            return self._sj(x_step)
        self._init_state(x_step)
        assert self.v is not None
        self.v = self.v * self.decay + x_step
        spike = (self.v >= self.threshold).to(x_step.dtype)
        self.v = self.v * (1.0 - spike)
        return spike


class ParametricLIFNode(_StatefulNeuronBase):
    def __init__(self, threshold: float = 1.0, init_decay: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold
        init_logit = torch.logit(torch.tensor(init_decay))
        self.decay_logit = nn.Parameter(init_logit)
        self._sj = (
            sj_neuron.ParametricLIFNode(v_threshold=threshold, init_tau=max(1e-4, 1.0 / (1.0 - init_decay)), step_mode="s")
            if _HAS_SPIKINGJELLY
            else None
        )

    def reset_state(self) -> None:
        super().reset_state()
        if self._sj is not None:
            self._sj.reset()

    @property
    def decay(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_logit)

    def step(self, x_step: torch.Tensor) -> torch.Tensor:
        if self._sj is not None:
            return self._sj(x_step)
        self._init_state(x_step)
        assert self.v is not None
        decay = self.decay.to(dtype=x_step.dtype, device=x_step.device)
        self.v = self.v * decay + x_step
        spike = (self.v >= self.threshold).to(x_step.dtype)
        self.v = self.v * (1.0 - spike)
        return spike


class Delay(nn.Module):
    """Temporal delay line."""

    is_stateful_temporal = True

    def __init__(self, delay_steps: int = 1) -> None:
        super().__init__()
        if delay_steps < 0:
            raise ValueError("delay_steps must be >= 0")
        self.delay_steps = delay_steps
        self._queue: deque[torch.Tensor] = deque()

    def reset_state(self) -> None:
        self._queue.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"Delay expects [T,B,...], got {tuple(x.shape)}")
        outs = []
        for t in range(x.shape[0]):
            self._queue.append(x[t])
            if len(self._queue) <= self.delay_steps:
                outs.append(torch.zeros_like(x[t]))
            else:
                outs.append(self._queue.popleft())
        return torch.stack(outs, dim=0)

