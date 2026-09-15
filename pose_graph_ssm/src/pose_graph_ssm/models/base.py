from __future__ import annotations

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMS normalization with multiplicative scale only."""

    def __init__(self, width: int, *, epsilon: float = 1e-6) -> None:
        super().__init__()
        if type(width) is not int or width <= 0:
            raise ValueError("RMSNorm width must be a positive integer")
        if type(epsilon) not in (int, float) or isinstance(epsilon, bool):
            raise ValueError("RMSNorm epsilon must be numeric")
        value = float(epsilon)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("RMSNorm epsilon must be positive and finite")
        self.width = width
        self.epsilon = value
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if type(value) is not torch.Tensor or value.ndim < 1 or value.shape[-1] != self.width:
            raise ValueError("RMSNorm input must end with the configured width")
        if not torch.is_floating_point(value) or not torch.isfinite(value).all().item():
            raise ValueError("RMSNorm input must be finite floating point")
        rms = torch.mean(value * value, dim=-1, keepdim=True)
        normalized = value * torch.rsqrt(rms + self.epsilon)
        return normalized * self.weight


class JointPooling(nn.Module):
    """Shared learned pooling used by both graph model arms."""

    def __init__(self, width: int) -> None:
        super().__init__()
        if type(width) is not int or width <= 0:
            raise ValueError("JointPooling width must be positive")
        self.width = width
        self.score = nn.Linear(width, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if type(value) is not torch.Tensor or value.ndim != 3 or value.shape[1:] != (25, self.width):
            raise ValueError("JointPooling input must have shape [batch,25,width]")
        if not torch.is_floating_point(value) or not torch.isfinite(value).all().item():
            raise ValueError("JointPooling input must be finite floating point")
        weights = torch.softmax(self.score(value).squeeze(-1), dim=1)
        return torch.sum(weights.unsqueeze(-1) * value, dim=1)
