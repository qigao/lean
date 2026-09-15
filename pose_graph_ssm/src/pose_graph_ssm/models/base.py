from __future__ import annotations

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMS normalization with multiplicative scale only.

    The absence of an additive bias is deliberate: standardized zero input must
    remain zero throughout the no-observation path used by retention/dropout
    experiments.
    """

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
