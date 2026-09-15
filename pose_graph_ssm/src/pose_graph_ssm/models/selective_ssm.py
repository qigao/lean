from __future__ import annotations

import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from .base import RMSNorm


_SSM_SPEC = {
    "id": "selective-diagonal-ssm-v1",
    "a": "-softplus(A_log)",
    "delta": "softplus(W_delta*u+b_delta)",
    "b": "tanh(W_B*u)",
    "c": "sigmoid(W_C*u)",
    "decay": "exp(clamp(delta*A,-20,0))",
    "state": "decay*state+(1-decay)*b",
    "readout": "c*state+D*u",
    "output": "rmsnorm(u+W_out(readout))",
    "clamp_min": -20.0,
    "clamp_max": 0.0,
}


def ssm_spec_fingerprint() -> str:
    encoded = json.dumps(_SSM_SPEC, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SelectiveSSMBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        if type(width) is not int or width <= 0:
            raise ValueError("SelectiveSSMBlock width must be a positive integer")
        self.width = width
        self.A_log = nn.Parameter(torch.zeros(width))
        self.D = nn.Parameter(torch.ones(width))
        self.delta_projection = nn.Linear(width, width, bias=True)
        self.B_projection = nn.Linear(width, width, bias=False)
        self.C_projection = nn.Linear(width, width, bias=False)
        self.output_projection = nn.Linear(width, width, bias=False)
        self.norm = RMSNorm(width)

    def initial_state(
        self,
        leading_shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if type(leading_shape) is not tuple or not leading_shape or any(type(v) is not int or v <= 0 for v in leading_shape):
            raise ValueError("SSM leading_shape must contain positive integers")
        return torch.zeros((*leading_shape, self.width), device=device, dtype=dtype)

    def _validate_input(self, value: torch.Tensor) -> None:
        if type(value) is not torch.Tensor or value.ndim < 2 or value.shape[-1] != self.width:
            raise ValueError("SSM input must end with configured width and include a batch dimension")
        if not torch.is_floating_point(value) or not torch.isfinite(value).all().item():
            raise ValueError("SSM input must be finite floating point")

    def _validate_state(self, state: torch.Tensor, value: torch.Tensor) -> None:
        if type(state) is not torch.Tensor or state.shape != value.shape:
            raise ValueError("SSM state shape must match input shape")
        if not torch.is_floating_point(state) or not torch.isfinite(state).all().item():
            raise ValueError("SSM state must be finite floating point")

    def decay(self, value: torch.Tensor) -> torch.Tensor:
        self._validate_input(value)
        delta = F.softplus(self.delta_projection(value))
        stable_a = -F.softplus(self.A_log).to(device=value.device, dtype=value.dtype)
        exponent = torch.clamp(delta * stable_a, min=-20.0, max=0.0)
        return torch.exp(exponent)

    def step(self, value: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_input(value)
        self._validate_state(state, value)
        decay = self.decay(value)
        b_value = torch.tanh(self.B_projection(value))
        c_value = torch.sigmoid(self.C_projection(value))
        next_state = decay * state + (1.0 - decay) * b_value
        readout = c_value * next_state + self.D.to(device=value.device, dtype=value.dtype) * value
        output = self.norm(value + self.output_projection(readout))
        if not torch.isfinite(output).all().item() or not torch.isfinite(next_state).all().item():
            raise ValueError("SSM produced nonfinite state or output")
        return output, next_state
