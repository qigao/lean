from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from .base import deterministic_silence_indices


class _Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(value)
        ctx.slope = float(slope)
        return (value >= 0.0).to(dtype=value.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (value,) = ctx.saved_tensors
        denominator = (1.0 + ctx.slope * value.abs()).pow(2)
        return grad_output / denominator, None


def surrogate_spike(value: torch.Tensor, slope: float) -> torch.Tensor:
    return _Spike.apply(value, float(slope))


class RSNNState(NamedTuple):
    membrane: torch.Tensor
    synaptic: torch.Tensor
    spikes: torch.Tensor


class RSNN(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        num_classes: int,
        membrane_decay: float = 0.95,
        synaptic_decay: float = 0.80,
        threshold: float = 1.0,
        reset: float = 0.0,
        surrogate_slope: float = 5.0,
    ) -> None:
        super().__init__()
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if type(num_classes) is not int or num_classes <= 1:
            raise ValueError("num_classes must be at least 2")
        for name, value in (
            ("membrane_decay", membrane_decay),
            ("synaptic_decay", synaptic_decay),
            ("threshold", threshold),
            ("reset", reset),
            ("surrogate_slope", surrogate_slope),
        ):
            if type(value) not in (int, float) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
        if not 0.0 <= float(membrane_decay) < 1.0:
            raise ValueError("membrane_decay must be in [0,1)")
        if not 0.0 <= float(synaptic_decay) < 1.0:
            raise ValueError("synaptic_decay must be in [0,1)")
        if float(threshold) <= float(reset):
            raise ValueError("threshold must exceed reset")
        if float(surrogate_slope) <= 0.0:
            raise ValueError("surrogate_slope must be positive")

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.membrane_decay = float(membrane_decay)
        self.synaptic_decay = float(synaptic_decay)
        self.threshold = float(threshold)
        self.reset = float(reset)
        self.surrogate_slope = float(surrogate_slope)

        self.input_projection = nn.Linear(input_dim, hidden_size, bias=False)
        self.recurrent = nn.Linear(hidden_size, hidden_size, bias=False)
        self.readout = nn.Linear(hidden_size, num_classes)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RSNNState:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        shape = (batch_size, self.hidden_size)
        zero = torch.zeros(shape, device=device, dtype=dtype)
        return RSNNState(zero.clone(), zero.clone(), zero.clone())

    def _validate_state(self, state: RSNNState, *, batch_size: int) -> None:
        if type(state) is not RSNNState:
            raise ValueError("RSNN state must be an RSNNState")
        expected = (batch_size, self.hidden_size)
        if any(tensor.shape != expected for tensor in state):
            raise ValueError("RSNN state shape mismatch")
        if any(not torch.is_floating_point(tensor) or not torch.isfinite(tensor).all().item() for tensor in state):
            raise ValueError("RSNN state must be finite floating point")

    def step(self, events_t: torch.Tensor, state: RSNNState) -> RSNNState:
        if type(events_t) is not torch.Tensor or events_t.ndim != 2 or events_t.shape[1] != self.input_dim:
            raise ValueError("RSNN frame input must have shape [batch,input_dim]")
        if not torch.is_floating_point(events_t) or not torch.isfinite(events_t).all().item():
            raise ValueError("RSNN frame input must be finite floating point")
        self._validate_state(state, batch_size=events_t.shape[0])

        synaptic = (
            self.synaptic_decay * state.synaptic
            + self.input_projection(events_t)
            + self.recurrent(state.spikes)
        )
        membrane = self.membrane_decay * state.membrane + synaptic
        spikes = surrogate_spike(membrane - self.threshold, self.surrogate_slope)
        if self.reset == 0.0:
            membrane = membrane * (1.0 - spikes.detach())
        else:
            membrane = torch.where(
                spikes.detach().bool(),
                torch.as_tensor(self.reset, device=membrane.device, dtype=membrane.dtype),
                membrane,
            )
        return RSNNState(membrane=membrane, synaptic=synaptic, spikes=spikes)

    def logits(self, state: RSNNState) -> torch.Tensor:
        if type(state) is not RSNNState:
            raise ValueError("RSNN state must be an RSNNState")
        if state.membrane.ndim != 2:
            raise ValueError("RSNN state must have batch dimension")
        self._validate_state(state, batch_size=state.membrane.shape[0])
        return self.readout(state.membrane + state.spikes)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        if type(events) is not torch.Tensor or events.ndim != 3 or events.shape[2] != self.input_dim:
            raise ValueError("RSNN input must have shape [batch,time,input_dim]")
        if events.shape[0] <= 0 or events.shape[1] <= 0:
            raise ValueError("RSNN input dimensions must be positive")
        if not torch.is_floating_point(events) or not torch.isfinite(events).all().item():
            raise ValueError("RSNN input must be finite floating point")
        state = self.initial_state(events.shape[0], events.device, events.dtype)
        for time_index in range(events.shape[1]):
            state = self.step(events[:, time_index], state)
        return self.logits(state)

    def silence_state(self, state: RSNNState, *, fraction: float, seed: int) -> RSNNState:
        if type(state) is not RSNNState or state.membrane.ndim != 2:
            raise ValueError("RSNN state must be a batched RSNNState")
        self._validate_state(state, batch_size=state.membrane.shape[0])
        indices = deterministic_silence_indices(self.hidden_size, fraction, seed)
        result = []
        for tensor in state:
            cloned = tensor.clone()
            cloned[:, list(indices)] = 0.0
            result.append(cloned)
        return RSNNState(*result)
