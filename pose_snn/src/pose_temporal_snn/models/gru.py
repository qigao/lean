from __future__ import annotations

import torch
from torch import nn

from .base import deterministic_silence_indices


class StreamingGRU(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if type(num_classes) is not int or num_classes <= 1:
            raise ValueError("num_classes must be at least 2")
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.cell = nn.GRUCell(input_dim, hidden_size)
        self.readout = nn.Linear(hidden_size, num_classes)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)

    def _validate_state(self, state: torch.Tensor, *, batch_size: int) -> None:
        if type(state) is not torch.Tensor or state.shape != (batch_size, self.hidden_size):
            raise ValueError("GRU state shape mismatch")
        if not torch.is_floating_point(state) or not torch.isfinite(state).all().item():
            raise ValueError("GRU state must be finite floating point")

    def step(self, events_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if type(events_t) is not torch.Tensor or events_t.ndim != 2 or events_t.shape[1] != self.input_dim:
            raise ValueError("GRU frame input must have shape [batch,input_dim]")
        if not torch.is_floating_point(events_t) or not torch.isfinite(events_t).all().item():
            raise ValueError("GRU frame input must be finite floating point")
        self._validate_state(state, batch_size=events_t.shape[0])
        return self.cell(events_t, state)

    def logits(self, state: torch.Tensor) -> torch.Tensor:
        if type(state) is not torch.Tensor or state.ndim != 2:
            raise ValueError("GRU state must be batched")
        self._validate_state(state, batch_size=state.shape[0])
        return self.readout(state)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        if type(events) is not torch.Tensor or events.ndim != 3 or events.shape[2] != self.input_dim:
            raise ValueError("GRU input must have shape [batch,time,input_dim]")
        if events.shape[0] <= 0 or events.shape[1] <= 0:
            raise ValueError("GRU input dimensions must be positive")
        if not torch.is_floating_point(events) or not torch.isfinite(events).all().item():
            raise ValueError("GRU input must be finite floating point")
        state = self.initial_state(events.shape[0], events.device, events.dtype)
        for time_index in range(events.shape[1]):
            state = self.step(events[:, time_index], state)
        return self.logits(state)

    def silence_state(self, state: torch.Tensor, *, fraction: float, seed: int) -> torch.Tensor:
        if type(state) is not torch.Tensor or state.ndim != 2:
            raise ValueError("GRU state must be batched")
        self._validate_state(state, batch_size=state.shape[0])
        indices = deterministic_silence_indices(self.hidden_size, fraction, seed)
        result = state.clone()
        result[:, list(indices)] = 0.0
        return result
