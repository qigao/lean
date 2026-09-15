from __future__ import annotations

import torch
from torch import nn


class StreamingGRU(nn.Module):
    def __init__(self, *, num_classes: int, width: int = 64) -> None:
        super().__init__()
        if type(num_classes) is not int or num_classes <= 1:
            raise ValueError("num_classes must be at least 2")
        if type(width) is not int or width <= 0:
            raise ValueError("GRU width must be positive")
        self.num_classes = num_classes
        self.width = width
        self.cell = nn.GRUCell(25 * 15, width)
        self.readout = nn.Linear(width, num_classes)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return torch.zeros((batch_size, self.width), device=device, dtype=dtype)

    def _validate_state(self, state: torch.Tensor, *, batch_size: int) -> None:
        if type(state) is not torch.Tensor or state.shape != (batch_size, self.width):
            raise ValueError("GRU state shape mismatch")
        if not torch.is_floating_point(state) or not torch.isfinite(state).all().item():
            raise ValueError("GRU state must be finite floating point")

    def step(self, frame_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if type(frame_t) is not torch.Tensor or frame_t.ndim != 3 or frame_t.shape[1:] != (25, 15):
            raise ValueError("GRU frame input must have shape [batch,25,15]")
        if frame_t.shape[0] <= 0:
            raise ValueError("GRU batch must be non-empty")
        if not torch.is_floating_point(frame_t) or not torch.isfinite(frame_t).all().item():
            raise ValueError("GRU frame input must be finite floating point")
        self._validate_state(state, batch_size=frame_t.shape[0])
        return self.cell(frame_t.reshape(frame_t.shape[0], 25 * 15), state)

    def logits(self, state: torch.Tensor) -> torch.Tensor:
        if type(state) is not torch.Tensor or state.ndim != 2:
            raise ValueError("GRU state must be batched")
        self._validate_state(state, batch_size=state.shape[0])
        return self.readout(state)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if type(sequence) is not torch.Tensor or sequence.ndim != 4 or sequence.shape[2:] != (25, 15):
            raise ValueError("GRU input must have shape [batch,time,25,15]")
        if sequence.shape[0] <= 0 or sequence.shape[1] <= 0:
            raise ValueError("GRU input dimensions must be positive")
        if not torch.is_floating_point(sequence) or not torch.isfinite(sequence).all().item():
            raise ValueError("GRU input must be finite floating point")
        state = self.initial_state(sequence.shape[0], sequence.device, sequence.dtype)
        for index in range(sequence.shape[1]):
            state = self.step(sequence[:, index], state)
        return self.logits(state)
