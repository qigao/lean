from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from .base import RMSNorm
from .selective_ssm import SelectiveSSMBlock


class SSMOnlyState(NamedTuple):
    recurrent: tuple[torch.Tensor, ...]
    representation: torch.Tensor


class SSMOnly(nn.Module):
    def __init__(self, *, num_classes: int, width: int = 64, blocks: int = 2) -> None:
        super().__init__()
        if type(num_classes) is not int or num_classes <= 1:
            raise ValueError("num_classes must be at least 2")
        if type(width) is not int or width <= 0:
            raise ValueError("SSMOnly width must be positive")
        if type(blocks) is not int or blocks <= 0:
            raise ValueError("SSMOnly blocks must be positive")
        self.num_classes = num_classes
        self.width = width
        self.blocks = blocks
        self.input_projection = nn.Linear(25 * 15, width, bias=False)
        self.input_norm = RMSNorm(width)
        self.ssm_blocks = nn.ModuleList(SelectiveSSMBlock(width) for _ in range(blocks))
        self.readout = nn.Linear(width, num_classes)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SSMOnlyState:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        recurrent = tuple(
            block.initial_state((batch_size,), device, dtype)
            for block in self.ssm_blocks
        )
        representation = torch.zeros((batch_size, self.width), device=device, dtype=dtype)
        return SSMOnlyState(recurrent=recurrent, representation=representation)

    def _validate_state(self, state: SSMOnlyState, *, batch_size: int) -> None:
        if type(state) is not SSMOnlyState:
            raise ValueError("SSMOnly state must be an SSMOnlyState")
        if len(state.recurrent) != self.blocks:
            raise ValueError("SSMOnly recurrent state count mismatch")
        expected = (batch_size, self.width)
        if state.representation.shape != expected:
            raise ValueError("SSMOnly representation shape mismatch")
        tensors = (*state.recurrent, state.representation)
        if any(tensor.shape != expected for tensor in tensors):
            raise ValueError("SSMOnly state shape mismatch")
        if any(not torch.is_floating_point(tensor) or not torch.isfinite(tensor).all().item() for tensor in tensors):
            raise ValueError("SSMOnly state must be finite floating point")

    def step(self, frame_t: torch.Tensor, state: SSMOnlyState) -> SSMOnlyState:
        if type(frame_t) is not torch.Tensor or frame_t.ndim != 3 or frame_t.shape[1:] != (25, 15):
            raise ValueError("SSMOnly frame input must have shape [batch,25,15]")
        if frame_t.shape[0] <= 0:
            raise ValueError("SSMOnly batch must be non-empty")
        if not torch.is_floating_point(frame_t) or not torch.isfinite(frame_t).all().item():
            raise ValueError("SSMOnly frame input must be finite floating point")
        self._validate_state(state, batch_size=frame_t.shape[0])

        value = frame_t.reshape(frame_t.shape[0], 25 * 15)
        value = self.input_norm(self.input_projection(value))
        next_recurrent: list[torch.Tensor] = []
        for block, recurrent_state in zip(self.ssm_blocks, state.recurrent):
            value, updated = block.step(value, recurrent_state)
            next_recurrent.append(updated)
        return SSMOnlyState(recurrent=tuple(next_recurrent), representation=value)

    def logits(self, state: SSMOnlyState) -> torch.Tensor:
        if type(state) is not SSMOnlyState or state.representation.ndim != 2:
            raise ValueError("SSMOnly state must be batched")
        self._validate_state(state, batch_size=state.representation.shape[0])
        return self.readout(state.representation)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if type(sequence) is not torch.Tensor or sequence.ndim != 4 or sequence.shape[2:] != (25, 15):
            raise ValueError("SSMOnly input must have shape [batch,time,25,15]")
        if sequence.shape[0] <= 0 or sequence.shape[1] <= 0:
            raise ValueError("SSMOnly input dimensions must be positive")
        if not torch.is_floating_point(sequence) or not torch.isfinite(sequence).all().item():
            raise ValueError("SSMOnly input must be finite floating point")
        state = self.initial_state(sequence.shape[0], sequence.device, sequence.dtype)
        for index in range(sequence.shape[1]):
            state = self.step(sequence[:, index], state)
        return self.logits(state)
