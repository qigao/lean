from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from ..graph import HumanGraphBlock
from .base import JointPooling
from .selective_ssm import SelectiveSSMBlock


class GraphSSMState(NamedTuple):
    recurrent: tuple[torch.Tensor, torch.Tensor]
    representation: torch.Tensor


class GraphSSM(nn.Module):
    def __init__(self, *, num_classes: int, width: int = 64) -> None:
        super().__init__()
        if type(num_classes) is not int or num_classes <= 1:
            raise ValueError("num_classes must be at least 2")
        if type(width) is not int or width <= 0:
            raise ValueError("GraphSSM width must be positive")
        self.num_classes = num_classes
        self.width = width
        self.input_projection = nn.Linear(15, width, bias=False)
        self.graph_blocks = nn.ModuleList((HumanGraphBlock(width), HumanGraphBlock(width)))
        self.ssm_blocks = nn.ModuleList((SelectiveSSMBlock(width), SelectiveSSMBlock(width)))
        self.pooling = JointPooling(width)
        self.readout = nn.Linear(width, num_classes)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> GraphSSMState:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        recurrent = tuple(
            block.initial_state((batch_size, 25), device, dtype)
            for block in self.ssm_blocks
        )
        representation = torch.zeros((batch_size, 25, self.width), device=device, dtype=dtype)
        return GraphSSMState(recurrent=recurrent, representation=representation)  # type: ignore[arg-type]

    def _validate_state(self, state: GraphSSMState, *, batch_size: int) -> None:
        if type(state) is not GraphSSMState or len(state.recurrent) != 2:
            raise ValueError("GraphSSM state mismatch")
        expected = (batch_size, 25, self.width)
        if state.representation.shape != expected:
            raise ValueError("GraphSSM representation shape mismatch")
        tensors = (*state.recurrent, state.representation)
        if any(tensor.shape != expected for tensor in tensors):
            raise ValueError("GraphSSM state shape mismatch")
        if any(not torch.is_floating_point(tensor) or not torch.isfinite(tensor).all().item() for tensor in tensors):
            raise ValueError("GraphSSM state must be finite floating point")

    def step(self, frame_t: torch.Tensor, state: GraphSSMState) -> GraphSSMState:
        if type(frame_t) is not torch.Tensor or frame_t.ndim != 3 or frame_t.shape[1:] != (25, 15):
            raise ValueError("GraphSSM frame input must have shape [batch,25,15]")
        if frame_t.shape[0] <= 0:
            raise ValueError("GraphSSM batch must be non-empty")
        if not torch.is_floating_point(frame_t) or not torch.isfinite(frame_t).all().item():
            raise ValueError("GraphSSM frame input must be finite floating point")
        self._validate_state(state, batch_size=frame_t.shape[0])

        value = self.input_projection(frame_t)
        next_recurrent: list[torch.Tensor] = []
        for graph, ssm, recurrent_state in zip(self.graph_blocks, self.ssm_blocks, state.recurrent):
            value = graph(value)
            value, updated = ssm.step(value, recurrent_state)
            next_recurrent.append(updated)
        return GraphSSMState(recurrent=tuple(next_recurrent), representation=value)  # type: ignore[arg-type]

    def logits(self, state: GraphSSMState) -> torch.Tensor:
        if type(state) is not GraphSSMState or state.representation.ndim != 3:
            raise ValueError("GraphSSM state must be batched")
        self._validate_state(state, batch_size=state.representation.shape[0])
        return self.readout(self.pooling(state.representation))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if type(sequence) is not torch.Tensor or sequence.ndim != 4 or sequence.shape[2:] != (25, 15):
            raise ValueError("GraphSSM input must have shape [batch,time,25,15]")
        if sequence.shape[0] <= 0 or sequence.shape[1] <= 0:
            raise ValueError("GraphSSM input dimensions must be positive")
        if not torch.is_floating_point(sequence) or not torch.isfinite(sequence).all().item():
            raise ValueError("GraphSSM input must be finite floating point")
        state = self.initial_state(sequence.shape[0], sequence.device, sequence.dtype)
        for index in range(sequence.shape[1]):
            state = self.step(sequence[:, index], state)
        return self.logits(state)
