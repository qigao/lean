from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from ..graph import HumanGraphBlock
from .base import JointPooling, RMSNorm


class CausalTemporalConvBlock(nn.Module):
    def __init__(self, width: int, *, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        if type(width) is not int or width <= 0:
            raise ValueError("temporal conv width must be positive")
        if kernel_size != 3:
            raise ValueError("V1 temporal conv kernel_size must be 3")
        if type(dilation) is not int or dilation <= 0:
            raise ValueError("temporal conv dilation must be positive")
        self.width = width
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cache_length = (kernel_size - 1) * dilation
        self.current_projection = nn.Linear(width, width, bias=False)
        self.lag1_projection = nn.Linear(width, width, bias=False)
        self.lag2_projection = nn.Linear(width, width, bias=False)
        self.activation = nn.GELU()
        self.norm = RMSNorm(width)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return torch.zeros((batch_size, self.cache_length, 25, self.width), device=device, dtype=dtype)

    def _validate_state(self, state: torch.Tensor, *, batch_size: int) -> None:
        expected = (batch_size, self.cache_length, 25, self.width)
        if type(state) is not torch.Tensor or state.shape != expected:
            raise ValueError("temporal conv cache shape mismatch")
        if not torch.is_floating_point(state) or not torch.isfinite(state).all().item():
            raise ValueError("temporal conv cache must be finite floating point")

    def step(self, value: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if type(value) is not torch.Tensor or value.ndim != 3 or value.shape[1:] != (25, self.width):
            raise ValueError("temporal conv input must have shape [batch,25,width]")
        if value.shape[0] <= 0:
            raise ValueError("temporal conv batch must be non-empty")
        if not torch.is_floating_point(value) or not torch.isfinite(value).all().item():
            raise ValueError("temporal conv input must be finite floating point")
        self._validate_state(state, batch_size=value.shape[0])
        lag1 = state[:, -self.dilation]
        lag2 = state[:, -(2 * self.dilation)]
        update = (
            self.current_projection(value)
            + self.lag1_projection(lag1)
            + self.lag2_projection(lag2)
        )
        output = self.norm(value + self.activation(update))
        next_state = torch.cat((state[:, 1:], value.unsqueeze(1)), dim=1)
        return output, next_state


class GraphTCNState(NamedTuple):
    caches: tuple[torch.Tensor, torch.Tensor]
    representation: torch.Tensor


class GraphTCN(nn.Module):
    def __init__(self, *, num_classes: int, width: int = 64) -> None:
        super().__init__()
        if type(num_classes) is not int or num_classes <= 1:
            raise ValueError("num_classes must be at least 2")
        if type(width) is not int or width <= 0:
            raise ValueError("GraphTCN width must be positive")
        self.num_classes = num_classes
        self.width = width
        self.input_projection = nn.Linear(15, width, bias=False)
        self.graph_blocks = nn.ModuleList((HumanGraphBlock(width), HumanGraphBlock(width)))
        self.temporal_blocks = nn.ModuleList((
            CausalTemporalConvBlock(width, kernel_size=3, dilation=1),
            CausalTemporalConvBlock(width, kernel_size=3, dilation=2),
        ))
        self.pooling = JointPooling(width)
        self.readout = nn.Linear(width, num_classes)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> GraphTCNState:
        caches = tuple(block.initial_state(batch_size, device, dtype) for block in self.temporal_blocks)
        representation = torch.zeros((batch_size, 25, self.width), device=device, dtype=dtype)
        return GraphTCNState(caches=caches, representation=representation)  # type: ignore[arg-type]

    def _validate_state(self, state: GraphTCNState, *, batch_size: int) -> None:
        if type(state) is not GraphTCNState or len(state.caches) != 2:
            raise ValueError("GraphTCN state mismatch")
        for block, cache in zip(self.temporal_blocks, state.caches):
            block._validate_state(cache, batch_size=batch_size)
        if state.representation.shape != (batch_size, 25, self.width):
            raise ValueError("GraphTCN representation shape mismatch")
        if not torch.is_floating_point(state.representation) or not torch.isfinite(state.representation).all().item():
            raise ValueError("GraphTCN representation must be finite floating point")

    def step(self, frame_t: torch.Tensor, state: GraphTCNState) -> GraphTCNState:
        if type(frame_t) is not torch.Tensor or frame_t.ndim != 3 or frame_t.shape[1:] != (25, 15):
            raise ValueError("GraphTCN frame input must have shape [batch,25,15]")
        if frame_t.shape[0] <= 0:
            raise ValueError("GraphTCN batch must be non-empty")
        if not torch.is_floating_point(frame_t) or not torch.isfinite(frame_t).all().item():
            raise ValueError("GraphTCN frame input must be finite floating point")
        self._validate_state(state, batch_size=frame_t.shape[0])
        value = self.input_projection(frame_t)
        next_caches: list[torch.Tensor] = []
        for graph, temporal, cache in zip(self.graph_blocks, self.temporal_blocks, state.caches):
            value = graph(value)
            value, updated = temporal.step(value, cache)
            next_caches.append(updated)
        return GraphTCNState(caches=tuple(next_caches), representation=value)  # type: ignore[arg-type]

    def logits(self, state: GraphTCNState) -> torch.Tensor:
        if type(state) is not GraphTCNState or state.representation.ndim != 3:
            raise ValueError("GraphTCN state must be batched")
        self._validate_state(state, batch_size=state.representation.shape[0])
        return self.readout(self.pooling(state.representation))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if type(sequence) is not torch.Tensor or sequence.ndim != 4 or sequence.shape[2:] != (25, 15):
            raise ValueError("GraphTCN input must have shape [batch,time,25,15]")
        if sequence.shape[0] <= 0 or sequence.shape[1] <= 0:
            raise ValueError("GraphTCN input dimensions must be positive")
        if not torch.is_floating_point(sequence) or not torch.isfinite(sequence).all().item():
            raise ValueError("GraphTCN input must be finite floating point")
        state = self.initial_state(sequence.shape[0], sequence.device, sequence.dtype)
        for index in range(sequence.shape[1]):
            state = self.step(sequence[:, index], state)
        return self.logits(state)
