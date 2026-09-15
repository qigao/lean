from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from ..events import NTU25_PARENT
from .base import deterministic_silence_indices
from .rsnn import surrogate_spike


class SpikingGraphState(NamedTuple):
    membrane: torch.Tensor
    synaptic: torch.Tensor
    spikes: torch.Tensor


def _build_adjacency() -> torch.Tensor:
    adjacency = torch.zeros((25, 25), dtype=torch.float32)
    for child, parent in enumerate(NTU25_PARENT):
        adjacency[child, child] = 1.0
        if parent >= 0:
            adjacency[child, parent] = 1.0
            adjacency[parent, child] = 1.0
    degree = adjacency.sum(dim=1, keepdim=True)
    if torch.any(degree <= 0).item():
        raise ValueError("NTU25 adjacency contains an isolated joint")
    return adjacency / degree


class SpikingGraph(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        hidden_size: int = 32,
        membrane_decay: float = 0.95,
        synaptic_decay: float = 0.80,
        threshold: float = 1.0,
        reset: float = 0.0,
        surrogate_slope: float = 5.0,
    ) -> None:
        super().__init__()
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if type(num_classes) is not int or num_classes <= 1:
            raise ValueError("num_classes must be at least 2")
        self.num_nodes = 25
        self.input_dim = 9
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.membrane_decay = float(membrane_decay)
        self.synaptic_decay = float(synaptic_decay)
        self.threshold = float(threshold)
        self.reset = float(reset)
        self.surrogate_slope = float(surrogate_slope)
        if not 0.0 <= self.membrane_decay < 1.0:
            raise ValueError("membrane_decay must be in [0,1)")
        if not 0.0 <= self.synaptic_decay < 1.0:
            raise ValueError("synaptic_decay must be in [0,1)")
        if self.threshold <= self.reset:
            raise ValueError("threshold must exceed reset")
        if self.surrogate_slope <= 0.0:
            raise ValueError("surrogate_slope must be positive")

        self.input_projection = nn.Linear(9, hidden_size, bias=False)
        self.self_recurrent = nn.Linear(hidden_size, hidden_size, bias=False)
        self.neighbor_recurrent = nn.Linear(hidden_size, hidden_size, bias=False)
        self.readout = nn.Linear(hidden_size, num_classes)
        self.register_buffer("adjacency", _build_adjacency(), persistent=True)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SpikingGraphState:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        shape = (batch_size, self.num_nodes, self.hidden_size)
        zero = torch.zeros(shape, device=device, dtype=dtype)
        return SpikingGraphState(zero.clone(), zero.clone(), zero.clone())

    def _validate_state(self, state: SpikingGraphState, *, batch_size: int) -> None:
        if type(state) is not SpikingGraphState:
            raise ValueError("SpikingGraph state must be a SpikingGraphState")
        expected = (batch_size, self.num_nodes, self.hidden_size)
        if any(tensor.shape != expected for tensor in state):
            raise ValueError("SpikingGraph state shape mismatch")
        if any(not torch.is_floating_point(tensor) or not torch.isfinite(tensor).all().item() for tensor in state):
            raise ValueError("SpikingGraph state must be finite floating point")

    def neighbor_activity(self, spikes: torch.Tensor) -> torch.Tensor:
        if type(spikes) is not torch.Tensor or spikes.ndim != 3 or spikes.shape[1:] != (25, self.hidden_size):
            raise ValueError("SpikingGraph spikes must have shape [batch,25,hidden]")
        if not torch.is_floating_point(spikes) or not torch.isfinite(spikes).all().item():
            raise ValueError("SpikingGraph spikes must be finite floating point")
        adjacency = self.adjacency.to(device=spikes.device, dtype=spikes.dtype)
        return torch.einsum("ij,bjh->bih", adjacency, spikes)

    def step(self, events_t: torch.Tensor, state: SpikingGraphState) -> SpikingGraphState:
        if type(events_t) is not torch.Tensor or events_t.ndim != 3 or events_t.shape[1:] != (25, 9):
            raise ValueError("SpikingGraph frame input must have shape [batch,25,9]")
        if not torch.is_floating_point(events_t) or not torch.isfinite(events_t).all().item():
            raise ValueError("SpikingGraph frame input must be finite floating point")
        self._validate_state(state, batch_size=events_t.shape[0])
        neighbor = self.neighbor_activity(state.spikes)
        synaptic = (
            self.synaptic_decay * state.synaptic
            + self.input_projection(events_t)
            + self.self_recurrent(state.spikes)
            + self.neighbor_recurrent(neighbor)
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
        return SpikingGraphState(membrane, synaptic, spikes)

    def logits(self, state: SpikingGraphState) -> torch.Tensor:
        if type(state) is not SpikingGraphState or state.membrane.ndim != 3:
            raise ValueError("SpikingGraph state must be a batched SpikingGraphState")
        self._validate_state(state, batch_size=state.membrane.shape[0])
        pooled = (state.membrane + state.spikes).mean(dim=1)
        return self.readout(pooled)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        if type(events) is not torch.Tensor or events.ndim != 4 or events.shape[2:] != (25, 9):
            raise ValueError("SpikingGraph input must have shape [batch,time,25,9]")
        if events.shape[0] <= 0 or events.shape[1] <= 0:
            raise ValueError("SpikingGraph input dimensions must be positive")
        if not torch.is_floating_point(events) or not torch.isfinite(events).all().item():
            raise ValueError("SpikingGraph input must be finite floating point")
        state = self.initial_state(events.shape[0], events.device, events.dtype)
        for time_index in range(events.shape[1]):
            state = self.step(events[:, time_index], state)
        return self.logits(state)

    def silence_state(
        self,
        state: SpikingGraphState,
        *,
        fraction: float,
        seed: int,
    ) -> SpikingGraphState:
        if type(state) is not SpikingGraphState or state.membrane.ndim != 3:
            raise ValueError("SpikingGraph state must be a batched SpikingGraphState")
        self._validate_state(state, batch_size=state.membrane.shape[0])
        total_units = self.num_nodes * self.hidden_size
        indices = deterministic_silence_indices(total_units, fraction, seed)
        result = []
        for tensor in state:
            cloned = tensor.clone()
            flat = cloned.reshape(cloned.shape[0], total_units)
            flat[:, list(indices)] = 0.0
            result.append(flat.reshape_as(cloned))
        return SpikingGraphState(*result)
