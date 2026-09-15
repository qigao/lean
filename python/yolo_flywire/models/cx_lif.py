from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import torch
from torch import nn

from ..cx_artifact import CxArtifact


@dataclass(frozen=True)
class CxDynamics:
    tau_membrane: float
    synaptic_decay: float
    refractory_steps: int
    threshold: float
    reset: float
    recurrent_delay_steps: int
    recurrent_gain: float
    magnitude_policy: str

    def __post_init__(self) -> None:
        for name in ("tau_membrane", "synaptic_decay", "threshold", "reset", "recurrent_gain"):
            value = getattr(self, name)
            if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite numeric")
        if self.tau_membrane <= 0.0:
            raise ValueError("tau_membrane must be positive")
        if not 0.0 <= self.synaptic_decay < 1.0:
            raise ValueError("synaptic_decay must be in [0,1)")
        if type(self.refractory_steps) is not int or self.refractory_steps < 0:
            raise ValueError("refractory_steps must be a non-negative integer")
        if self.threshold <= self.reset:
            raise ValueError("threshold must exceed reset")
        if type(self.recurrent_delay_steps) is not int or self.recurrent_delay_steps != 1:
            raise ValueError("Gate 1 V1 requires exactly one recurrent delay step")
        if self.recurrent_gain <= 0.0:
            raise ValueError("recurrent_gain must be positive")
        if self.magnitude_policy not in ("raw", "log1p"):
            raise ValueError("magnitude_policy must be raw or log1p")

    @property
    def membrane_decay(self) -> float:
        return math.exp(-1.0 / float(self.tau_membrane))


class CxLifState(NamedTuple):
    membrane: torch.Tensor
    synaptic: torch.Tensor
    refractory: torch.Tensor
    spikes: torch.Tensor


@dataclass(frozen=True)
class CxRunStats:
    steps: int
    batch_size: int
    total_spikes: int
    recurrent_synaptic_events: int
    active_neurons_per_step: tuple[int, ...]
    peak_state_bytes: int


class CxLifClassifier(nn.Module):
    """Fixed recurrent FlyWire reservoir with biological I/O routing.

    R0 treats the connectome as the experimental condition. Recurrent source,
    target, and signed weights are non-persistent buffers, so state-dict loading
    transfers only trainable input/readout parameters and cannot overwrite the
    topology of another experimental arm.
    """

    def __init__(
        self,
        input_dim: int,
        artifact: CxArtifact,
        num_classes: int,
        *,
        dynamics: CxDynamics,
    ) -> None:
        super().__init__()
        if type(input_dim) is not int or input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if type(num_classes) is not int or num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not artifact.input_indices or not artifact.output_indices:
            raise ValueError("CX LIF requires non-empty biological input and output populations")

        self.num_nodes = artifact.graph.num_nodes
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dynamics = dynamics

        self.register_buffer(
            "input_indices",
            torch.tensor(artifact.input_indices, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "output_indices",
            torch.tensor(artifact.output_indices, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "recurrent_source",
            torch.tensor(artifact.graph.src, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "recurrent_target",
            torch.tensor(artifact.graph.dst, dtype=torch.int64),
            persistent=False,
        )

        magnitude = torch.tensor(artifact.graph.weight, dtype=torch.float32)
        if dynamics.magnitude_policy == "log1p":
            if torch.any(magnitude < 0).item():
                raise ValueError("log1p CX recurrent magnitude requires non-negative synapse counts")
            magnitude = torch.log1p(magnitude)
        signs = torch.tensor(artifact.edge_signs, dtype=torch.float32)
        recurrent_weight = magnitude * signs * float(dynamics.recurrent_gain)
        self.register_buffer("recurrent_weight", recurrent_weight, persistent=False)

        self.input_projection = nn.Linear(input_dim, len(artifact.input_indices), bias=False)
        self.readout = nn.Linear(len(artifact.output_indices), num_classes)

    def initial_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CxLifState:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        shape = (batch_size, self.num_nodes)
        return CxLifState(
            membrane=torch.zeros(shape, device=device, dtype=dtype),
            synaptic=torch.zeros(shape, device=device, dtype=dtype),
            refractory=torch.zeros(shape, device=device, dtype=torch.int64),
            spikes=torch.zeros(shape, device=device, dtype=dtype),
        )

    def inject(self, events_t: torch.Tensor) -> torch.Tensor:
        if events_t.ndim != 2 or events_t.shape[1] != self.input_dim:
            raise ValueError("CX LIF frame input must have shape [batch,input_dim]")
        if not torch.is_floating_point(events_t) or not torch.isfinite(events_t).all().item():
            raise ValueError("CX LIF frame input must be finite floating point")
        projected = self.input_projection(events_t)
        injected = events_t.new_zeros((events_t.shape[0], self.num_nodes))
        return injected.index_copy(1, self.input_indices, projected)

    def _recurrent_current(self, spikes: torch.Tensor) -> torch.Tensor:
        if spikes.ndim != 2 or spikes.shape[1] != self.num_nodes:
            raise ValueError("CX LIF spike state shape mismatch")
        current = spikes.new_zeros(spikes.shape)
        if self.recurrent_source.numel() == 0:
            return current
        edge_activity = spikes.index_select(1, self.recurrent_source)
        edge_activity = edge_activity * self.recurrent_weight.to(dtype=spikes.dtype)
        return current.index_add(1, self.recurrent_target, edge_activity)

    def step(self, events_t: torch.Tensor, state: CxLifState) -> CxLifState:
        if state.membrane.shape != (events_t.shape[0], self.num_nodes):
            raise ValueError("CX LIF state batch/node shape mismatch")
        if not (
            state.synaptic.shape == state.membrane.shape
            and state.refractory.shape == state.membrane.shape
            and state.spikes.shape == state.membrane.shape
        ):
            raise ValueError("CX LIF state tensors must share one shape")

        injected = self.inject(events_t)
        recurrent = self._recurrent_current(state.spikes)
        synaptic = float(self.dynamics.synaptic_decay) * state.synaptic + recurrent
        membrane = float(self.dynamics.membrane_decay) * state.membrane + synaptic + injected

        refractory_active = state.refractory > 0
        membrane = torch.where(
            refractory_active,
            torch.as_tensor(self.dynamics.reset, dtype=membrane.dtype, device=membrane.device),
            membrane,
        )
        spikes_bool = (membrane >= float(self.dynamics.threshold)) & ~refractory_active
        spikes = spikes_bool.to(dtype=membrane.dtype)
        membrane = torch.where(
            spikes_bool,
            torch.as_tensor(self.dynamics.reset, dtype=membrane.dtype, device=membrane.device),
            membrane,
        )
        refractory = torch.clamp(state.refractory - 1, min=0)
        if self.dynamics.refractory_steps:
            refractory = torch.where(
                spikes_bool,
                torch.full_like(refractory, self.dynamics.refractory_steps),
                refractory,
            )
        return CxLifState(
            membrane=membrane,
            synaptic=synaptic,
            refractory=refractory,
            spikes=spikes,
        )

    @staticmethod
    def _state_bytes(state: CxLifState) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in state)

    def encode(self, events: torch.Tensor) -> tuple[torch.Tensor, CxRunStats]:
        if events.ndim != 3 or events.shape[0] <= 0 or events.shape[1] <= 0 or events.shape[2] != self.input_dim:
            raise ValueError("CX LIF input must have shape [batch,time,input_dim]")
        if not torch.is_floating_point(events) or not torch.isfinite(events).all().item():
            raise ValueError("CX LIF input must be finite floating point")

        state = self.initial_state(batch_size=events.shape[0], device=events.device, dtype=events.dtype)
        output_trace = events.new_zeros((events.shape[0], self.output_indices.numel()))
        total_spikes = 0
        recurrent_synaptic_events = 0
        active_per_step: list[int] = []
        peak_state_bytes = self._state_bytes(state)

        for time_index in range(events.shape[1]):
            if self.recurrent_source.numel():
                recurrent_synaptic_events += int(
                    torch.count_nonzero(state.spikes.index_select(1, self.recurrent_source)).item()
                )
            state = self.step(events[:, time_index, :], state)
            step_spikes = int(torch.count_nonzero(state.spikes).item())
            total_spikes += step_spikes
            active_per_step.append(step_spikes)
            peak_state_bytes = max(peak_state_bytes, self._state_bytes(state))
            output_trace = output_trace + state.spikes.index_select(1, self.output_indices)
            output_trace = output_trace + state.membrane.index_select(1, self.output_indices)

        latent = output_trace / float(events.shape[1])
        return latent, CxRunStats(
            steps=int(events.shape[1]),
            batch_size=int(events.shape[0]),
            total_spikes=total_spikes,
            recurrent_synaptic_events=recurrent_synaptic_events,
            active_neurons_per_step=tuple(active_per_step),
            peak_state_bytes=peak_state_bytes,
        )

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        latent, _ = self.encode(events)
        return self.readout(latent)
