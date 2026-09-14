from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..graphs import DirectedGraph
from ..pose_batches import PoseBatch
from ._padded import validate_pose_batch


def _adjacency(graph: DirectedGraph) -> torch.Tensor:
    adjacency = torch.zeros((graph.num_nodes, graph.num_nodes), dtype=torch.float32)
    for source, target, weight in zip(graph.src, graph.dst, graph.weight):
        adjacency[target, source] = float(weight)
    return adjacency


class GraphRecurrentClassifier(nn.Module):
    """Recurrent message passing over a fixed directed graph topology."""

    def __init__(
        self,
        input_dim: int,
        graph: DirectedGraph,
        node_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        if node_dim <= 0:
            raise ValueError("node_dim must be positive")
        self.num_nodes = graph.num_nodes
        self.node_dim = node_dim

        # Topology is experimental condition, not a learned parameter and not overwritten by state_dict loading.
        self.register_buffer("adjacency", _adjacency(graph), persistent=False)

        self.input_projection = nn.Linear(input_dim, graph.num_nodes * node_dim)
        self.self_projection = nn.Linear(node_dim, node_dim, bias=False)
        self.message_projection = nn.Linear(node_dim, node_dim, bias=False)
        self.readout = nn.Linear(node_dim, num_classes)

    def _step(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = x_t.shape[0]
        injected = self.input_projection(x_t).reshape(batch_size, self.num_nodes, self.node_dim)
        messages = torch.einsum("ij,bjd->bid", self.adjacency, state)
        return torch.tanh(
            injected
            + self.self_projection(state)
            + self.message_projection(messages)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("expected [batch, time, feature] input")
        state = x.new_zeros((x.shape[0], self.num_nodes, self.node_dim))
        for time_index in range(x.shape[1]):
            state = self._step(x[:, time_index, :], state)
        return state.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.encode(x))

    def encode_padded(self, batch: PoseBatch) -> torch.Tensor:
        """Run the original recurrence only for actual observed sample steps."""
        validate_pose_batch(batch, input_dim=self.input_projection.in_features, model=self)
        x = batch.features
        state = x.new_zeros((x.shape[0], self.num_nodes, self.node_dim))
        for time_index in range(int(batch.lengths.max().item())):
            active = batch.time_mask[:, time_index].nonzero(as_tuple=False).flatten()
            updated = self._step(
                x[:, time_index, :].index_select(0, active),
                state.index_select(0, active),
            )
            state = state.index_copy(0, active, updated)
        return state.mean(dim=1)

    def forward_padded(self, batch: PoseBatch) -> torch.Tensor:
        return self.readout(self.encode_padded(batch))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class GraphDiagnosticClassifier(nn.Module):
    """Development-only recurrence variant that isolates input/readout bottlenecks.

    It deliberately does not replace GraphRecurrentClassifier or its frozen KTH
    comparison behavior. Callers must choose every non-baseline policy explicitly.
    """

    def __init__(
        self,
        input_dim: int,
        graph: DirectedGraph,
        node_dim: int,
        num_classes: int,
        *,
        input_policy: str = "dense_all_nodes",
        input_node_indices: tuple[int, ...] = (),
        readout_policy: str = "mean",
    ) -> None:
        super().__init__()
        if node_dim <= 0:
            raise ValueError("node_dim must be positive")
        if input_policy not in ("dense_all_nodes", "selected_nodes"):
            raise ValueError("unsupported graph diagnostic input policy")
        if readout_policy not in ("mean", "flatten"):
            raise ValueError("unsupported graph diagnostic readout policy")
        if type(input_node_indices) is not tuple or any(type(index) is not int for index in input_node_indices):
            raise ValueError("graph diagnostic input node indices must be an integer tuple")
        if input_policy == "dense_all_nodes":
            if input_node_indices:
                raise ValueError("dense graph diagnostic input must not declare selected nodes")
            projected_nodes = graph.num_nodes
        else:
            if (not input_node_indices or len(set(input_node_indices)) != len(input_node_indices)
                    or any(index < 0 or index >= graph.num_nodes for index in input_node_indices)):
                raise ValueError("selected graph diagnostic input nodes must be unique and in range")
            projected_nodes = len(input_node_indices)

        self.num_nodes = graph.num_nodes
        self.node_dim = node_dim
        self.input_policy = input_policy
        self.input_node_indices = input_node_indices
        self.readout_policy = readout_policy
        self.register_buffer("adjacency", _adjacency(graph), persistent=False)
        self.input_projection = nn.Linear(input_dim, projected_nodes * node_dim)
        self.self_projection = nn.Linear(node_dim, node_dim, bias=False)
        self.message_projection = nn.Linear(node_dim, node_dim, bias=False)
        readout_dim = node_dim if readout_policy == "mean" else graph.num_nodes * node_dim
        self.readout = nn.Linear(readout_dim, num_classes)

    def _inject(self, x_t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 2 or x_t.shape[1] != self.input_projection.in_features:
            raise ValueError("graph diagnostic frame input shape mismatch")
        batch = x_t.shape[0]
        projected = self.input_projection(x_t)
        if self.input_policy == "dense_all_nodes":
            return projected.reshape(batch, self.num_nodes, self.node_dim)
        values = projected.reshape(batch, len(self.input_node_indices), self.node_dim)
        injected = x_t.new_zeros((batch, self.num_nodes, self.node_dim))
        indices = torch.tensor(self.input_node_indices, dtype=torch.int64, device=x_t.device)
        return injected.index_copy(1, indices, values)

    def _step(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        injected = self._inject(x_t)
        messages = torch.einsum("ij,bjd->bid", self.adjacency, state)
        return torch.tanh(
            injected
            + self.self_projection(state)
            + self.message_projection(messages)
        )

    def _read_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.readout_policy == "mean":
            return state.mean(dim=1)
        return state.reshape(state.shape[0], self.num_nodes * self.node_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[2] != self.input_projection.in_features:
            raise ValueError("expected [batch, time, feature] diagnostic input")
        state = x.new_zeros((x.shape[0], self.num_nodes, self.node_dim))
        for time_index in range(x.shape[1]):
            state = self._step(x[:, time_index, :], state)
        return self._read_state(state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.encode(x))

    def encode_padded(self, batch: PoseBatch) -> torch.Tensor:
        validate_pose_batch(batch, input_dim=self.input_projection.in_features, model=self)
        x = batch.features
        state = x.new_zeros((x.shape[0], self.num_nodes, self.node_dim))
        for time_index in range(int(batch.lengths.max().item())):
            active = batch.time_mask[:, time_index].nonzero(as_tuple=False).flatten()
            updated = self._step(
                x[:, time_index, :].index_select(0, active),
                state.index_select(0, active),
            )
            state = state.index_copy(0, active, updated)
        return self._read_state(state)

    def forward_padded(self, batch: PoseBatch) -> torch.Tensor:
        return self.readout(self.encode_padded(batch))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def diagnose_graph_recurrence(
    model: GraphRecurrentClassifier | GraphDiagnosticClassifier,
    x: torch.Tensor,
) -> dict[str, Any]:
    """Measure recurrence scale/saturation without training or mutating model state."""
    if type(model) not in (GraphRecurrentClassifier, GraphDiagnosticClassifier):
        raise ValueError("graph recurrence diagnostic requires an explicit graph classifier")
    if (type(x) is not torch.Tensor or x.layout != torch.strided or x.ndim != 3
            or x.shape[0] <= 0 or x.shape[1] <= 0
            or x.shape[2] != model.input_projection.in_features
            or not torch.isfinite(x).all().item()):
        raise ValueError("graph recurrence diagnostic input must be finite [batch,time,feature]")

    state = x.new_zeros((x.shape[0], model.num_nodes, model.node_dim))
    per_step: list[dict[str, float | int]] = []
    with torch.no_grad():
        for time_index in range(x.shape[1]):
            frame = x[:, time_index, :]
            if type(model) is GraphDiagnosticClassifier:
                injected = model._inject(frame)
            else:
                injected = model.input_projection(frame).reshape(
                    frame.shape[0], model.num_nodes, model.node_dim,
                )
            messages = torch.einsum("ij,bjd->bid", model.adjacency, state)
            self_term = model.self_projection(state)
            message_term = model.message_projection(messages)
            preactivation = injected + self_term + message_term
            state = torch.tanh(preactivation)
            per_step.append({
                "time_index": time_index,
                "injected_rms": float(injected.square().mean().sqrt().item()),
                "message_rms": float(messages.square().mean().sqrt().item()),
                "message_term_rms": float(message_term.square().mean().sqrt().item()),
                "preactivation_rms": float(preactivation.square().mean().sqrt().item()),
                "state_rms": float(state.square().mean().sqrt().item()),
                "saturation_fraction": float((state.abs() >= 0.99).to(torch.float32).mean().item()),
            })

    return {
        "steps": int(x.shape[1]),
        "batch_size": int(x.shape[0]),
        "num_nodes": model.num_nodes,
        "node_dim": model.node_dim,
        "node_state_dim": model.num_nodes * model.node_dim,
        "readout_dim": model.readout.in_features,
        "per_step": per_step,
    }
