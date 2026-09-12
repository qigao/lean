from __future__ import annotations

import torch
from torch import nn

from ..graphs import DirectedGraph


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

        adjacency = torch.zeros((graph.num_nodes, graph.num_nodes), dtype=torch.float32)
        for source, target, weight in zip(graph.src, graph.dst, graph.weight):
            adjacency[target, source] = float(weight)
        # Topology is experimental condition, not a learned parameter and not overwritten by state_dict loading.
        self.register_buffer("adjacency", adjacency, persistent=False)

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

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
