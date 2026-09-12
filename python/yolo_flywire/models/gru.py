from __future__ import annotations

import torch
from torch import nn


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return hidden[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.encode(x))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
