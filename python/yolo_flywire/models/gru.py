from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from ..pose_batches import PoseBatch
from ._padded import validate_pose_batch


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

    def encode_padded(self, batch: PoseBatch) -> torch.Tensor:
        """Return the last observed state in caller order, without padding steps."""
        validate_pose_batch(batch, input_dim=self.gru.input_size, model=self)
        packed = pack_padded_sequence(
            batch.features, batch.lengths, batch_first=True, enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return hidden[-1]

    def forward_padded(self, batch: PoseBatch) -> torch.Tensor:
        return self.readout(self.encode_padded(batch))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
