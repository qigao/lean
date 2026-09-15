from __future__ import annotations

import hashlib
import json

import torch
from torch import nn

from .features import NTU25_PARENT
from .models.base import RMSNorm


def _binary_adjacency() -> torch.Tensor:
    adjacency = torch.zeros((25, 25), dtype=torch.float32)
    for joint in range(25):
        adjacency[joint, joint] = 1.0
    for child, parent in enumerate(NTU25_PARENT):
        if parent < 0:
            continue
        adjacency[child, parent] = 1.0
        adjacency[parent, child] = 1.0
    return adjacency


def ntu25_adjacency() -> torch.Tensor:
    """Return frozen row-normalized NTU25 anatomical adjacency plus self loops."""
    adjacency = _binary_adjacency()
    degree = adjacency.sum(dim=1, keepdim=True)
    if torch.any(degree <= 0.0).item():
        raise ValueError("NTU25 adjacency contains an isolated joint")
    return adjacency / degree


def adjacency_fingerprint() -> str:
    adjacency = ntu25_adjacency().to(dtype=torch.float64, device="cpu")
    payload = {
        "id": "ntu25-undirected-self-row-normalized-v1",
        "parent_map": list(NTU25_PARENT),
        "adjacency": adjacency.tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class HumanGraphBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        if type(width) is not int or width <= 0:
            raise ValueError("HumanGraphBlock width must be a positive integer")
        self.width = width
        self.self_projection = nn.Linear(width, width, bias=False)
        self.neighbor_projection = nn.Linear(width, width, bias=False)
        self.activation = nn.GELU()
        self.norm = RMSNorm(width)
        self.register_buffer("adjacency", ntu25_adjacency(), persistent=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if type(value) is not torch.Tensor or value.ndim != 3 or value.shape[1:] != (25, self.width):
            raise ValueError("HumanGraphBlock input must have shape [batch,25,width]")
        if value.shape[0] <= 0:
            raise ValueError("HumanGraphBlock batch must be non-empty")
        if not torch.is_floating_point(value) or not torch.isfinite(value).all().item():
            raise ValueError("HumanGraphBlock input must be finite floating point")
        adjacency = self.adjacency.to(device=value.device, dtype=value.dtype)
        neighbors = torch.einsum("ij,bjc->bic", adjacency, value)
        update = self.self_projection(value) + self.neighbor_projection(neighbors)
        return self.norm(value + self.activation(update))
