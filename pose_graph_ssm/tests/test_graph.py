from __future__ import annotations

import torch

from pose_graph_ssm.features import NTU25_PARENT
from pose_graph_ssm.graph import HumanGraphBlock, adjacency_fingerprint, ntu25_adjacency


def test_ntu25_adjacency_contains_only_anatomical_links_and_self_loops():
    adjacency = ntu25_adjacency()
    assert adjacency.shape == (25, 25)
    torch.testing.assert_close(adjacency.sum(dim=1), torch.ones(25))
    allowed = {(i, i) for i in range(25)}
    for child, parent in enumerate(NTU25_PARENT):
        if parent >= 0:
            allowed.add((child, parent))
            allowed.add((parent, child))
    nonzero = {(int(i), int(j)) for i, j in (adjacency > 0).nonzero(as_tuple=False).tolist()}
    assert nonzero == allowed


def test_adjacency_fingerprint_is_stable_sha256():
    first = adjacency_fingerprint()
    second = adjacency_fingerprint()
    assert first == second
    assert len(first) == 64


def test_graph_block_preserves_shape_and_zero_input():
    block = HumanGraphBlock(64)
    x = torch.zeros(2, 25, 64)
    y = block(x)
    assert y.shape == x.shape
    torch.testing.assert_close(y, torch.zeros_like(y))


def test_graph_block_uses_bias_free_transforms_and_frozen_adjacency():
    block = HumanGraphBlock(64)
    assert block.self_projection.bias is None
    assert block.neighbor_projection.bias is None
    assert "adjacency" not in dict(block.named_parameters())
    assert "adjacency" in dict(block.named_buffers())
    torch.testing.assert_close(block.adjacency, ntu25_adjacency())
