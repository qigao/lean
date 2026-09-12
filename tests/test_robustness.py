from __future__ import annotations

import torch

from yolo_flywire.eval import RobustnessReport, add_keypoint_noise, mask_keypoints
from yolo_flywire.graphs import DirectedGraph, lesion_graph
from yolo_flywire.manifests import RunManifest


def _encoded_fixture() -> torch.Tensor:
    # K=2 keypoints => feature layout is 2K xy, 2K velocity, K confidence, K mask.
    return torch.tensor(
        [[
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [0.2, 0.1, 1.2, 1.1, 0.2, 0.1, 0.2, 0.1, 1.0, 1.0, 1.0, 1.0],
            [0.4, 0.2, 1.4, 1.2, 0.2, 0.1, 0.2, 0.1, 1.0, 1.0, 1.0, 1.0],
        ]],
        dtype=torch.float32,
    )


def test_keypoint_perturbations_are_seed_deterministic_and_non_mutating():
    x = _encoded_fixture()
    original = x.clone()
    a = add_keypoint_noise(x, sigma=0.05, seed=17)
    b = add_keypoint_noise(x, sigma=0.05, seed=17)
    assert torch.equal(a, b)
    assert torch.equal(x, original)

    m1 = mask_keypoints(x, probability=0.5, seed=23)
    m2 = mask_keypoints(x, probability=0.5, seed=23)
    assert torch.equal(m1, m2)
    assert torch.equal(x, original)


def test_lesion_graph_removes_only_declared_node_and_edge():
    graph = DirectedGraph(
        num_nodes=5,
        src=(0, 0, 1, 2, 3, 4),
        dst=(1, 2, 2, 3, 4, 0),
        weight=(1, 2, 1, 1, 3, 1),
    )
    lesioned = lesion_graph(graph, node_ids=(2,), edge_indices=(4,))
    assert lesioned.num_nodes == 4
    assert lesioned.num_edges == 2
    assert sorted(lesioned.weight) == [1, 1]


def test_robustness_evidence_does_not_upgrade_topology_claim():
    report = RobustnessReport.example()
    assert report.clean.macro_f1 >= 0.0

    robustness = RunManifest.example(
        claim="robustness_advantage",
        compared_families=("flywire",),
    )
    robustness.validate_claim_boundary()

    topology = RunManifest.example(
        claim="topology_specific_advantage",
        compared_families=("flywire",),
    )
    try:
        topology.validate_claim_boundary()
    except ValueError as exc:
        assert "rewired" in str(exc)
    else:
        raise AssertionError("robustness evidence must not imply topology evidence")
