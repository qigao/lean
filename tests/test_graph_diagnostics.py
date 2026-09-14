import math

import torch

import yolo_flywire.graphs as graphs
import yolo_flywire.models.graph_rnn as graph_rnn
from yolo_flywire.pose_batches import PoseBatch


def _graph():
    return graphs.DirectedGraph(
        num_nodes=3,
        src=(0, 1, 2, 2),
        dst=(0, 0, 1, 2),
        weight=(100.0, 20.0, 5.0, 50.0),
    )


def test_graph_diagnostic_transform_normalizes_incoming_weight_and_can_drop_diagonal():
    transform = getattr(graphs, "diagnostic_graph", None)
    assert transform is not None
    normalized = transform(_graph(), weight_policy="log1p_incoming_l1", diagonal_policy="drop")
    assert all(source != target for source, target in zip(normalized.src, normalized.dst))
    incoming = {}
    for target, weight in zip(normalized.dst, normalized.weight):
        incoming[target] = incoming.get(target, 0.0) + abs(weight)
    assert incoming
    assert all(math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-7) for total in incoming.values())


def test_graph_activation_trace_exposes_raw_saturation_without_changing_model_api():
    trace = getattr(graph_rnn, "diagnose_graph_recurrence", None)
    assert trace is not None
    model = graph_rnn.GraphRecurrentClassifier(4, _graph(), 2, 3)
    with torch.no_grad():
        model.input_projection.weight.fill_(0.25)
        model.input_projection.bias.zero_()
        model.self_projection.weight.copy_(torch.eye(2))
        model.message_projection.weight.copy_(torch.eye(2))
    x = torch.ones((1, 3, 4), dtype=torch.float32)
    report = trace(model, x)
    assert report["steps"] == 3
    assert len(report["per_step"]) == 3
    assert report["per_step"][1]["saturation_fraction"] > 0.0
    assert report["readout_dim"] == 2
    assert report["node_state_dim"] == 6


def test_graph_diagnostic_readout_can_preserve_node_identity_under_same_recurrence():
    classifier = getattr(graph_rnn, "GraphDiagnosticClassifier", None)
    assert classifier is not None
    mean_model = classifier(4, _graph(), 2, 3, readout_policy="mean")
    flat_model = classifier(4, _graph(), 2, 3, readout_policy="flatten")
    x = torch.zeros((2, 5, 4), dtype=torch.float32)
    assert tuple(mean_model.encode(x).shape) == (2, 2)
    assert tuple(flat_model.encode(x).shape) == (2, 6)
    assert flat_model.parameter_count() < 50000


def test_graph_diagnostic_seed_routing_injects_only_declared_nodes():
    classifier = getattr(graph_rnn, "GraphDiagnosticClassifier", None)
    assert classifier is not None
    model = classifier(
        4, _graph(), 2, 3,
        input_policy="selected_nodes",
        input_node_indices=(0, 2),
    )
    with torch.no_grad():
        model.input_projection.weight.fill_(1.0)
        model.input_projection.bias.zero_()
    injected = model._inject(torch.ones((1, 4), dtype=torch.float32))
    assert torch.count_nonzero(injected[:, 1, :]).item() == 0
    assert torch.count_nonzero(injected[:, 0, :]).item() > 0
    assert torch.count_nonzero(injected[:, 2, :]).item() > 0


def test_graph_diagnostic_padded_execution_keeps_finished_states_and_node_aware_readout():
    graph = graphs.DirectedGraph(
        num_nodes=3,
        src=(0, 1, 2),
        dst=(1, 2, 0),
        weight=(1.0, 1.0, 1.0),
    )
    model = graph_rnn.GraphDiagnosticClassifier(
        121, graph, 2, 2,
        input_policy="selected_nodes",
        input_node_indices=(0, 2),
        readout_policy="flatten",
    )
    features = torch.zeros((2, 3, 121), dtype=torch.float32)
    features[0, :2, 0] = 1.0
    features[1, :, 0] = 1.0
    lengths = torch.tensor([2, 3], dtype=torch.int64)
    mask = torch.tensor([[True, True, False], [True, True, True]], dtype=torch.bool)
    batch = PoseBatch(features, lengths, mask)
    encoded = model.encode_padded(batch)
    logits = model.forward_padded(batch)
    assert tuple(encoded.shape) == (2, 6)
    assert tuple(logits.shape) == (2, 2)
    assert torch.isfinite(encoded).all().item()
    assert torch.isfinite(logits).all().item()


def test_selected_node_high_dim_diagnostic_fits_shared_parameter_ceiling():
    graph = graphs.DirectedGraph(
        num_nodes=187,
        src=tuple(range(187)),
        dst=tuple((index + 1) % 187 for index in range(187)),
        weight=tuple(1.0 for _ in range(187)),
    )
    model = graph_rnn.GraphDiagnosticClassifier(
        121, graph, 23, 6,
        input_policy="selected_nodes",
        input_node_indices=(122, 123, 124, 125, 126, 127, 128, 129),
        readout_policy="flatten",
    )
    assert model.parameter_count() == 49318
    assert model.parameter_count() <= 50000
