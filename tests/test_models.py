import torch

from yolo_flywire.graphs import DirectedGraph, random_sparse_graph
from yolo_flywire.models import GRUClassifier, GraphRecurrentClassifier


def test_models_share_batch_time_feature_input_contract():
    x = torch.randn(4, 12, 18)
    gru = GRUClassifier(18, 16, 6)
    graph = random_sparse_graph(12, 24, seed=1)
    grnn = GraphRecurrentClassifier(18, graph, node_dim=8, num_classes=6)
    assert gru(x).shape == (4, 6)
    assert grnn(x).shape == (4, 6)


def test_graph_recurrent_latent_is_topology_sensitive_with_fixed_parameters():
    x = torch.arange(1, 1 + 2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3) / 10.0
    graph_a = DirectedGraph(
        num_nodes=4,
        src=(0, 1, 2, 3),
        dst=(1, 2, 3, 0),
        weight=(1.0, 1.0, 1.0, 1.0),
    )
    graph_b = DirectedGraph(
        num_nodes=4,
        src=(0, 0, 1, 2),
        dst=(1, 2, 3, 3),
        weight=(1.0, 1.0, 1.0, 1.0),
    )
    a = GraphRecurrentClassifier(3, graph_a, node_dim=4, num_classes=2)
    b = GraphRecurrentClassifier(3, graph_b, node_dim=4, num_classes=2)
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        for parameter in a.parameters():
            parameter.fill_(0.1)
        b.load_state_dict(a.state_dict())
    assert not torch.allclose(a.encode(x), b.encode(x))
