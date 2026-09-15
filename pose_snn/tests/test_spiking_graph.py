import torch

from pose_temporal_snn.events import NTU25_PARENT
from pose_temporal_snn.models.spiking_graph import SpikingGraph, SpikingGraphState


def test_forward_equals_manual_streaming_steps():
    torch.manual_seed(7)
    model = SpikingGraph(num_classes=10, hidden_size=32)
    events = torch.zeros(2, 8, 25, 9, dtype=torch.float32)
    state = model.initial_state(2, events.device, events.dtype)
    for time_index in range(events.shape[1]):
        state = model.step(events[:, time_index], state)
    torch.testing.assert_close(model(events), model.logits(state))


def test_neighbor_activity_is_restricted_to_anatomical_neighbors_and_self():
    model = SpikingGraph(num_classes=10, hidden_size=32)
    spikes = torch.zeros(1, 25, 32)
    spikes[:, 5, 0] = 1.0
    neighbor = model.neighbor_activity(spikes)
    receiving = set((neighbor[0, :, 0] != 0).nonzero().flatten().tolist())
    expected = {5}
    parent = NTU25_PARENT[5]
    expected.add(parent)
    expected.update(index for index, candidate_parent in enumerate(NTU25_PARENT) if candidate_parent == 5)
    assert receiving == expected


def test_adjacency_is_frozen_row_normalized_buffer():
    model = SpikingGraph(num_classes=10, hidden_size=32)
    names = {name for name, _ in model.named_parameters()}
    assert "adjacency" not in names
    buffers = dict(model.named_buffers())
    assert "adjacency" in buffers
    adjacency = buffers["adjacency"]
    torch.testing.assert_close(adjacency.sum(dim=1), torch.ones(25))
    assert torch.all(adjacency >= 0).item()


def test_silence_state_is_deterministic_and_silences_exact_quarter():
    model = SpikingGraph(num_classes=10, hidden_size=32)
    ones = torch.ones(1, 25, 32)
    state = SpikingGraphState(ones.clone(), ones.clone(), ones.clone())
    first = model.silence_state(state, fraction=0.25, seed=19)
    second = model.silence_state(state, fraction=0.25, seed=19)
    torch.testing.assert_close(first.membrane, second.membrane)
    assert int((first.membrane == 0).sum().item()) == 200
    assert torch.all(state.membrane == 1).item()


def test_spiking_graph_is_deterministic_for_fixed_seed_and_under_ceiling():
    torch.manual_seed(23)
    first = SpikingGraph(num_classes=10, hidden_size=32)
    torch.manual_seed(23)
    second = SpikingGraph(num_classes=10, hidden_size=32)
    events = torch.randn(1, 5, 25, 9)
    torch.testing.assert_close(first(events), second(events))
    count = sum(parameter.numel() for parameter in first.parameters() if parameter.requires_grad)
    assert count <= 50_000


def test_spiking_graph_rejects_flat_or_wrong_node_input():
    model = SpikingGraph(num_classes=10, hidden_size=32)
    state = model.initial_state(1, torch.device("cpu"), torch.float32)
    for bad in (torch.zeros(1, 225), torch.zeros(1, 24, 9), torch.zeros(1, 25, 8)):
        try:
            model.step(bad, state)
        except ValueError as exc:
            assert "shape" in str(exc)
        else:
            raise AssertionError("invalid node input must fail")
