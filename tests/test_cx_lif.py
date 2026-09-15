from __future__ import annotations

import torch

from yolo_flywire.cx_artifact import CxArtifact, CxNode
from yolo_flywire.graphs import DirectedGraph
from yolo_flywire.models.cx_lif import CxDynamics, CxLifClassifier


TEST_DYNAMICS = CxDynamics(
    tau_membrane=20.0,
    synaptic_decay=0.8,
    refractory_steps=2,
    threshold=0.5,
    reset=0.0,
    recurrent_delay_steps=1,
    recurrent_gain=0.1,
    magnitude_policy="log1p",
)


def _artifact(*, alternate: bool = False) -> CxArtifact:
    nodes = (
        CxNode(10, "ER1", "ER", "input"),
        CxNode(11, "PFNa", "PFN", "input"),
        CxNode(20, "EPG", "EPG", "core"),
        CxNode(21, "PEN_a", "PEN", "core"),
        CxNode(30, "PFL2", "PFL", "output"),
        CxNode(31, "PFL3", "PFL", "output"),
    )
    if alternate:
        edges = (
            (0, 3, 8.0, +1),
            (1, 2, 7.0, +1),
            (2, 5, 9.0, +1),
            (3, 4, 6.0, +1),
            (4, 2, 5.0, -1),
            (5, 3, 5.0, -1),
        )
    else:
        edges = (
            (0, 2, 8.0, +1),
            (1, 3, 7.0, +1),
            (2, 4, 9.0, +1),
            (3, 5, 6.0, +1),
            (4, 3, 5.0, -1),
            (5, 2, 5.0, -1),
        )
    graph = DirectedGraph(
        num_nodes=len(nodes),
        src=tuple(src for src, _, _, _ in edges),
        dst=tuple(dst for _, dst, _, _ in edges),
        weight=tuple(weight for _, _, weight, _ in edges),
    )
    return CxArtifact(
        nodes=nodes,
        graph=graph,
        input_indices=(0, 1),
        core_indices=(2, 3),
        output_indices=(4, 5),
        edge_signs=tuple(sign for _, _, _, sign in edges),
        source_hashes=(("consolidated_cell_types", "a" * 64), ("connections_princeton", "b" * 64)),
        matched_primary_types=tuple(node.primary_type for node in nodes),
        fingerprint=("d" if alternate else "c") * 64,
    )


def test_input_projection_targets_only_biological_inputs_and_readout_only_outputs():
    artifact = _artifact()
    model = CxLifClassifier(12, artifact, 4, dynamics=TEST_DYNAMICS)
    assert model.input_projection.out_features == len(artifact.input_indices)
    assert model.input_projection.bias is None
    assert model.readout.in_features == len(artifact.output_indices)
    assert tuple(model.input_indices.tolist()) == artifact.input_indices
    assert tuple(model.output_indices.tolist()) == artifact.output_indices


def test_recurrent_weights_are_fixed_buffers_not_trainable_parameters():
    model = CxLifClassifier(12, _artifact(), 4, dynamics=TEST_DYNAMICS)
    parameter_names = {name for name, _ in model.named_parameters()}
    assert parameter_names == {
        "input_projection.weight",
        "readout.weight",
        "readout.bias",
    }
    buffer_names = {name for name, _ in model.named_buffers()}
    assert {"recurrent_source", "recurrent_target", "recurrent_weight"}.issubset(buffer_names)


def test_input_injection_is_zero_outside_biological_input_nodes():
    artifact = _artifact()
    model = CxLifClassifier(3, artifact, 2, dynamics=TEST_DYNAMICS)
    with torch.no_grad():
        model.input_projection.weight.fill_(1.0)
    injected = model.inject(torch.ones(2, 3))
    assert injected.shape == (2, artifact.graph.num_nodes)
    assert torch.count_nonzero(injected[:, list(artifact.core_indices)]).item() == 0
    assert torch.count_nonzero(injected[:, list(artifact.output_indices)]).item() == 0
    assert torch.all(injected[:, list(artifact.input_indices)] > 0)


def test_recurrent_weight_sign_and_magnitude_come_from_artifact():
    artifact = _artifact()
    model = CxLifClassifier(3, artifact, 2, dynamics=TEST_DYNAMICS)
    expected = torch.log1p(torch.tensor(artifact.graph.weight))
    expected = expected * torch.tensor(artifact.edge_signs, dtype=torch.float32) * TEST_DYNAMICS.recurrent_gain
    torch.testing.assert_close(model.recurrent_weight.cpu(), expected)


def test_same_trainable_parameters_but_different_topology_change_spike_trajectory():
    real = CxLifClassifier(2, _artifact(), 2, dynamics=TEST_DYNAMICS)
    alternate = CxLifClassifier(2, _artifact(alternate=True), 2, dynamics=TEST_DYNAMICS)
    with torch.no_grad():
        real.input_projection.weight.fill_(1.0)
        real.readout.weight.copy_(torch.eye(2))
        real.readout.bias.zero_()
        alternate.load_state_dict(real.state_dict())

    events = torch.ones(1, 8, 2)
    real_latent, real_stats = real.encode(events)
    alt_latent, alt_stats = alternate.encode(events)
    assert not torch.allclose(real_latent, alt_latent)
    assert real_stats.steps == alt_stats.steps == 8


def test_zero_input_advances_recurrent_state_without_external_injection():
    model = CxLifClassifier(2, _artifact(), 2, dynamics=TEST_DYNAMICS)
    state = model.initial_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
    state = state._replace(
        spikes=torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
        membrane=torch.zeros_like(state.membrane),
        synaptic=torch.zeros_like(state.synaptic),
    )
    next_state = model.step(torch.zeros(1, 2), state)
    assert torch.count_nonzero(next_state.synaptic).item() > 0
    assert torch.count_nonzero(model.inject(torch.zeros(1, 2))).item() == 0


def test_encode_reports_spike_and_synaptic_event_accounting():
    model = CxLifClassifier(2, _artifact(), 2, dynamics=TEST_DYNAMICS)
    events = torch.ones(3, 5, 2)
    _, stats = model.encode(events)
    assert stats.steps == 5
    assert stats.batch_size == 3
    assert stats.total_spikes >= 0
    assert stats.recurrent_synaptic_events >= 0
    assert len(stats.active_neurons_per_step) == 5
    assert stats.peak_state_bytes > 0


def test_forward_returns_class_logits():
    model = CxLifClassifier(2, _artifact(), 3, dynamics=TEST_DYNAMICS)
    logits = model(torch.zeros(4, 6, 2))
    assert logits.shape == (4, 3)
