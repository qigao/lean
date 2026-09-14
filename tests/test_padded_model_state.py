"""Padded model runtime boundary and optimizer/topology regression checks."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from yolo_flywire.graphs import DirectedGraph
from yolo_flywire.models import GRUClassifier, GraphRecurrentClassifier
from yolo_flywire.pose_batches import PoseBatch, collate_pose_features


def _model(kind):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        if kind == "gru":
            return GRUClassifier(121, 4, 3).float().cpu()
        graph = DirectedGraph(3, (0, 0, 1, 2), (0, 1, 2, 0), (.2, .5, .8, .3))
        return GraphRecurrentClassifier(121, graph, 2, 3).float().cpu()


def _batch(extra=0):
    # These are observed missing-person frames with genuine intervals.
    clips = []
    for count in (2, 4, 1):
        clip = np.zeros((count, 121), dtype=np.float64)
        clip[1:, 120] = .04
        clips.append(clip)
    batch = collate_pose_features(tuple(clips))
    return PoseBatch(F.pad(batch.features, (0, 0, 0, extra)), batch.lengths.clone(),
                     F.pad(batch.time_mask, (0, extra), value=False))


@pytest.mark.parametrize("kind", ("gru", "graph"))
@pytest.mark.parametrize("case", ("double", "half", "meta", "readout_double", "readout_meta"))
def test_model_runtime_mismatch_rejected_before_any_recurrent_work(kind, case, monkeypatch):
    model = _model(kind)
    batch = _batch()
    if case == "double":
        model.double()
    elif case == "half":
        model.half()
    elif case == "meta":
        model.to("meta")
    elif case == "readout_double":
        model.readout.double()
    else:
        model.readout.to("meta")
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported model runtime reached recurrence")
    if kind == "gru":
        monkeypatch.setattr(model.gru, "forward", forbidden)
    else:
        monkeypatch.setattr(model, "_step", forbidden)
    # Both interfaces require a consistent complete model, including its readout.
    for name in ("encode_padded", "forward_padded"):
        with pytest.raises(ValueError, match="model"):
            getattr(model, name)(batch)


@pytest.mark.parametrize("case", ("adjacency_double", "adjacency_meta", "message_half"))
def test_graph_runtime_gate_covers_nonpersistent_buffers_and_nested_parameters(case, monkeypatch):
    model = _model("graph")
    batch = _batch()
    if case == "adjacency_double":
        model.adjacency = model.adjacency.double()
    elif case == "adjacency_meta":
        model.adjacency = model.adjacency.to("meta")
    else:
        model.message_projection.half()
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported graph state reached recurrence")
    monkeypatch.setattr(model, "_step", forbidden)
    for name in ("encode_padded", "forward_padded"):
        with pytest.raises(ValueError, match="model"):
            getattr(model, name)(_batch())


@pytest.mark.parametrize("kind", ("gru", "graph"))
def test_runtime_validation_is_repeated_after_a_valid_call(kind, monkeypatch):
    model = _model(kind)
    batch = _batch()
    assert torch.isfinite(model.forward_padded(batch)).all()
    model.readout.double()
    def forbidden(*args, **kwargs):
        raise AssertionError("model mutation bypassed the per-call runtime gate")
    if kind == "gru":
        monkeypatch.setattr(model.gru, "forward", forbidden)
    else:
        monkeypatch.setattr(model, "_step", forbidden)
    with pytest.raises(ValueError, match="model"):
        model.forward_padded(batch)


@pytest.mark.parametrize("kind", ("gru", "graph"))
@pytest.mark.parametrize("optimizer_class", (torch.optim.SGD, torch.optim.Adam))
def test_padding_cannot_change_optimizer_updates_or_parameter_identity(kind, optimizer_class):
    models = (_model(kind), _model(kind))
    before = deepcopy(models[0].state_dict())
    for model, batch in zip(models, (_batch(), _batch(extra=8)), strict=True):
        identities = tuple(id(parameter) for parameter in model.parameters())
        optimizer = optimizer_class(model.parameters(), lr=.01)
        loss = F.cross_entropy(model.forward_padded(batch), torch.tensor([0, 1, 2]))
        loss.backward()
        optimizer.step()
        assert identities == tuple(id(parameter) for parameter in model.parameters())
    assert models[0].parameter_count() == models[1].parameter_count()
    for name, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[name], rtol=0, atol=0)
    assert any(not torch.equal(value, before[name]) for name, value in models[0].state_dict().items())


def test_loading_shared_parameters_never_replaces_graph_experimental_condition():
    first = _model("graph")
    graph = DirectedGraph(3, (0, 0, 1, 2), (0, 2, 0, 1), (.2, .5, .8, .3))
    second = GraphRecurrentClassifier(121, graph, 2, 3).float().cpu()
    adjacency = second.adjacency.clone()
    assert "adjacency" not in first.state_dict()
    second.load_state_dict(first.state_dict())
    assert torch.equal(second.adjacency, adjacency)
    assert not torch.equal(first.adjacency, second.adjacency)
    expected = second.encode_padded(_batch()).detach().clone()
    second.load_state_dict(deepcopy(first.state_dict()))
    torch.testing.assert_close(second.encode_padded(_batch(extra=5)), expected, rtol=0, atol=0)
    assert not torch.allclose(first.encode_padded(_batch()), expected)
