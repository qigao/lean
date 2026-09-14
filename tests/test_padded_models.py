"""Model ingestion must distinguish observed missing frames from trailing padding."""
from copy import deepcopy
from dataclasses import replace
import inspect
import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from yolo_flywire.graphs import DirectedGraph
from yolo_flywire.models import GRUClassifier, GraphRecurrentClassifier
from yolo_flywire.pose_batches import PoseBatch, collate_pose_features


def _sequence(count, offset=0.):
    values = np.zeros((count, 121), dtype=np.float64)
    values[:, :34] = np.arange(34)[None, :] / 100 + offset
    values[:, 68:85] = .8
    values[:, 85:102] = 1.
    values[1:, 102:119] = 1.
    values[:, 119] = .9
    values[1:, 120] = .04
    return values


def _batch():
    # Intentionally not length-sorted. Duplicate lengths must keep their order too.
    return collate_pose_features((_sequence(2, .1), _sequence(4, -.2),
                                  _sequence(1, .4), _sequence(2, -.3)))


def _extend(batch, extra=5):
    return PoseBatch(F.pad(batch.features, (0, 0, 0, extra)), batch.lengths.clone(),
                     F.pad(batch.time_mask, (0, extra), value=False))


def _model(kind, input_dim=121):
    torch.manual_seed(203)
    if kind == "gru":
        return GRUClassifier(input_dim, 4, 3).float().cpu()
    graph = DirectedGraph(3, (0, 0, 1, 2), (0, 1, 2, 0), (.2, .4, .3, .5))
    return GraphRecurrentClassifier(input_dim, graph, 2, 3).float().cpu()


@pytest.fixture(params=("gru", "graph"))
def model(request):
    return _model(request.param)


def test_padded_outputs_match_individual_unpadded_runs_in_original_order(model):
    batch = _batch()
    actual = model.forward_padded(batch)
    hidden = model.encode_padded(batch)
    expected = torch.cat([model(batch.features[i:i+1, :count])
                          for i, count in enumerate(batch.lengths.tolist())])
    expected_hidden = torch.cat([model.encode(batch.features[i:i+1, :count])
                                 for i, count in enumerate(batch.lengths.tolist())])
    assert actual.shape == (4, 3) and actual.dtype == torch.float32
    assert actual.device.type == "cpu" and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(hidden, expected_hidden, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual, model.readout(hidden), rtol=0, atol=0)


def test_extra_trailing_padding_changes_neither_hidden_nor_predictions(model):
    batch = _batch()
    longer = _extend(batch, 11)
    for mode in (True, False):
        model.train(mode)
        torch.testing.assert_close(model.encode_padded(batch), model.encode_padded(longer), rtol=0, atol=0)
        torch.testing.assert_close(model.forward_padded(batch), model.forward_padded(longer), rtol=0, atol=0)


def test_batch_companions_and_permutation_do_not_change_a_clip_prediction(model):
    batch = _batch()
    predictions = model.forward_padded(batch)
    permutation = torch.tensor([2, 0, 3, 1])
    shuffled = PoseBatch(batch.features[permutation], batch.lengths[permutation], batch.time_mask[permutation])
    torch.testing.assert_close(model.forward_padded(shuffled), predictions[permutation], rtol=1e-5, atol=1e-6)
    alone = collate_pose_features((_sequence(2, .1),))
    torch.testing.assert_close(model.forward_padded(alone)[0], predictions[0], rtol=1e-5, atol=1e-6)


def test_observed_missing_frames_are_real_recurrent_steps(model):
    # Known nonzero recurrence even with no observed person. No random assertion.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        if isinstance(model, GRUClassifier):
            model.gru.bias_ih_l0[8:].fill_(math.atanh(.5))
        else:
            model.input_projection.bias.fill_(math.atanh(.5))
            model.self_projection.weight.copy_(.5 * torch.eye(2))
        model.readout.weight.fill_(1.)
    one = np.zeros((1, 121), dtype=np.float64)
    two = np.zeros((2, 121), dtype=np.float64)
    two[1, 120] = .2
    batch = collate_pose_features((one, two))
    hidden = model.encode_padded(batch)
    if isinstance(model, GRUClassifier):
        expected = torch.tensor([[.25] * 4, [.375] * 4])
    else:
        expected = torch.tensor([[.5] * 2, [math.tanh(math.atanh(.5) + .25)] * 2])
    torch.testing.assert_close(hidden, expected, rtol=1e-6, atol=1e-7)
    assert not torch.equal(hidden[0], hidden[1])
    torch.testing.assert_close(model.forward_padded(batch), model.readout(expected), rtol=1e-6, atol=1e-7)


def test_backward_matches_unpadded_examples_and_has_zero_padding_gradient(model):
    reference = deepcopy(model)
    batch = _extend(_batch(), 3)
    batch.features.requires_grad_(True)
    targets = torch.tensor([2, 0, 1, 2])  # Labels remain outside observation features.
    loss = F.cross_entropy(model.forward_padded(batch), targets)
    loss.backward()
    expected = torch.cat([reference(batch.features.detach()[i:i+1, :count])
                          for i, count in enumerate(batch.lengths.tolist())])
    F.cross_entropy(expected, targets).backward()
    assert torch.isfinite(loss) and batch.features.grad is not None
    assert torch.count_nonzero(batch.features.grad[~batch.time_mask]) == 0
    assert torch.count_nonzero(batch.features.grad[batch.time_mask]) > 0
    for actual, wanted in zip(model.parameters(), reference.parameters(), strict=True):
        assert actual.grad is not None and torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, wanted.grad, rtol=2e-5, atol=2e-6)


def test_extra_padding_does_not_change_parameter_or_observed_input_gradients(model):
    second = deepcopy(model)
    first_batch, second_batch = _batch(), _extend(_batch())
    for batch, network in ((first_batch, model), (second_batch, second)):
        batch.features.requires_grad_(True)
        network.forward_padded(batch).square().sum().backward()
    torch.testing.assert_close(first_batch.features.grad, second_batch.features.grad[:, :4], rtol=0, atol=0)
    for left, right in zip(model.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


def test_padded_path_does_not_delegate_to_dense_encode_or_mutate_state(model, monkeypatch):
    batch = _batch()
    before = tuple(t.clone() for t in (batch.features, batch.lengths, batch.time_mask))
    parameters = {name: t.clone() for name, t in model.state_dict().items()}
    count = model.parameter_count()
    adjacency = model.adjacency.clone() if isinstance(model, GraphRecurrentClassifier) else None
    def forbidden(*args, **kwargs):
        raise AssertionError("padded input must not use the fixed-length encode path")
    monkeypatch.setattr(model, "encode", forbidden)
    actual = model.forward_padded(batch)
    assert torch.isfinite(actual).all()
    for now, old in zip((batch.features, batch.lengths, batch.time_mask), before, strict=True):
        assert torch.equal(now, old)
    assert model.parameter_count() == count
    assert set(model.state_dict()) == set(parameters)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, parameters[name])
    if adjacency is not None:
        assert torch.equal(model.adjacency, adjacency)
        assert "adjacency" not in model.state_dict()


def test_graph_step_never_receives_padding_and_preserves_original_intervals(monkeypatch):
    model = _model("graph")
    batch = _extend(_batch(), 10)
    original = model._step
    seen = []
    def recording(x, state):
        seen.append(x.detach().clone())
        return original(x, state)
    monkeypatch.setattr(model, "_step", recording)
    model.forward_padded(batch)
    assert [len(x) for x in seen] == [4, 3, 1, 1]
    for time, values in enumerate(seen):
        assert torch.equal(values, batch.features[batch.time_mask[:, time], time])
    assert sum(len(x) for x in seen) == batch.lengths.sum().item()


def test_strided_observation_tensor_is_accepted_without_copying_away_gradients(model):
    batch = _batch()
    storage = torch.zeros((4, 4, 242))
    storage[:, :, ::2] = batch.features
    features = storage[:, :, ::2].detach().requires_grad_(True)
    assert not features.is_contiguous()
    actual = model.forward_padded(replace(batch, features=features))
    torch.testing.assert_close(actual, model.forward_padded(batch), rtol=0, atol=0)
    actual.sum().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0


def test_batch_api_is_explicit_and_has_no_metadata_or_fallback_switches(model):
    for name in ("encode_padded", "forward_padded"):
        method = getattr(type(model), name)
        assert tuple(inspect.signature(method).parameters) == ("self", "batch")


_BAD_CASES = (
    "not_batch", "bare_features", "feature_list", "feature_float64", "feature_sparse", "feature_meta",
    "empty_batch", "empty_time", "wrong_width", "wrong_rank", "nonfinite", "dirty_padding",
    "length_list", "length_float", "length_bool", "length_wrong_shape", "length_zero",
    "length_negative", "length_too_large", "length_meta", "mask_list", "mask_float",
    "mask_wrong_shape", "mask_hole", "mask_stale", "mask_meta",
)


def _bad_batch(case):
    batch = _batch()
    if case == "not_batch":
        return None
    if case == "bare_features":
        return batch.features
    if case.startswith("feature_"):
        changes = {"feature_list": lambda x: x.tolist(), "feature_float64": lambda x: x.double(),
                   "feature_sparse": lambda x: x.to_sparse(), "feature_meta": lambda x: x.to("meta")}
        return replace(batch, features=changes[case](batch.features))
    if case == "empty_batch":
        return PoseBatch(batch.features[:0], batch.lengths[:0], batch.time_mask[:0])
    if case == "empty_time":
        return replace(batch, features=batch.features[:, :0], time_mask=batch.time_mask[:, :0])
    if case == "wrong_width":
        return replace(batch, features=batch.features[:, :, :-1])
    if case == "wrong_rank":
        return replace(batch, features=batch.features[0])
    if case == "nonfinite":
        batch.features[0, 0, 0] = float("nan")
    elif case == "dirty_padding":
        batch.features[0, 3, 0] = 1.
    elif case.startswith("length_"):
        changes = {"length_list": lambda x: x.tolist(), "length_float": lambda x: x.float(),
                   "length_bool": lambda x: x.bool(), "length_wrong_shape": lambda x: x[:, None],
                   "length_meta": lambda x: x.to("meta")}
        if case in changes:
            return replace(batch, lengths=changes[case](batch.lengths))
        batch.lengths[0] = {"length_zero": 0, "length_negative": -1, "length_too_large": 5}[case]
    elif case.startswith("mask_"):
        changes = {"mask_list": lambda x: x.tolist(), "mask_float": lambda x: x.float(),
                   "mask_wrong_shape": lambda x: x[:, :-1], "mask_meta": lambda x: x.to("meta")}
        if case in changes:
            return replace(batch, time_mask=changes[case](batch.time_mask))
        if case == "mask_hole":
            batch.time_mask[0] = torch.tensor([True, False, True, False])
        else:
            batch.time_mask[0, 1] = False
    return batch


@pytest.mark.parametrize("case", _BAD_CASES)
def test_malformed_mutable_batch_is_rejected_before_recurrent_computation(model, case, monkeypatch):
    method = model.forward_padded
    def forbidden(*args, **kwargs):
        raise AssertionError("malformed batch reached recurrence")
    if isinstance(model, GRUClassifier):
        monkeypatch.setattr(model.gru, "forward", forbidden)
    else:
        monkeypatch.setattr(model, "_step", forbidden)
    with pytest.raises(ValueError):
        method(_bad_batch(case))


@pytest.mark.parametrize("kind", ("gru", "graph"))
def test_wrong_model_input_width_is_not_silently_projected_or_truncated(kind):
    model = _model(kind, input_dim=102)
    with pytest.raises(ValueError):
        model.forward_padded(_batch())
