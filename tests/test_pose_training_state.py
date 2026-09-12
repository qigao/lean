"""Reject invalid experiment state before reset or saturated recurrent computation."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from test_pose_training import _api, _config, _model, _partition


_STATE_CASES = (
    ("gru", "bias", float("inf")),
    ("graph", "bias", float("-inf")),
    ("graph", "adjacency", float("nan")),
    ("gru", "readout", float("nan")),
)


def _corrupt(model, kind, field, value):
    with torch.no_grad():
        if field == "adjacency":
            model.adjacency[0, 0] = value
        elif field == "readout":
            model.readout.bias[0] = value
        elif kind == "gru":
            model.gru.bias_ih_l0.fill_(value)
        else:
            model.input_projection.bias.fill_(value)


@pytest.mark.parametrize("kind,field,value", _STATE_CASES)
def test_nonfinite_state_is_rechecked_before_validation_recurrence(kind, field, value, monkeypatch):
    api, model, part = _api(), _model(kind), _partition("validation")
    # A prior successful call cannot cache trust in mutable model/buffer state.
    api.evaluate_padded(model, part, batch_size=3)
    _corrupt(model, kind, field, value)
    original, calls = model.forward_padded, []
    def recording(batch):
        calls.append(True)
        return original(batch)
    monkeypatch.setattr(model, "forward_padded", recording)
    with pytest.raises(ValueError, match="finite"):
        api.evaluate_padded(model, part, batch_size=3)
    assert calls == [], "invalid state must be rejected before a recurrent call"


@pytest.mark.parametrize("kind,field,value", _STATE_CASES)
def test_nonfinite_state_is_not_silently_repaired_by_training_reset(kind, field, value, monkeypatch):
    api, model = _api(), _model(kind)
    train, validation = _partition("train"), _partition("validation")
    _corrupt(model, kind, field, value)
    before = deepcopy(model.state_dict())
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid state reached reset or recurrence")
    monkeypatch.setattr(api, "_reset_parameters", forbidden)
    monkeypatch.setattr(model, "forward_padded", forbidden)
    with pytest.raises(ValueError, match="finite"):
        api.train_padded_model(model, train, validation, _config())
    for name, tensor in before.items():
        torch.testing.assert_close(model.state_dict()[name], tensor, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("kind", ("gru", "graph"))
def test_partial_parameter_freeze_cannot_silently_change_training_condition(kind, monkeypatch):
    api, model = _api(), _model(kind)
    next(model.parameters()).requires_grad_(False)
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported partial freeze reached parameter reset")
    monkeypatch.setattr(api, "_reset_parameters", forbidden)
    with pytest.raises(ValueError, match="trainable"):
        api.train_padded_model(model, _partition("train"), _partition("validation"), _config())


@pytest.mark.parametrize("kind", ("gru", "graph"))
def test_validation_accepts_finite_frozen_models_without_changing_flags(kind):
    api, model, validation = _api(), _model(kind), _partition("validation")
    expected = api.evaluate_padded(model, validation, batch_size=3)
    model.requires_grad_(False)
    assert api.evaluate_padded(model, validation, batch_size=3) == expected
    assert all(not parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize("stage", ("training", "validation"))
def test_failed_training_restores_caller_randomness_and_flags(stage, monkeypatch):
    api, model = _api(), _model("gru")
    train, validation = _partition("train"), _partition("validation")
    outer = random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()
    outer_flags = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        random.seed(17); np.random.seed(29); torch.manual_seed(41)
        torch.use_deterministic_algorithms(False, warn_only=True)
        expected = random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()
        def broken(*args, **kwargs):
            random.random(); np.random.random(); torch.rand(1)
            raise RuntimeError("deliberate compute failure")
        if stage == "training":
            monkeypatch.setattr(model, "forward_padded", broken)
        else:
            monkeypatch.setattr(api, "evaluate_padded", broken)
        with pytest.raises(RuntimeError, match="deliberate compute failure"):
            api.train_padded_model(model, train, validation, _config())
        assert random.getstate() == expected[0]
        actual = np.random.get_state()
        assert actual[0] == expected[1][0] and np.array_equal(actual[1], expected[1][1]) and actual[2:] == expected[1][2:]
        assert torch.equal(torch.random.get_rng_state(), expected[2])
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        random.setstate(outer[0]); np.random.set_state(outer[1]); torch.random.set_rng_state(outer[2])
        torch.use_deterministic_algorithms(outer_flags[0], warn_only=outer_flags[1])
