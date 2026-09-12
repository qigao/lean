"""Development training contracts; generated feature fixtures, not recognition evidence."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib
import inspect
import json
import random

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from yolo_flywire.eval import MetricBundle
from yolo_flywire.graphs import DirectedGraph
from yolo_flywire.models import GRUClassifier, GraphRecurrentClassifier
from yolo_flywire.pose_batches import PoseBatch, collate_pose_features
from yolo_flywire.train import TrainConfig, _reset_parameters


def _api():
    return importlib.import_module("yolo_flywire.pose_training")


def _model(kind):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        if kind == "gru":
            return GRUClassifier(121, 4, 3).float().cpu()
        return GraphRecurrentClassifier(
            121, DirectedGraph(3, (0, 0, 1, 2), (0, 1, 2, 0), (.2, .3, .4, .5)), 2, 3,
        ).float().cpu()


def _partition(split, extra=0):
    clips = []
    for row, length in enumerate((3, 1, 4, 2, 1, 3, 2)):
        x = np.zeros((length, 121), dtype=np.float64)
        x[:, 0] = row + 1  # Observation marker for row-alignment tests, not a label.
        x[:, 68:102] = 1
        x[1:, 102:119] = 1
        x[:, 119] = 1
        x[1:, 120] = .04
        clips.append(x)
    batch = collate_pose_features(tuple(clips))
    batch = PoseBatch(F.pad(batch.features, (0, 0, 0, extra)), batch.lengths,
                      F.pad(batch.time_mask, (0, extra), value=False))
    return _api().PosePartition(
        observations=batch, targets=torch.tensor([2, 0, 1, 2, 1, 0, 2], dtype=torch.int64),
        classes=("sit", "wave", "point"), sample_ids=tuple(f"{split}-{i}" for i in range(7)),
        subjects=(1,) * 7 if split == "train" else (14,) * 7, split=split,
    )


def _config(**changes):
    return replace(TrainConfig(seed=7, epochs=3, lr=.01, batch_size=3, parameter_ceiling=10000), **changes)


@pytest.mark.parametrize("kind", ("gru", "graph"))
def test_actual_training_is_repeatable_padding_invariant_and_budgeted(kind, monkeypatch):
    api = _api()
    results = []
    for extra in (0, 0, 9):
        model, train, validation = _model(kind), _partition("train", extra), _partition("validation", extra)
        before = tuple(t.clone() for t in (train.observations.features, train.targets,
                                          validation.observations.features, validation.targets))
        adjacency = model.adjacency.clone() if kind == "graph" else None
        parameter_ids = tuple(id(p) for p in model.parameters())
        def forbidden(*args, **kwargs):
            raise AssertionError("development training must not call the fixed-length model path")
        monkeypatch.setattr(model, "forward", forbidden)
        run = api.train_padded_model(model, train, validation, _config())
        assert run.model is model and run.optimizer_steps == 9
        assert 1 <= run.best_epoch <= 3 and len(run.epoch_order_hashes) == 3
        assert tuple(id(p) for p in model.parameters()) == parameter_ids
        assert not model.training and all(p.grad is None for p in model.parameters())
        assert api.evaluate_padded(model, validation, batch_size=3).macro_f1 == run.best_validation_macro_f1
        for original, tensor in zip(before, (train.observations.features, train.targets,
                                             validation.observations.features, validation.targets), strict=True):
            assert torch.equal(original, tensor)
        if adjacency is not None:
            assert torch.equal(model.adjacency, adjacency)
        generator = torch.Generator(device="cpu").manual_seed(7)
        for digest in run.epoch_order_hashes:
            order = torch.randperm(7, generator=generator).tolist()
            encoded = json.dumps([train.sample_ids[i] for i in order], separators=(",", ":")).encode()
            assert digest == hashlib.sha256(encoded).hexdigest()
        results.append(run)
    assert len({r.state_hash for r in results}) == 1
    assert len({r.best_epoch for r in results}) == 1
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        initial = _model(kind)
        _reset_parameters(initial)
    assert any(not torch.equal(v, initial.state_dict()[k]) for k, v in results[0].model.state_dict().items())


@pytest.mark.parametrize("kind", ("gru", "graph"))
def test_every_minibatch_keeps_targets_lengths_and_observed_rows_aligned(kind, monkeypatch):
    api = _api()
    train, validation, model = _partition("train"), _partition("validation"), _model(kind)
    original_forward, original_loss = model.forward_padded, F.cross_entropy
    observed, updates = [], []
    def forward(batch):
        if model.training and torch.is_grad_enabled():
            observed[:] = [int(v) - 1 for v in batch.features[:, 0, 0]]
            expected = train.observations.lengths[observed]
            assert torch.equal(batch.lengths, expected)
            assert batch.time_mask.sum(dim=1).tolist() == expected.tolist()
        return original_forward(batch)
    def loss(logits, targets, *args, **kwargs):
        assert torch.equal(targets, train.targets[observed])
        updates.append(tuple(observed))
        return original_loss(logits, targets, *args, **kwargs)
    monkeypatch.setattr(model, "forward_padded", forward)
    monkeypatch.setattr(F, "cross_entropy", loss)
    api.train_padded_model(model, train, validation, _config())
    assert [len(i) for i in updates] == [3, 3, 1] * 3
    for start in (0, 3, 6):
        assert sorted(i for batch in updates[start:start+3] for i in batch) == list(range(7))


@pytest.mark.parametrize("kind", ("gru", "graph"))
def test_validation_metrics_use_whole_partition_and_fixed_class_vocabulary(kind):
    api, model = _api(), _model(kind)
    part = _partition("validation")
    obs = part.observations
    part = replace(part, observations=PoseBatch(obs.features[:3], obs.lengths[:3], obs.time_mask[:3]),
                   targets=torch.tensor([0, 0, 1]), sample_ids=part.sample_ids[:3], subjects=part.subjects[:3])
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    # All predictions are zero. The absent third class remains in the denominator.
    for size in (1, 2, 9):
        score = api.evaluate_padded(model, part, batch_size=size)
        assert score.macro_f1 == pytest.approx(.8 / 3)
        assert score.balanced_accuracy == pytest.approx(1 / 3)


@pytest.mark.parametrize("kind", ("gru", "graph"))
def test_evaluation_has_no_gradients_and_restores_each_module_mode_even_on_error(kind, monkeypatch):
    api, model, part = _api(), _model(kind), _partition("validation")
    model.train()
    model.readout.eval()
    modes = tuple(m.training for m in model.modules())
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    original = model.forward_padded
    def checked(batch):
        assert not torch.is_grad_enabled() and all(not m.training for m in model.modules())
        return original(batch)
    monkeypatch.setattr(model, "forward_padded", checked)
    api.evaluate_padded(model, part, batch_size=2)
    assert tuple(m.training for m in model.modules()) == modes
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
    def broken(batch):
        raise RuntimeError("fixture inference failure")
    monkeypatch.setattr(model, "forward_padded", broken)
    with pytest.raises(RuntimeError, match="fixture inference"):
        api.evaluate_padded(model, part, batch_size=2)
    assert tuple(m.training for m in model.modules()) == modes


@pytest.mark.parametrize("kind", ("gru", "graph"))
def test_validation_alone_selects_earliest_best_checkpoint_without_shortening_budget(kind, monkeypatch):
    api, states = _api(), []
    scores = iter((.75, .2, .75))
    def validation_score(model, partition, *, batch_size):
        assert partition.split == "validation"
        assert batch_size == 3
        states.append(deepcopy(model.state_dict()))
        return MetricBundle(next(scores), 0.)
    monkeypatch.setattr(api, "evaluate_padded", validation_score)
    run = api.train_padded_model(_model(kind), _partition("train"), _partition("validation"), _config())
    assert run.best_epoch == 1 and run.best_validation_macro_f1 == .75 and run.optimizer_steps == 9
    for key, value in states[0].items():
        assert torch.equal(run.model.state_dict()[key], value)
    assert any(not torch.equal(states[0][key], states[-1][key]) for key in states[0])


@pytest.mark.parametrize("kind", ("gru", "graph"))
def test_changing_validation_labels_does_not_change_optimizer_trajectory(kind, monkeypatch):
    api, trajectories = _api(), [[], []]
    evaluate = api.evaluate_padded
    for i in (0, 1):
        part = _partition("validation")
        if i:
            part = replace(part, targets=(part.targets + 1) % 3)
        def record(model, validation, *, batch_size):
            trajectories[i].append(deepcopy(model.state_dict()))
            return evaluate(model, validation, batch_size=batch_size)
        monkeypatch.setattr(api, "evaluate_padded", record)
        api.train_padded_model(_model(kind), _partition("train"), part, _config())
    for first, second in zip(*trajectories, strict=True):
        assert all(torch.equal(first[k], second[k]) for k in first)


def test_training_restores_caller_random_states_and_determinism_flags():
    api, model = _api(), _model("gru")
    train, validation = _partition("train"), _partition("validation")
    random.seed(13); np.random.seed(29); torch.manual_seed(31)
    states = random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()
    flags = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    api.train_padded_model(model, train, validation, _config())
    assert random.getstate() == states[0]
    actual = np.random.get_state()
    assert actual[0] == states[1][0] and np.array_equal(actual[1], states[1][1]) and actual[2:] == states[1][2:]
    assert torch.equal(torch.random.get_rng_state(), states[2])
    assert flags == (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())


_BAD_PARTITIONS = ("test", "wrong_role", "overlap_id", "overlap_subject", "classes", "duplicate_class",
                   "target_length", "target_float", "target_range", "target_meta", "duplicate_id",
                   "subjects_length", "subject_bool", "stale_mask", "dirty_padding", "runtime")


@pytest.mark.parametrize("case", _BAD_PARTITIONS)
def test_invalid_partition_pair_rejected_before_reset_or_inference(case, monkeypatch):
    api, model = _api(), _model("gru")
    train, validation = _partition("train"), _partition("validation")
    if case == "test": validation = replace(validation, split="final_test")
    elif case == "wrong_role": train = replace(train, split="validation")
    elif case == "overlap_id": validation = replace(validation, sample_ids=train.sample_ids)
    elif case == "overlap_subject": validation = replace(validation, subjects=train.subjects)
    elif case == "classes": validation = replace(validation, classes=validation.classes[::-1])
    elif case == "duplicate_class": validation = replace(validation, classes=("sit", "sit", "wave"))
    elif case == "target_length": validation = replace(validation, targets=validation.targets[:1])
    elif case == "target_float": validation = replace(validation, targets=validation.targets.float())
    elif case == "target_range": validation.targets[0] = 3
    elif case == "target_meta": validation = replace(validation, targets=validation.targets.to("meta"))
    elif case == "duplicate_id": validation = replace(validation, sample_ids=("same",) * 7)
    elif case == "subjects_length": validation = replace(validation, subjects=(14,))
    elif case == "subject_bool": validation = replace(validation, subjects=(True,) * 7)
    elif case == "stale_mask": validation.observations.lengths[0] = 1
    elif case == "dirty_padding": validation.observations.features[1, 2, 0] = 1
    else: model.readout.double()
    before = deepcopy(model.state_dict())
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid input reached model reset or inference")
    monkeypatch.setattr(api, "_reset_parameters", forbidden)
    monkeypatch.setattr(model, "forward_padded", forbidden)
    with pytest.raises(ValueError):
        api.train_padded_model(model, train, validation, _config())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


@pytest.mark.parametrize("changes", ({"epochs": 0}, {"epochs": True}, {"batch_size": 0},
    {"seed": -1}, {"seed": 2**32}, {"lr": 0.}, {"lr": float("nan")}, {"lr": float("inf")},
    {"lr": True}, {"parameter_ceiling": 1}))
def test_invalid_or_over_budget_config_never_resets_model(changes, monkeypatch):
    api, model = _api(), _model("graph")
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid config reached reset")
    monkeypatch.setattr(api, "_reset_parameters", forbidden)
    with pytest.raises(ValueError):
        api.train_padded_model(model, _partition("train"), _partition("validation"), _config(**changes))


@pytest.mark.parametrize("kind", ("gru", "graph"))
def test_nonfinite_logits_are_errors_not_a_valid_zero_score(kind, monkeypatch):
    api, model = _api(), _model(kind)
    def invalid(batch):
        return torch.full((len(batch.lengths), 3), float("nan"))
    monkeypatch.setattr(model, "forward_padded", invalid)
    with pytest.raises(ValueError, match="finite"):
        api.evaluate_padded(model, _partition("validation"), batch_size=2)
    with pytest.raises(ValueError, match="finite"):
        api.train_padded_model(model, _partition("train"), _partition("validation"), _config())


def test_development_api_has_no_final_test_or_dense_fallback_switches():
    api = _api()
    assert tuple(inspect.signature(api.train_padded_model).parameters) == ("model", "train", "validation", "config")
    assert tuple(inspect.signature(api.evaluate_padded).parameters) == ("model", "validation", "batch_size")
    for split in ("train", "final_test"):
        with pytest.raises(ValueError):
            api.evaluate_padded(_model("gru"), _partition(split), batch_size=2)
