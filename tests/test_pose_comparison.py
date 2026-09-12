"""Four-arm development orchestration; generated inputs, not recognition evidence."""
from dataclasses import replace
import importlib
import json
import random

import numpy as np
import pytest
import torch

from test_pose_development import extracted, _options, _prepare
from integration.pose_comparison_fixture import graph_inputs, sha, write_json

FAMILIES = ("gru", "random_graph", "rewired", "flywire")


def _api():
    return importlib.import_module("yolo_flywire.pose_comparison")


def _spec(**changes):
    values = dict(seeds=(7, 11), epochs=2, lr=.01, batch_size=4, max_updates=6,
                  parameter_ceiling=10000, gru_hidden_dim=4, graph_node_dim=2)
    values.update(changes)
    return _api().PoseComparisonSpec(**values)


@pytest.fixture
def comparison_case(extracted, tmp_path):
    prepared = _prepare(extracted)
    options = {**_options(extracted), **graph_inputs(tmp_path / "graphs"),
               "expected_binding_sha256": prepared.binding_sha256}
    return extracted[1], options, prepared


def _run(case, **changes):
    bundle, options, _ = case
    return _api().run_pose_comparison(bundle, **{**options, "config": _spec(), **changes})


def test_all_four_arms_execute_actual_equal_budget_and_remain_development_only(comparison_case, monkeypatch):
    api = _api()
    actual, observed = api.train_padded_model, []
    def train(model, training, validation, config):
        assert training.split == "train" and validation.split == "validation"
        assert training.classes == validation.classes
        assert all("P003" not in sid for part in (training, validation) for sid in part.sample_ids)
        run = actual(model, training, validation, config)
        observed.append((run, config))
        return run
    monkeypatch.setattr(api, "train_padded_model", train)
    report = _run(comparison_case)
    assert len(observed) == 8
    assert report["evidence_scope"] == "development_validation_only"
    assert report["final_test_evaluated"] is False and report["topology_claim_evaluated"] is False
    assert report["prepared_binding_sha256"] == comparison_case[2].binding_sha256
    assert report["topology_protocol_sha256"] == comparison_case[1]["expected_topology_sha256"]
    assert report["control_bundle_sha256"] == comparison_case[1]["expected_controls_sha256"]
    rows = report["arms"]
    assert [(r["seed"], r["family"]) for r in rows] == [(s, f) for s in (7, 11) for f in FAMILIES]
    for row, (run, config) in zip(rows, observed, strict=True):
        assert row["optimizer_steps"] == run.optimizer_steps == 6
        assert row["state_hash"] == run.state_hash
        assert row["best_epoch"] == run.best_epoch
        assert row["validation_metrics"]["macro_f1"] == run.best_validation_macro_f1
        assert row["epoch_order_hashes"] == list(run.epoch_order_hashes)
        assert row["input_binding_sha256"] == report["prepared_binding_sha256"]
        assert row["parameter_count"] == sum(p.numel() for p in run.model.parameters()) <= 10000
        assert config.seed == row["seed"] and (config.epochs, config.batch_size, config.lr) == (2, 4, .01)
        if row["family"] == "gru":
            assert row["topology_fingerprint"] == "none" and row["adjacency_sha256"] is None
        else:
            assert len(row["topology_fingerprint"]) == len(row["adjacency_sha256"]) == 64
    protocol = json.loads(comparison_case[1]["topology_protocol"].read_text())
    for seed in (7, 11):
        arms = {r["family"]: r for r in rows if r["seed"] == seed}
        assert len({tuple(r["epoch_order_hashes"]) for r in arms.values()}) == 1
        assert len({arms[f]["parameter_count"] for f in FAMILIES[1:]}) == 1
        assert arms["flywire"]["topology_fingerprint"] == protocol["selected_graph_fingerprint"]
        assert arms["rewired"]["topology_fingerprint"] == protocol["rewired_graph_fingerprints"][str(seed)]
        paired = next(r for r in report["paired_validation"] if r["seed"] == seed)
        assert paired["difference"] == arms["flywire"]["validation_metrics"]["macro_f1"] - arms["rewired"]["validation_metrics"]["macro_f1"]
    assert "pose_comparison.py" in report["execution"]["source_sha256"]
    assert "models/graph_rnn.py" in report["execution"]["source_sha256"]
    json.dumps(report, allow_nan=False)
    comparison_case[2].verify()


def test_repeatable_with_caller_rng_and_defaults_preserved(comparison_case):
    _api()
    random.seed(453); np.random.seed(67); torch.manual_seed(93)
    states = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    flags = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    first, second = _run(comparison_case), _run(comparison_case)
    assert first == second
    assert random.getstate() == states[0]
    after = np.random.get_state()
    assert after[0] == states[1][0] and after[2:] == states[1][2:]
    np.testing.assert_array_equal(after[1], states[1][1])
    assert torch.equal(torch.get_rng_state(), states[2])
    assert flags == (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())


@pytest.mark.parametrize("key", ("expected_binding_sha256", "expected_manifest_sha256", "expected_encoder_hash",
                                 "expected_topology_sha256", "expected_controls_sha256"))
def test_independent_pins_reject_before_any_training(comparison_case, monkeypatch, key):
    api = _api()
    def forbidden(*args, **kwargs):
        raise AssertionError("mismatched external pin reached optimizer")
    monkeypatch.setattr(api, "train_padded_model", forbidden)
    with pytest.raises(ValueError):
        _run(comparison_case, **{key: "0" * 64})


@pytest.mark.parametrize("field,value", (("seeds", (7, 7)), ("seeds", (True,)), ("seeds", (-1,)),
    ("seeds", (7,)), ("epochs", 0), ("batch_size", True), ("lr", float("nan")),
    ("max_updates", 5), ("batch_size", 5), ("parameter_ceiling", 1),
    ("gru_hidden_dim", 10000), ("graph_node_dim", 10000)))
def test_invalid_or_unmatched_execution_budget_rejected_before_training(comparison_case, monkeypatch, field, value):
    api = _api()
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid configuration reached optimization")
    monkeypatch.setattr(api, "train_padded_model", forbidden)
    with pytest.raises(ValueError):
        _run(comparison_case, config=_spec(**{field: value}))


@pytest.mark.parametrize("asset", ("connectivity", "controls", "topology_protocol"))
def test_changed_graph_asset_bytes_rejected_before_training(comparison_case, monkeypatch, asset):
    api = _api()
    with comparison_case[1][asset].open("ab") as handle:
        handle.write(b" ")
    def forbidden(*args, **kwargs):
        raise AssertionError("changed graph source reached optimization")
    monkeypatch.setattr(api, "train_padded_model", forbidden)
    with pytest.raises(ValueError):
        _run(comparison_case)


@pytest.mark.parametrize("problem", ("missing-seed", "fingerprint", "weight", "budget", "stale-identity"))
def test_repinned_controls_still_require_frozen_fingerprints_and_invariants(comparison_case, monkeypatch, problem):
    api = _api()
    path = comparison_case[1]["controls"]
    record = json.loads(path.read_text())
    control = record["controls"]["7"]
    if problem == "missing-seed": del record["controls"]["11"]
    elif problem == "fingerprint": control["fingerprint"] = "0" * 64
    elif problem == "weight": control["graph"]["weight"][0] += 1
    elif problem == "budget": control["successful_swaps"] -= 1
    else: record["identity"]["generator_hash"] = "0" * 64
    write_json(path, record)
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid matched control reached optimization")
    monkeypatch.setattr(api, "train_padded_model", forbidden)
    with pytest.raises(ValueError):
        _run(comparison_case, expected_controls_sha256=sha(path))


@pytest.mark.parametrize("problem", ("updates", "order", "adjacency", "state-hash", "input"))
def test_corrupted_execution_cannot_publish_comparison(comparison_case, monkeypatch, problem):
    api = _api()
    actual = api.train_padded_model
    def train(model, training, validation, config):
        result = actual(model, training, validation, config)
        if problem == "updates": return replace(result, optimizer_steps=result.optimizer_steps - 1)
        if problem == "order": return replace(result, epoch_order_hashes=("0" * 64,) * config.epochs)
        if problem == "state-hash": return replace(result, state_hash="0" * 64)
        if problem == "input": training.targets[0] = (training.targets[0] + 1) % 10
        if problem == "adjacency" and hasattr(model, "adjacency"): model.adjacency[0, 0] += 1
        return result
    monkeypatch.setattr(api, "train_padded_model", train)
    with pytest.raises(ValueError):
        _run(comparison_case)


@pytest.mark.parametrize("delta", (-.125, 0., .125))
def test_negative_null_and_positive_validation_differences_do_not_authorize_claims(comparison_case, monkeypatch, delta):
    api = _api()
    from yolo_flywire.eval import MetricBundle
    actual, scores = api.train_padded_model, {}
    def train(model, training, validation, config):
        run = actual(model, training, validation, config)
        score = .5 + delta if len(scores) % 4 == 3 else .5
        scores[id(model)] = score
        return replace(run, best_validation_macro_f1=score)
    monkeypatch.setattr(api, "train_padded_model", train)
    monkeypatch.setattr(api, "evaluate_padded", lambda model, *args, **kwargs: MetricBundle(scores[id(model)], .5))
    report = _run(comparison_case)
    assert all(row["difference"] == delta for row in report["paired_validation"])
    assert report["topology_claim_evaluated"] is False
    assert report["final_test_evaluated"] is False
