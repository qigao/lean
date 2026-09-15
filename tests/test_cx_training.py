from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from yolo_flywire.cx_artifact import CxArtifact, CxNode
from yolo_flywire.cx_development import CxDevelopmentBinding, CxDevelopmentBudget
from yolo_flywire.cx_training import CxEventSample, train_validation_arm
from yolo_flywire.graphs import DirectedGraph
from yolo_flywire.models.cx_lif import CxDynamics


SEEDS = (7, 11, 19, 23, 31)
DYNAMICS = CxDynamics(
    tau_membrane=20.0,
    synaptic_decay=0.8,
    refractory_steps=1,
    threshold=0.25,
    reset=0.0,
    recurrent_delay_steps=1,
    recurrent_gain=0.1,
    magnitude_policy="log1p",
)


def _budget() -> CxDevelopmentBudget:
    return CxDevelopmentBudget(
        optimizer="adam",
        learning_rate=1e-2,
        epochs=2,
        max_updates=6,
        batch_size=1,
        early_stopping="best-validation-macro-f1-over-all-epochs",
        retention_horizon=2,
        perturbation_fraction=0.5,
        recovery_horizon=2,
    )


def _artifact(kind: str) -> CxArtifact:
    nodes = (
        CxNode(10, "ER1", "ER", "input"),
        CxNode(11, "PFNa", "PFN", "input"),
        CxNode(20, "EPG", "EPG", "core"),
        CxNode(21, "PEN_a", "PEN", "core"),
        CxNode(30, "PFL2", "PFL", "output"),
        CxNode(31, "PFL3", "PFL", "output"),
    )
    topologies = {
        "real": ((0, 2), (1, 3), (2, 4), (3, 5), (4, 3), (5, 2)),
        "degree": ((0, 3), (1, 2), (2, 5), (3, 4), (4, 2), (5, 3)),
        "block": ((0, 2), (1, 3), (2, 5), (3, 4), (4, 3), (5, 2)),
    }
    pairs = topologies[kind]
    graph = DirectedGraph(
        num_nodes=6,
        src=tuple(src for src, _ in pairs),
        dst=tuple(dst for _, dst in pairs),
        weight=(8.0, 7.0, 9.0, 6.0, 5.0, 5.0),
    )
    fingerprints = {"real": "a" * 64, "degree": "b" * 64, "block": "c" * 64}
    return CxArtifact(
        nodes=nodes,
        graph=graph,
        input_indices=(0, 1),
        core_indices=(2, 3),
        output_indices=(4, 5),
        edge_signs=(1, 1, 1, 1, -1, -1),
        source_hashes=(("consolidated_cell_types", "d" * 64), ("connections_princeton", "e" * 64)),
        matched_primary_types=tuple(node.primary_type for node in nodes),
        fingerprint=fingerprints[kind],
    )


def _binding() -> CxDevelopmentBinding:
    return CxDevelopmentBinding(
        dataset_content_hash="1" * 64,
        split_hash="2" * 64,
        event_encoder_hash="3" * 64,
        dynamics_hash="4" * 64,
        input_node_fingerprint="5" * 64,
        output_node_fingerprint="6" * 64,
        real_graph_fingerprint="a" * 64,
        degree_graph_fingerprints=tuple((seed, "b" * 64 if seed == 7 else f"{seed:064x}") for seed in SEEDS),
        block_graph_fingerprints=tuple((seed, "c" * 64 if seed == 7 else f"{seed + 100:064x}") for seed in SEEDS),
        budget=_budget(),
        seeds=SEEDS,
    )


def _events(sign: float) -> torch.Tensor:
    value = torch.zeros(8, 4, dtype=torch.float32)
    value[1:5, 0] = sign
    value[3:7, 1] = -sign
    value[2:6, 2] = sign
    return value


def _data():
    train = (
        CxEventSample("tr-0a", 0, _events(+1.0)),
        CxEventSample("tr-1a", 1, _events(-1.0)),
        CxEventSample("tr-0b", 0, torch.roll(_events(+1.0), 1, 0)),
        CxEventSample("tr-1b", 1, torch.roll(_events(-1.0), 1, 0)),
    )
    validation = (
        CxEventSample("va-0", 0, _events(+1.0)),
        CxEventSample("va-1", 1, _events(-1.0)),
    )
    return train, validation


def test_same_seed_real_and_null_arms_share_initialization_order_and_budget():
    binding = _binding()
    train, validation = _data()
    outcomes = {
        arm: train_validation_arm(
            _artifact(arm),
            binding=binding,
            arm=arm,
            seed=7,
            train_samples=train,
            validation_samples=validation,
            dynamics=DYNAMICS,
        )
        for arm in ("real", "degree", "block")
    }
    assert len({run.trace.initial_parameter_fingerprint for run in outcomes.values()}) == 1
    assert len({run.trace.training_order_fingerprint for run in outcomes.values()}) == 1
    assert len({run.result.scoreboard.parameter_count for run in outcomes.values()}) == 1
    assert outcomes["real"].result.graph_fingerprint == "a" * 64
    assert outcomes["degree"].result.graph_fingerprint == "b" * 64
    assert outcomes["block"].result.graph_fingerprint == "c" * 64
    assert all(0 <= run.trace.best_epoch < binding.budget.epochs for run in outcomes.values())
    assert all(run.trace.updates <= binding.budget.max_updates for run in outcomes.values())


def test_same_arm_seed_and_data_are_deterministic():
    binding = _binding()
    train, validation = _data()
    first = train_validation_arm(
        _artifact("real"), binding=binding, arm="real", seed=7,
        train_samples=train, validation_samples=validation, dynamics=DYNAMICS,
    )
    second = train_validation_arm(
        _artifact("real"), binding=binding, arm="real", seed=7,
        train_samples=train, validation_samples=validation, dynamics=DYNAMICS,
    )
    assert first.trace == second.trace
    assert first.result == second.result


def test_artifact_must_match_arm_seed_binding_pin():
    binding = _binding()
    train, validation = _data()
    with pytest.raises(ValueError, match="artifact fingerprint"):
        train_validation_arm(
            _artifact("degree"), binding=binding, arm="real", seed=7,
            train_samples=train, validation_samples=validation, dynamics=DYNAMICS,
        )


def test_training_rejects_bad_events_and_missing_validation():
    with pytest.raises(ValueError, match="signed events"):
        CxEventSample("bad", 0, torch.full((8, 4), 0.5))
    binding = _binding()
    train, _ = _data()
    with pytest.raises(ValueError, match="validation"):
        train_validation_arm(
            _artifact("real"), binding=binding, arm="real", seed=7,
            train_samples=train, validation_samples=(), dynamics=DYNAMICS,
        )


def test_changed_budget_pin_is_not_silently_accepted():
    binding = _binding()
    train, validation = _data()
    changed = replace(binding, budget=replace(binding.budget, learning_rate=2e-2))
    outcome = train_validation_arm(
        _artifact("real"), binding=changed, arm="real", seed=7,
        train_samples=train, validation_samples=validation, dynamics=DYNAMICS,
    )
    assert outcome.result.budget.learning_rate == pytest.approx(2e-2)
