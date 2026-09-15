from __future__ import annotations

from dataclasses import replace

import pytest

from yolo_flywire.cx_development import (
    CxDevelopmentBinding,
    CxDevelopmentBudget,
    CxDevelopmentResult,
    CxScoreboard,
    assert_split_access,
    development_split_for_subject,
    summarize_gate1,
    validate_comparison_results,
)
from yolo_flywire.cx_protocol import load_cx_protocol


SEEDS = (7, 11, 19, 23, 31)


def _protocol(tmp_path):
    # Use the repository preflight unchanged; development bindings are measured
    # independently before Task 9 freezes those measurements into the protocol.
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    return load_cx_protocol(root / "protocols" / "v2-cx-temporal-gate1-preflight.json")


def _budget():
    return CxDevelopmentBudget(
        optimizer="adam",
        learning_rate=1e-3,
        epochs=3,
        max_updates=100,
        batch_size=1,
        early_stopping="best-validation-macro-f1-over-all-epochs",
        retention_horizon=4,
        perturbation_fraction=0.25,
        recovery_horizon=3,
    )


def _binding():
    return CxDevelopmentBinding(
        dataset_content_hash="1" * 64,
        split_hash="2" * 64,
        event_encoder_hash="3" * 64,
        dynamics_hash="4" * 64,
        input_node_fingerprint="5" * 64,
        output_node_fingerprint="6" * 64,
        real_graph_fingerprint="7" * 64,
        degree_graph_fingerprints=tuple((seed, f"{seed:064x}") for seed in SEEDS),
        block_graph_fingerprints=tuple((seed, f"{seed + 100:064x}") for seed in SEEDS),
        budget=_budget(),
        seeds=SEEDS,
    )


def _score(*, early: float, retention: float, recovery: float) -> CxScoreboard:
    return CxScoreboard(
        full_macro_f1=0.8,
        early_macro_f1_by_ratio=((0.10, 0.3), (0.20, 0.4), (0.40, 0.6), (0.60, 0.75), (0.80, 0.8), (1.0, 0.8)),
        early_prediction_auc=early,
        retention_auc=retention,
        retention_t50=4.0,
        recovery_rate=recovery,
        median_recovery_steps=1.0,
        spike_rate=0.1,
        active_neuron_fraction=0.1,
        synaptic_events_per_frame=12.0,
        parameter_count=123,
    )


def _result(binding: CxDevelopmentBinding, arm: str, seed: int, *, early: float, retention: float, recovery: float):
    graph = (
        binding.real_graph_fingerprint
        if arm == "real"
        else dict(binding.degree_graph_fingerprints)[seed]
        if arm == "degree"
        else dict(binding.block_graph_fingerprints)[seed]
    )
    return CxDevelopmentResult(
        arm=arm,
        seed=seed,
        graph_fingerprint=graph,
        dataset_content_hash=binding.dataset_content_hash,
        split_hash=binding.split_hash,
        event_encoder_hash=binding.event_encoder_hash,
        dynamics_hash=binding.dynamics_hash,
        input_node_fingerprint=binding.input_node_fingerprint,
        output_node_fingerprint=binding.output_node_fingerprint,
        budget=binding.budget,
        scoreboard=_score(early=early, retention=retention, recovery=recovery),
    )


def _matrix(binding: CxDevelopmentBinding):
    rows = []
    for seed in SEEDS:
        rows.extend(
            (
                _result(binding, "real", seed, early=0.70, retention=0.60, recovery=0.70),
                _result(binding, "degree", seed, early=0.67, retention=0.53, recovery=0.65),
                _result(binding, "block", seed, early=0.69, retention=0.58, recovery=0.68),
            )
        )
    return tuple(rows)


def test_development_validation_split_is_frozen_inside_official_outer_train():
    assert development_split_for_subject(31) == "validation"
    assert development_split_for_subject(85) == "validation"
    assert development_split_for_subject(56) == "train"
    assert development_split_for_subject(3) == "final_test"
    with pytest.raises(ValueError, match="subject"):
        development_split_for_subject(0)


def test_budget_rejects_changed_or_unbounded_training_contract():
    with pytest.raises(ValueError, match="optimizer"):
        replace(_budget(), optimizer="sgd")
    with pytest.raises(ValueError, match="positive"):
        replace(_budget(), epochs=0)
    with pytest.raises(ValueError, match="max_updates"):
        replace(_budget(), max_updates=0)
    with pytest.raises(ValueError, match="batch_size"):
        replace(_budget(), batch_size=2)


def test_preflight_keeps_final_test_sealed():
    protocol = _protocol(None)
    assert_split_access(protocol, "train")
    assert_split_access(protocol, "validation")
    with pytest.raises(ValueError, match="final test is sealed"):
        assert_split_access(protocol, "final_test")


def test_complete_comparison_matrix_is_accepted_only_when_all_pins_match():
    binding = _binding()
    rows = _matrix(binding)
    validated = validate_comparison_results(binding, rows)
    assert validated == rows


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        (lambda row: replace(row, event_encoder_hash="a" * 64), "event_encoder_hash"),
        (lambda row: replace(row, input_node_fingerprint="a" * 64), "input_node_fingerprint"),
        (lambda row: replace(row, output_node_fingerprint="a" * 64), "output_node_fingerprint"),
        (lambda row: replace(row, dynamics_hash="a" * 64), "dynamics_hash"),
        (lambda row: replace(row, budget=replace(row.budget, epochs=4)), "budget"),
        (lambda row: replace(row, graph_fingerprint="a" * 64), "graph_fingerprint"),
    ],
)
def test_comparison_rejects_any_arm_with_changed_contract_pin(mutation, pattern):
    binding = _binding()
    rows = list(_matrix(binding))
    rows[4] = mutation(rows[4])
    with pytest.raises(ValueError, match=pattern):
        validate_comparison_results(binding, tuple(rows))


def test_comparison_rejects_missing_extra_or_duplicate_arm_seed_cells():
    binding = _binding()
    rows = list(_matrix(binding))
    with pytest.raises(ValueError, match="complete arm/seed matrix"):
        validate_comparison_results(binding, tuple(rows[:-1]))
    with pytest.raises(ValueError, match="complete arm/seed matrix"):
        validate_comparison_results(binding, tuple(rows + [rows[0]]))


def test_gate1a_passes_only_on_primary_plus_temporal_effect_rule():
    binding = _binding()
    summary = summarize_gate1(binding, _matrix(binding))
    assert summary.gate1a.status == "pass"
    assert summary.gate1a.early_mean_difference == pytest.approx(0.03)
    assert summary.gate1a.positive_early_seeds == 5
    assert summary.gate1a.retention_mean_difference == pytest.approx(0.07)
    assert summary.gate1b.status == "null"


def test_full_sequence_accuracy_cannot_turn_a_temporal_null_into_pass():
    binding = _binding()
    rows = []
    for seed in SEEDS:
        real = _result(binding, "real", seed, early=0.70, retention=0.60, recovery=0.70)
        real = replace(real, scoreboard=replace(real.scoreboard, full_macro_f1=0.99))
        rows.extend(
            (
                real,
                _result(binding, "degree", seed, early=0.695, retention=0.595, recovery=0.695),
                _result(binding, "block", seed, early=0.695, retention=0.595, recovery=0.695),
            )
        )
    summary = summarize_gate1(binding, tuple(rows))
    assert summary.gate1a.status == "null"
    assert summary.gate1b.status == "null"


def test_strong_reverse_early_effect_is_reported_negative_not_null():
    binding = _binding()
    rows = []
    for seed in SEEDS:
        rows.extend(
            (
                _result(binding, "real", seed, early=0.60, retention=0.60, recovery=0.70),
                _result(binding, "degree", seed, early=0.64, retention=0.60, recovery=0.70),
                _result(binding, "block", seed, early=0.64, retention=0.60, recovery=0.70),
            )
        )
    summary = summarize_gate1(binding, tuple(rows))
    assert summary.gate1a.status == "negative"
    assert summary.gate1b.status == "negative"
