from __future__ import annotations

from dataclasses import replace

import pytest

from pose_graph_ssm.protocol import load_protocol
from pose_graph_ssm.results import DevelopmentCell, reduce_development_matrix, require_development_split
from pose_graph_ssm.training import RunResult, architecture_fingerprint, training_protocol_fingerprint


PROTOCOL = load_protocol("pose_graph_ssm/protocols/v1-development-preflight.json")


def _run(kind: str, seed: int, *, early: float, retention: float, recovery: float, full: float = 0.99) -> RunResult:
    return RunResult(
        model_kind=kind,
        seed=seed,
        architecture_fingerprint=architecture_fingerprint(kind, PROTOCOL),
        sample_order_fingerprint=f"{seed:064x}"[-64:],
        parameter_count=1000,
        update_count=80,
        best_epoch=1,
        best_validation_macro_f1=full,
        full_sequence_macro_f1=full,
        early_macro_f1_by_ratio=tuple((ratio, early) for ratio in PROTOCOL.observation_ratios),
        early_prediction_auc=early,
        retention_auc=retention,
        dropout_recovery_rate=recovery,
    )


def _cell(run: RunResult, *, split_hash: str = "2" * 64) -> DevelopmentCell:
    return DevelopmentCell(
        run=run,
        binding_fingerprint="a" * 64,
        dataset_content_hash="1" * 64,
        split_hash=split_hash,
        feature_stats_hash="3" * 64,
        training_fingerprint=training_protocol_fingerprint(PROTOCOL),
    )


def _matrix(*, primary_delta: float = 0.03, graph_delta: float = 0.02, ssm_delta: float = 0.02):
    cells = []
    for seed in PROTOCOL.seeds:
        gru = 0.50
        graph_ssm = gru + primary_delta
        ssm_only = graph_ssm - graph_delta
        graph_tcn = graph_ssm - ssm_delta
        values = {
            "gru": (gru, 0.50, 0.50),
            "ssm_only": (ssm_only, 0.52, 0.52),
            "graph_tcn": (graph_tcn, 0.53, 0.53),
            "graph_ssm": (graph_ssm, 0.57, 0.57),
        }
        for kind in PROTOCOL.model_kinds:
            early, retention, recovery = values[kind]
            cells.append(_cell(_run(kind, seed, early=early, retention=retention, recovery=recovery)))
    return cells


def test_complete_positive_matrix_passes_primary_and_both_attributions():
    summary = reduce_development_matrix(_matrix(), PROTOCOL)
    assert summary.primary_pass is True
    assert summary.graph_contribution_pass is True
    assert summary.ssm_contribution_pass is True
    assert summary.disposition == "continue_gate2"
    assert summary.cell_count == 20


def test_full_sequence_f1_cannot_override_temporal_null():
    cells = _matrix(primary_delta=0.0)
    cells = [replace(cell, run=replace(cell.run, full_sequence_macro_f1=0.999, best_validation_macro_f1=0.999)) for cell in cells]
    summary = reduce_development_matrix(cells, PROTOCOL)
    assert summary.primary_pass is False
    assert summary.disposition == "stop_primary_null"


def test_reducer_rejects_missing_or_duplicate_cells():
    cells = _matrix()
    with pytest.raises(ValueError, match="20|matrix"):
        reduce_development_matrix(cells[:-1], PROTOCOL)
    with pytest.raises(ValueError, match="duplicate"):
        reduce_development_matrix([*cells[:-1], cells[0]], PROTOCOL)


def test_reducer_rejects_binding_and_training_pin_drift():
    cells = _matrix()
    cells[0] = replace(cells[0], split_hash="f" * 64)
    with pytest.raises(ValueError, match="split"):
        reduce_development_matrix(cells, PROTOCOL)

    cells = _matrix()
    cells[0] = replace(cells[0], training_fingerprint="e" * 64)
    with pytest.raises(ValueError, match="training"):
        reduce_development_matrix(cells, PROTOCOL)


def test_reducer_rejects_model_and_sample_order_pin_drift():
    cells = _matrix()
    cells[0] = replace(cells[0], run=replace(cells[0].run, architecture_fingerprint="0" * 64))
    with pytest.raises(ValueError, match="architecture"):
        reduce_development_matrix(cells, PROTOCOL)

    cells = _matrix()
    graph_ssm_index = next(index for index, cell in enumerate(cells) if cell.run.seed == 7 and cell.run.model_kind == "graph_ssm")
    cells[graph_ssm_index] = replace(
        cells[graph_ssm_index],
        run=replace(cells[graph_ssm_index].run, sample_order_fingerprint="9" * 64),
    )
    with pytest.raises(ValueError, match="sample-order"):
        reduce_development_matrix(cells, PROTOCOL)


def test_final_test_split_is_hard_sealed():
    assert require_development_split("validation") == "validation"
    with pytest.raises(ValueError, match="final test is sealed"):
        require_development_split("final_test")
