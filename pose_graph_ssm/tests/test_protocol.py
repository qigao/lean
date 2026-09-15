from pathlib import Path
import json

import pytest

from pose_graph_ssm.protocol import load_protocol, validate_protocol_dict


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "v1-development-preflight.json"


def _raw() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_preflight_freezes_graph_ssm_v1_and_seals_final_test():
    p = load_protocol(PROTOCOL)
    assert p.protocol_id == "pose-graph-ssm-v1-development-preflight"
    assert p.actions == (8, 9, 22, 23, 26, 27, 31, 34, 35, 36)
    assert p.validation_subjects == (14, 28, 35, 46, 50, 54, 74, 83, 84, 86, 103)
    assert p.model_kinds == ("gru", "ssm_only", "graph_tcn", "graph_ssm")
    assert p.seeds == (7, 11, 19, 23, 31)
    assert p.observation_ratios == (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
    assert p.parameter_ceiling == 120_000
    assert p.retention_ratio == pytest.approx(0.40)
    assert p.retention_horizon == 20
    assert p.dropout_burst == 8
    assert p.recovery_horizon == 10
    assert p.training.optimizer == "adamw"
    assert p.training.learning_rate == pytest.approx(3e-4)
    assert p.training.weight_decay == pytest.approx(1e-4)
    assert p.training.batch_size == 1
    assert p.training.epochs == 20
    assert p.training.max_updates == 4000
    assert p.final_test_enabled is False
    assert p.dataset_content_hash is None
    assert p.feature_stats_hash is None
    assert p.model_fingerprints is None


def test_protocol_rejects_abandoned_stack_fields():
    raw = _raw()
    raw["flywire_release"] = "v783"
    with pytest.raises(ValueError, match="schema"):
        validate_protocol_dict(raw)


def test_protocol_rejects_changed_model_roster():
    raw = _raw()
    raw["model_kinds"] = ["gru", "graph_ssm"]
    with pytest.raises(ValueError, match="model"):
        validate_protocol_dict(raw)


def test_protocol_rejects_validation_subject_outside_outer_train():
    raw = _raw()
    raw["validation_subjects"][-1] = 3
    with pytest.raises(ValueError, match="validation"):
        validate_protocol_dict(raw)


def test_final_test_requires_every_measured_pin():
    raw = _raw()
    raw["final_test_enabled"] = True
    with pytest.raises(ValueError, match="final test provenance"):
        validate_protocol_dict(raw)


@pytest.mark.parametrize("field,value", [
    ("parameter_ceiling", 120001),
    ("retention_horizon", 19),
    ("dropout_burst", 7),
    ("recovery_horizon", 9),
    ("seeds", [7, 11, 19, 23, 99]),
])
def test_protocol_rejects_changed_frozen_constants(field, value):
    raw = _raw()
    raw[field] = value
    with pytest.raises(ValueError, match="frozen|remain"):
        validate_protocol_dict(raw)


@pytest.mark.parametrize("value", ["A" * 64, "0" * 63, "bad", 7])
def test_protocol_rejects_malformed_optional_digest(value):
    raw = _raw()
    raw["adjacency_hash"] = value
    with pytest.raises(ValueError, match="adjacency_hash"):
        validate_protocol_dict(raw)
