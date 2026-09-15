from pathlib import Path
import json

import pytest

from pose_temporal_snn.protocol import load_protocol, validate_protocol_dict


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "v1-development-preflight.json"


def test_preflight_freezes_v1_without_opening_final_test():
    protocol = load_protocol(PROTOCOL)
    assert protocol.actions == (8, 9, 22, 23, 26, 27, 31, 34, 35, 36)
    assert protocol.validation_subjects == (14, 28, 35, 46, 50, 54, 74, 83, 84, 86, 103)
    assert protocol.event_quantile == pytest.approx(0.75)
    assert protocol.observation_ratios == (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
    assert protocol.seeds == (7, 11, 19, 23, 31)
    assert protocol.parameter_ceiling == 50_000
    assert protocol.rsnn.hidden_size == 96
    assert protocol.spiking_graph.hidden_size == 32
    assert protocol.gru.hidden_size == 56
    assert protocol.final_test_enabled is False


def test_final_test_requires_all_measured_pins():
    raw = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    raw["final_test_enabled"] = True
    with pytest.raises(ValueError, match="final test provenance"):
        validate_protocol_dict(raw)


def test_flywire_fields_are_not_part_of_the_new_schema():
    raw = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    raw["flywire_release"] = "forbidden"
    with pytest.raises(ValueError, match="schema"):
        validate_protocol_dict(raw)


def test_validation_subjects_must_stay_inside_outer_train():
    raw = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    raw["validation_subjects"][-1] = 3
    with pytest.raises(ValueError, match="validation subjects"):
        validate_protocol_dict(raw)


def test_model_parameter_ceiling_and_training_budget_are_frozen():
    protocol = load_protocol(PROTOCOL)
    assert protocol.training.optimizer == "adam"
    assert protocol.training.learning_rate == pytest.approx(1e-3)
    assert protocol.training.batch_size == 1
    assert protocol.training.epochs == 10
    assert protocol.training.max_updates == 2000
    assert protocol.training.checkpoint_rule == "best-validation-macro-f1-over-all-epochs"
