from pathlib import Path
import json

import pytest

from yolo_flywire.cx_protocol import load_cx_protocol, validate_cx_protocol_dict


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "v2-cx-temporal-gate1-preflight.json"


def _raw() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_preflight_freezes_gate1_choices_without_claiming_final_readiness():
    protocol = load_cx_protocol(PROTOCOL)
    assert protocol.protocol_id == "v2-cx-temporal-gate1-preflight"
    assert protocol.dataset == "FAFB"
    assert protocol.release == "v783"
    assert protocol.cell_types_product == "consolidated_cell_types"
    assert protocol.connections_product == "connections_princeton"
    assert protocol.connection_threshold == 5
    assert protocol.input_families == ("ER", "ExR", "PFN")
    assert protocol.core_families == ("EPG", "PEG", "PEN", "Delta7", "hDelta", "vDelta")
    assert protocol.output_families == ("PFL",)
    assert protocol.seeds == (7, 11, 19, 23, 31)
    assert protocol.observation_ratios == (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
    assert protocol.early_auc_effect == pytest.approx(0.02)
    assert protocol.temporal_effect == pytest.approx(0.05)
    assert protocol.positive_seed_count == 4
    assert protocol.final_test_enabled is False
    assert protocol.cx_artifact_fingerprint is None
    assert protocol.degree_null_fingerprints is None
    assert protocol.block_null_fingerprints is None


def test_protocol_rejects_role_overlap():
    raw = _raw()
    raw["core_families"].append("PFN")
    with pytest.raises(ValueError, match="family roles must be disjoint"):
        validate_cx_protocol_dict(raw)


@pytest.mark.parametrize("field,value", [
    ("dataset", "male-cns"),
    ("release", "v999"),
    ("connection_threshold", 4),
    ("seeds", [7, 11, 19, 23, 99]),
    ("observation_ratios", [0.2, 0.4, 0.6, 0.8, 1.0]),
])
def test_protocol_rejects_changed_frozen_constants(field, value):
    raw = _raw()
    raw[field] = value
    with pytest.raises(ValueError, match=field.replace("_", " ") + "|frozen"):
        validate_cx_protocol_dict(raw)


def test_protocol_rejects_changed_neurotransmitter_sign_policy():
    raw = _raw()
    raw["nt_signs"]["GLUT"] = 1
    with pytest.raises(ValueError, match="neurotransmitter sign"):
        validate_cx_protocol_dict(raw)


@pytest.mark.parametrize("value", ["A" * 64, "0" * 63, "not-a-digest", 7])
def test_protocol_rejects_malformed_optional_digest(value):
    raw = _raw()
    raw["cx_artifact_fingerprint"] = value
    with pytest.raises(ValueError, match="cx_artifact_fingerprint"):
        validate_cx_protocol_dict(raw)


def test_final_test_cannot_be_enabled_with_null_provenance():
    raw = _raw()
    raw["final_test_enabled"] = True
    with pytest.raises(ValueError, match="final test provenance"):
        validate_cx_protocol_dict(raw)


def test_role_lookup_is_explicit():
    protocol = load_cx_protocol(PROTOCOL)
    assert protocol.role_for_family("ER") == "input"
    assert protocol.role_for_family("EPG") == "core"
    assert protocol.role_for_family("PFL") == "output"
    with pytest.raises(ValueError, match="unknown CX family"):
        protocol.role_for_family("T4")
