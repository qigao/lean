from pathlib import Path
import json

import pytest

from yolo_flywire.cx_protocol import load_cx_protocol, validate_cx_protocol_dict


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "v2-cx-temporal-gate1-preflight.json"

_REAL_SOURCE_HASHES = (
    ("consolidated_cell_types", "8aba246d71dc40361677493629972ce3883048c3d02010adc42bda22962a1a2d"),
    ("connections_princeton", "445f996bf6c4b1803b9ba186189138a3061ff8623aa94c0abcf38af30a5bd48b"),
)
_REAL_ARTIFACT = "ee1387398e0d6d7404d78bdbce99783209e18f4ae408001d9a9dd2596ed75cd7"
_REAL_INPUT = "f917eb9fa128c2090671ecba8fd1c75db5dd57d2e603b1b0a649bea64cfd17b8"
_REAL_OUTPUT = "f966a185ccb9d34369c55ba20e1f509c1bbc16265e1b4880e45618ad451aa597"
_REAL_DEGREE = (
    ("7", "6c47a71b0ad9787608b84e89f2cce5615b2b29a5cf2a9cd0edc35ff6486fb6f0"),
    ("11", "6ad62b7cc2e95429c9bfc7a0fde2e44f0c4d6a4e6f027f3c1fccef6a3cb170f0"),
    ("19", "038553265a0d4835e8f2107eb6a499bf232b6612fed17c75c79459ce7e567944"),
    ("23", "465e5eb04527f30577119fab1eb00b9ea965f006774e91b8470a0706f472e82b"),
    ("31", "dbf6cb2b43f6dcced1f43575cfe566c43501a541815eee7db392dab854583bc8"),
)
_REAL_BLOCK = (
    ("7", "06afe32d006fca0be3d940bd2d38ef7cac99930f6963f52e555e6965c3738edf"),
    ("11", "fbbe1b51ff7a14db4b0b356cbef1196903f3684891c74e80546172b0b1ed3e53"),
    ("19", "85fa3c223228c1360e4116cb03bec41d71a632e8cc79a10180f5894ea963f6a9"),
    ("23", "a36f4549f059c7cea3155ae344a2d5651d0a4cfd2909a640d5910f17134f1bd6"),
    ("31", "d07aa024a6556d76c9223eb6a12376509f365075fa8fd2f3ce7fe53a6bc40486"),
)


def _raw() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def test_preflight_freezes_gate1_choices_and_measured_cx_without_final_readiness():
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
    assert protocol.source_hashes == _REAL_SOURCE_HASHES
    assert protocol.cx_artifact_fingerprint == _REAL_ARTIFACT
    assert protocol.input_node_fingerprint == _REAL_INPUT
    assert protocol.output_node_fingerprint == _REAL_OUTPUT
    assert protocol.degree_null_fingerprints == _REAL_DEGREE
    assert protocol.block_null_fingerprints == _REAL_BLOCK
    assert protocol.dataset_content_hash is None
    assert protocol.split_hash is None
    assert protocol.event_encoder_hash is None
    assert protocol.dynamics_hash is None


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


def test_final_test_cannot_be_enabled_with_remaining_null_provenance():
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
