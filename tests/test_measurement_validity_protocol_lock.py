from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from narrative_dynamics.measurement_validity import MeasurementValidityProtocol
from narrative_dynamics.studies.feher_hare_measurement_validity_v1 import (
    FEHER_HARE_R3_LOCK_COMMIT,
    FeherHareMeasurementAnchor,
    build_feher_hare_measurement_protocol,
)


_ROOT = Path(__file__).resolve().parents[1]
_LOCK_PATH = (
    _ROOT
    / "research-locks"
    / "feher-hare-measurement-validity-v1-protocol.json"
)
_DESIGN_PATH = (
    "docs/superpowers/specs/"
    "2026-08-30-narrative-measurement-validity-v1-design.md"
)
_SCHEMA = "feher-hare-measurement-validity-v1-protocol-lock-v1"
_FIELDS = {
    "schema",
    "design_path",
    "design_file_sha256",
    "r3_lock_commit",
    "anchor",
    "protocol",
    "protocol_hash",
    "parameter_training_performed",
    "parameter_selection_performed",
    "final_test_values_exposed_to_audit",
    "final_test_outcomes_analyzed",
    "final_model_execution",
}
_EXECUTION_BOUNDARIES = (
    "parameter_training_performed",
    "parameter_selection_performed",
    "final_test_values_exposed_to_audit",
    "final_test_outcomes_analyzed",
    "final_model_execution",
)


class MeasurementValidityProtocolLockTests(unittest.TestCase):
    def test_frozen_protocol_rebuilds_from_exact_pre_final_anchor(self) -> None:
        payload = json.loads(_LOCK_PATH.read_text(encoding="utf-8"))

        self.assertIsInstance(payload, dict)
        self.assertEqual(set(payload), _FIELDS)
        self.assertEqual(payload["schema"], _SCHEMA)
        self.assertEqual(payload["design_path"], _DESIGN_PATH)
        design_bytes = (_ROOT / _DESIGN_PATH).read_bytes()
        self.assertEqual(
            payload["design_file_sha256"],
            f"sha256:{hashlib.sha256(design_bytes).hexdigest()}",
        )
        self.assertEqual(payload["r3_lock_commit"], FEHER_HARE_R3_LOCK_COMMIT)

        anchor = FeherHareMeasurementAnchor.from_payload(payload["anchor"])
        self.assertEqual(anchor.lock_commit, FEHER_HARE_R3_LOCK_COMMIT)
        self.assertEqual(anchor.to_payload(), payload["anchor"])

        protocol = MeasurementValidityProtocol.from_payload(payload["protocol"])
        self.assertEqual(protocol.to_payload(), payload["protocol"])
        self.assertEqual(protocol.content_hash, payload["protocol_hash"])

        rebuilt = build_feher_hare_measurement_protocol(anchor)
        self.assertEqual(rebuilt.content_hash, payload["protocol_hash"])
        self.assertEqual(rebuilt.to_payload(), payload["protocol"])

        for field in _EXECUTION_BOUNDARIES:
            with self.subTest(field=field):
                self.assertIs(payload[field], False)


if __name__ == "__main__":
    unittest.main()
