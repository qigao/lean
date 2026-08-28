from __future__ import annotations

import unittest

try:
    import narrative_dynamics.external_validation as external_validation
except ImportError as error:
    external_validation = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None

FORBIDDEN = {
    "empirically_identified",
    "structurally_identified",
    "cognitive_mechanism_confirmed",
    "latent_cognition_identified",
    "real_person_goal_recovered",
}


class ExternalValidationClaimFirewallTests(unittest.TestCase):
    def _api(self):
        if external_validation is None:
            self.fail(f"external validation API is missing: {IMPORT_ERROR}")
        return external_validation

    def test_claim_scope_is_exact_and_immutable(self):
        api = self._api()
        self.assertEqual(
            api.EXTERNAL_CLAIM_SCOPE,
            "external_observational_predictive_only",
        )

    def test_public_status_vocabularies_are_closed(self):
        api = self._api()
        self.assertEqual(
            {status.value for status in api.PredictiveAdequacyStatus},
            {"predictive_adequacy_met", "predictive_adequacy_not_met"},
        )
        self.assertEqual(
            {status.value for status in api.PredictiveSeparationStatus},
            {
                "predictively_separated_under_protocol",
                "not_predictively_separated_under_protocol",
            },
        )
        self.assertEqual(
            {status.value for status in api.ExternalConstraintStatus},
            {
                "constrained_under_external_protocol",
                "not_constrained_under_external_protocol",
            },
        )

    def test_external_validation_exposes_no_identification_status(self):
        api = self._api()
        self.assertFalse(hasattr(api, "IdentificationStatus"))

    def test_forbidden_empirical_identification_values_are_not_valid_statuses(self):
        api = self._api()
        public_values = {
            status.value
            for enum_type in (
                api.PredictiveAdequacyStatus,
                api.PredictiveSeparationStatus,
                api.ExternalConstraintStatus,
            )
            for status in enum_type
        }
        self.assertTrue(FORBIDDEN.isdisjoint(public_values))


if __name__ == "__main__":
    unittest.main()
