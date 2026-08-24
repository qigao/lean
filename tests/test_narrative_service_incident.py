from __future__ import annotations

import unittest

_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.domains.service_incident import (
        service_direct_model,
        service_epistemic_model,
        service_incident_domain,
        service_incident_recovered_claim_story,
        service_incident_stale_claim_story,
        service_omniscient_model,
    )
except ImportError as error:
    _IMPORT_ERROR = error

from narrative_dynamics.narrative.analysis import (
    analyze_narrative,
    build_trajectory,
    derive_analysis_scope,
)
from narrative_dynamics.narrative.interventions import (
    evaluate_counterfactual,
    generate_minimal_interventions,
)


class ServiceIncidentConformanceTests(unittest.TestCase):
    def require_service_incident(self) -> None:
        if _IMPORT_ERROR is not None:
            self.fail(f"service incident conformance domain is missing: {_IMPORT_ERROR}")

    def test_pair_and_model_contrast(self) -> None:
        self.require_service_incident()
        recovered = service_incident_recovered_claim_story()
        stale = service_incident_stale_claim_story()
        domain = service_incident_domain()

        self.assertEqual(recovered.events, stale.events)
        self.assertEqual(recovered.observations, stale.observations)
        self.assertEqual(recovered.receptions, stale.receptions)
        self.assertEqual(recovered.decisions, stale.decisions)
        self.assertNotEqual(
            recovered.claims[0].proposition.value,
            stale.claims[0].proposition.value,
        )

        expected = {
            "recovered": ("restart_service", "leave_running", "leave_running"),
            "stale": ("restart_service", "restart_service", "leave_running"),
        }
        for name, story in (("recovered", recovered), ("stale", stale)):
            actual = tuple(
                analyze_narrative(story, domain, "d1", model).baseline.selected_action
                for model in (
                    service_direct_model(),
                    service_epistemic_model(),
                    service_omniscient_model(),
                )
            )
            self.assertEqual(actual, expected[name])

    def test_reception_ablation(self) -> None:
        self.require_service_incident()
        domain = service_incident_domain()
        model = service_epistemic_model()
        for story, action_changed in (
            (service_incident_recovered_claim_story(), True),
            (service_incident_stale_claim_story(), False),
        ):
            scope = derive_analysis_scope(story, domain, "d1")
            baseline = build_trajectory(story, domain, scope, model)
            intervention = next(
                item
                for item in generate_minimal_interventions(story, domain, scope)
                if item.kind == "remove_reception"
            )
            result = evaluate_counterfactual(
                story,
                domain,
                scope,
                model,
                baseline,
                intervention,
            )
            self.assertEqual(result.status, "valid")
            self.assertIsNotNone(result.first_divergence)
            self.assertEqual(result.first_divergence.logical_time, 3)
            self.assertEqual(result.first_divergence.action_changed, action_changed)


if __name__ == "__main__":
    unittest.main()
