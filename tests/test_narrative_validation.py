from __future__ import annotations

from dataclasses import replace
import unittest

_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.domain import validate_narrative
except ImportError as error:
    _IMPORT_ERROR = error

from narrative_dynamics.narrative.ir import (
    ActionOption,
    Observation,
    Proposition,
    Reception,
    TypedValue,
)
from narrative_test_support import make_test_domain, make_test_story


class NarrativeValidationTests(unittest.TestCase):
    def require_validator(self) -> None:
        if _IMPORT_ERROR is not None:
            self.fail(f"canonical narrative validator is missing: {_IMPORT_ERROR}")

    def test_support_requires_access_but_not_truth(self) -> None:
        self.require_validator()
        domain = make_test_domain()
        story = make_test_story()
        stale = replace(
            story.claims[0],
            proposition=replace(
                story.claims[0].proposition,
                value=TypedValue("HealthState", "healthy"),
            ),
        )
        validate_narrative(replace(story, claims=(stale,)), domain)
        with self.assertRaisesRegex(ValueError, "speaker.*support"):
            validate_narrative(
                make_test_story(alice_observes_recovery=False), domain
            )

    def test_global_time_uniqueness(self) -> None:
        self.require_validator()
        domain = make_test_domain()
        story = make_test_story()
        claim = replace(story.claims[0], logical_time=1)
        with self.assertRaisesRegex(
            ValueError, "logical times must be globally unique"
        ):
            validate_narrative(replace(story, claims=(claim,)), domain)

    def test_domain_identity_and_typed_proposition_are_checked(self) -> None:
        self.require_validator()
        domain = make_test_domain()
        story = make_test_story()
        with self.assertRaisesRegex(ValueError, "domain identity"):
            validate_narrative(
                replace(story, domain_spec_hash="sha256:" + "9" * 64),
                domain,
            )
        bad_claim = replace(
            story.claims[0],
            proposition=Proposition(
                story.claims[0].proposition.subject,
                "service.secret",
                "equals",
                TypedValue("HealthState", "recovered"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "state variable"):
            validate_narrative(replace(story, claims=(bad_claim,)), domain)

    def test_prior_claim_support_requires_reception_by_speaker(self) -> None:
        self.require_validator()
        domain = make_test_domain()
        story = make_test_story(second_claim_value="failed")
        second = replace(story.claims[1], support_refs=("c1",))
        supported = replace(
            story,
            claims=(story.claims[0], second),
            receptions=story.receptions
            + (Reception("r-support", "c1", "carol"),),
        )
        validate_narrative(supported, domain)
        with self.assertRaisesRegex(ValueError, "speaker.*support"):
            validate_narrative(
                replace(supported, receptions=story.receptions), domain
            )

    def test_observation_pairs_and_decision_actions_fail_closed(self) -> None:
        self.require_validator()
        domain = make_test_domain()
        story = make_test_story()
        duplicate_observation = Observation("o-extra", "bob", "e1")
        with self.assertRaisesRegex(ValueError, "observation.*unique"):
            validate_narrative(
                replace(
                    story,
                    observations=story.observations + (duplicate_observation,),
                ),
                domain,
            )

        bad_decision = replace(
            story.decisions[0],
            actions=(ActionOption("restart", "other-action", {}),),
        )
        with self.assertRaisesRegex(ValueError, "decision action type"):
            validate_narrative(
                replace(story, decisions=(bad_decision,)), domain
            )


if __name__ == "__main__":
    unittest.main()
