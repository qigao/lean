from __future__ import annotations

import unittest

from narrative_dynamics.narrative.observation_projection import (
    ObservationCapabilitySpec,
    ObservationFact,
    ObservationProjectionModelSpec,
    ObserverProjectionSpec,
    project_world_observations,
)
from narrative_dynamics.narrative.world import (
    advance_world_step,
    world_state_from_story,
)
import tests.test_narrative_conflict_resolution as conflict


class ServiceOwnerProjectionHook:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        cell = conflict._cell("svc", "Service", "service.owner")
        value = next_visible.get(cell)
        if value is None:
            return ()
        return (ObservationFact(cell, "equals", value),)


def make_projection_model(domain):
    capability = ObservationCapabilitySpec("service.owner", "any")
    return ObservationProjectionModelSpec(
        "conflict-projection",
        "1",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        (
            ObserverProjectionSpec(
                "Agent",
                "vision",
                (capability,),
                (capability,),
                ServiceOwnerProjectionHook(),
            ),
        ),
    )


class NarrativeConflictProjectionTests(unittest.TestCase):
    def test_resolved_component_projects_effective_world_with_participant_lineage(self):
        domain = conflict.make_conflict_domain()
        story = conflict.make_conflict_story(domain)
        resolver = conflict.make_resolver(
            domain,
            conflict.ClaimResolver(),
            ("claim-service-action",),
        )
        world_model = conflict.make_conflict_world_model(domain, resolver)
        prior = world_state_from_story(story, domain)
        world_step = advance_world_step(
            story,
            domain,
            prior,
            world_model,
            (
                conflict._conflict_intent(
                    "d-alice-claim-svc",
                    "alice-claim-svc",
                    "1",
                ),
                conflict._conflict_intent(
                    "d-bob-claim-svc",
                    "bob-claim-svc",
                    "2",
                ),
            ),
        )

        self.assertEqual(len(world_step.conflict_resolutions), 1)
        resolution = world_step.conflict_resolutions[0]
        participant_hashes = tuple(
            sorted(
                participant.transition_record_hash
                for participant in resolution.context.participants
            )
        )

        projected = project_world_observations(
            story,
            domain,
            world_step,
            make_projection_model(domain),
        )

        self.assertEqual(projected.source_world_step_hash, world_step.content_hash)
        self.assertTrue(projected.observations)
        for observation in projected.observations:
            self.assertEqual(
                observation.fact.cell,
                conflict._cell("svc", "Service", "service.owner"),
            )
            self.assertEqual(
                observation.fact.value,
                conflict._agent_ref("alice"),
            )
            self.assertEqual(
                observation.source_transition_hashes,
                participant_hashes,
            )


if __name__ == "__main__":
    unittest.main()
