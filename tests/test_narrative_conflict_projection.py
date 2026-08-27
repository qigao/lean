from __future__ import annotations

from types import MappingProxyType
import unittest

from narrative_dynamics.narrative.domain import StateDelta, StateDeltaOp
from narrative_dynamics.narrative.observation_projection import (
    ObservationCapabilitySpec,
    ObservationFact,
    ObservationProjectionError,
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


class ServiceOwnerClearProjectionHook:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        cell = conflict._cell("svc", "Service", "service.owner")
        if cell in next_visible:
            return ()
        return (ObservationFact(cell, "clear", None),)


class ClearClaimResolver:
    def __call__(self, snapshot, context):
        return StateDelta(
            (
                StateDeltaOp(
                    "clear",
                    "svc",
                    "service.owner",
                    None,
                ),
            )
        )


def make_projection_model(domain, hook=None):
    capability = ObservationCapabilitySpec("service.owner", "any")
    if hook is None:
        hook = ServiceOwnerProjectionHook()
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
                hook,
            ),
        ),
    )


def make_resolved_claim_step(resolver_hook=None):
    domain = conflict.make_conflict_domain()
    story = conflict.make_conflict_story(domain)
    if resolver_hook is None:
        resolver_hook = conflict.ClaimResolver()
    resolver = conflict.make_resolver(
        domain,
        resolver_hook,
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
    return domain, story, world_step


def participant_hashes(world_step):
    resolution = world_step.conflict_resolutions[0]
    return tuple(
        sorted(
            participant.transition_record_hash
            for participant in resolution.context.participants
        )
    )


class NarrativeConflictProjectionTests(unittest.TestCase):
    def test_resolved_component_projects_effective_world_with_participant_lineage(self):
        domain, story, world_step = make_resolved_claim_step()

        self.assertEqual(len(world_step.conflict_resolutions), 1)
        expected_hashes = participant_hashes(world_step)

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
                expected_hashes,
            )

    def test_resolved_explicit_clear_projects_as_effective_current_step_clear(self):
        domain, story, world_step = make_resolved_claim_step(ClearClaimResolver())
        expected_hashes = participant_hashes(world_step)

        projected = project_world_observations(
            story,
            domain,
            world_step,
            make_projection_model(domain, ServiceOwnerClearProjectionHook()),
        )

        self.assertTrue(projected.observations)
        for observation in projected.observations:
            self.assertEqual(observation.fact.relation, "clear")
            self.assertIsNone(observation.fact.value)
            self.assertEqual(
                observation.source_transition_hashes,
                expected_hashes,
            )

    def test_forged_participant_transition_reference_rejects_projection(self):
        domain, story, world_step = make_resolved_claim_step()
        resolution = world_step.conflict_resolutions[0]
        first, *rest = resolution.context.participants
        forged_participant = conflict._forge(
            first,
            transition_record_hash=conflict._hash("missing-transition"),
        )
        forged_context = conflict._forge(
            resolution.context,
            participants=(forged_participant, *rest),
        )
        forged_resolution = conflict._forge(
            resolution,
            context=forged_context,
        )
        forged_step = conflict._forge(
            world_step,
            conflict_resolutions=(forged_resolution,),
        )

        with self.assertRaises(ObservationProjectionError):
            project_world_observations(
                story,
                domain,
                forged_step,
                make_projection_model(domain),
            )

    def test_forged_v2_batch_hash_rejects_projection(self):
        domain, story, world_step = make_resolved_claim_step()
        forged_next = conflict._forge(
            world_step.next_state,
            transition_batch_hash=conflict._hash("forged-v2-batch"),
        )
        forged_step = conflict._forge(world_step, next_state=forged_next)

        with self.assertRaises(ObservationProjectionError):
            project_world_observations(
                story,
                domain,
                forged_step,
                make_projection_model(domain),
            )

    def test_effective_replay_mismatch_rejects_projection(self):
        domain, story, world_step = make_resolved_claim_step()
        values = dict(world_step.next_state.values)
        values[conflict._cell("svc", "Service", "service.owner")] = (
            conflict._agent_ref("bob")
        )
        forged_next = conflict._forge(
            world_step.next_state,
            values=MappingProxyType(values),
        )
        forged_step = conflict._forge(world_step, next_state=forged_next)

        with self.assertRaises(ObservationProjectionError):
            project_world_observations(
                story,
                domain,
                forged_step,
                make_projection_model(domain),
            )

    def test_unresolved_raw_overlap_without_resolution_rejects_projection(self):
        domain, story, world_step = make_resolved_claim_step()
        forged_step = conflict._forge(
            world_step,
            conflict_resolutions=(),
        )

        with self.assertRaises(ObservationProjectionError):
            project_world_observations(
                story,
                domain,
                forged_step,
                make_projection_model(domain),
            )


if __name__ == "__main__":
    unittest.main()
