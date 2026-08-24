from __future__ import annotations

from collections.abc import Mapping
import random

from narrative_dynamics.contracts import ModelRun, Scenario, TraceEvent
from narrative_dynamics.story.replay_v2 import (
    latest_epistemic_location,
    testimony_state,
)
from narrative_dynamics.story.scenario_v2 import decode_testimony_scenario


class TestimonySearchModel:
    """Choose the search action matching the actor's latest supported evidence."""

    name = "testimony-search"

    def simulate(
        self,
        scenario: Scenario,
        parameters: Mapping[str, float],
        rng: random.Random,
    ) -> ModelRun:
        if parameters:
            raise ValueError("testimony-search requires an empty parameter mapping")

        story = decode_testimony_scenario(scenario)
        decision = story.decision
        state = testimony_state(
            story,
            decision.actor,
            at_time=decision.time,
        )
        location = latest_epistemic_location(state, decision.object)
        matching = [
            action.id
            for action in decision.actions
            if action.location == location.location
        ]
        if len(matching) != 1:
            raise ValueError(
                "testimony-aware target location must match exactly one decision action"
            )

        selected = matching[0]
        scores = {
            action.id: 1.0 if action.id == selected else 0.0
            for action in decision.actions
        }
        policy = dict(scores)
        basis = {
            "kind": location.evidence_kind,
            "target_object": decision.object,
            "resolved_location": location.location,
            "supporting_id": location.supporting_id,
            "source_agent": location.source_agent,
        }

        return ModelRun(
            events=(
                TraceEvent(
                    tick=0,
                    kind="epistemic_basis_resolved",
                    data=basis,
                ),
                TraceEvent(
                    tick=1,
                    kind="action_policy_computed",
                    data={"policy": policy},
                ),
                TraceEvent(
                    tick=2,
                    kind="action_selected",
                    data={"action": selected},
                ),
            ),
            outcome={
                "epistemic_basis": basis,
                "action_scores": scores,
                "policy": policy,
                "selected_action": selected,
            },
        )
