from __future__ import annotations

"""Small film-conformance slice for the Generic Narrative Engine.

The fixture abstracts the red-pill choice from *The Matrix* into canonical
semantics without reproducing screenplay dialogue: the experienced world is
objectively represented as simulated, Morpheus has direct evidence for that
state and communicates it to Neo, and Neo chooses between the red and blue
pills using only the evidence capability granted to the decision model.
"""

from narrative_dynamics.narrative.decision import (
    DecisionChoice,
    DecisionModelSpec,
    DecisionResolutionError,
    EvidenceAccess,
    require_resolved_cell,
)
from narrative_dynamics.narrative.domain import (
    ActionTypeSpec,
    DecisionTypeSpec,
    DomainSpec,
    EntityTypeSpec,
    EventTypeSpec,
    ParameterSpec,
    SemanticHookBinding,
    StateDelta,
    StateDeltaOp,
    StateEffectSpec,
    StateVariableSpec,
    ValueTypeSpec,
)
from narrative_dynamics.narrative.ir import (
    ActionOption,
    Claim,
    Decision,
    Entity,
    EntityRef,
    GenericNarrative,
    NarrativeEvent,
    Observation,
    Proposition,
    Reception,
    StateCellRef,
    TypedValue,
)


def _world_ref() -> EntityRef:
    return EntityRef("experienced-world", "World")


def _world_cell() -> StateCellRef:
    return StateCellRef(_world_ref(), "world.mode")


class WorldModeHook:
    def __call__(self, prior_state, event):
        del prior_state
        world = event.arguments["world"].value
        mode = event.arguments["mode"]
        if not isinstance(world, EntityRef):
            raise TypeError("world mode establishment requires a WorldRef")
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    world.entity_id,
                    "world.mode",
                    mode,
                ),
            )
        )


def matrix_domain() -> DomainSpec:
    return DomainSpec(
        domain_id="film-the-matrix-conformance",
        version="1",
        entity_types=(EntityTypeSpec("Agent"), EntityTypeSpec("World")),
        value_types=(
            ValueTypeSpec("WorldRef", "entity_ref", entity_type="World"),
            ValueTypeSpec("WorldMode", "enum", ("simulated", "real")),
        ),
        state_variables=(
            StateVariableSpec("world.mode", "World", "WorldMode"),
        ),
        event_types=(
            EventTypeSpec(
                "WorldModeEstablished",
                None,
                (
                    ParameterSpec("world", "WorldRef"),
                    ParameterSpec("mode", "WorldMode"),
                ),
                (StateEffectSpec("world.mode", "world"),),
                "world_mode_established",
            ),
        ),
        action_types=(ActionTypeSpec("pill-action", ()),),
        decision_types=(
            DecisionTypeSpec(
                "pill-choice",
                "Agent",
                "pill-action",
            ),
        ),
        semantic_hooks=(
            SemanticHookBinding(
                "world_mode_established",
                "record whether the experienced world is simulated or real",
                WorldModeHook(),
            ),
        ),
    )


def matrix_red_pill_story() -> GenericNarrative:
    domain = matrix_domain()
    world = _world_ref()
    simulated = TypedValue("WorldMode", "simulated")
    return GenericNarrative(
        domain_id=domain.domain_id,
        domain_version=domain.version,
        domain_spec_hash=domain.content_hash,
        entities=(
            Entity("neo", "Agent"),
            Entity("morpheus", "Agent"),
            Entity("experienced-world", "World"),
        ),
        events=(
            NarrativeEvent(
                "matrix-is-simulation",
                1,
                "WorldModeEstablished",
                None,
                {
                    "world": TypedValue("WorldRef", world),
                    "mode": simulated,
                },
            ),
        ),
        observations=(
            Observation(
                "obs-morpheus-matrix",
                "morpheus",
                "matrix-is-simulation",
            ),
        ),
        claims=(
            Claim(
                "morpheus-matrix-claim",
                2,
                "morpheus",
                Proposition(
                    world,
                    "world.mode",
                    "equals",
                    simulated,
                ),
                ("matrix-is-simulation",),
            ),
        ),
        receptions=(
            Reception(
                "recv-neo-morpheus-claim",
                "morpheus-matrix-claim",
                "neo",
            ),
        ),
        decisions=(
            Decision(
                "neo-pill-choice",
                3,
                "neo",
                "pill-choice",
                (_world_cell(),),
                (
                    ActionOption("take_red_pill", "pill-action", {}),
                    ActionOption("take_blue_pill", "pill-action", {}),
                ),
            ),
        ),
    )


class PillChoiceHook:
    def __call__(self, context):
        view = require_resolved_cell(context, _world_cell())
        assert view.resolved_value is not None
        mode = view.resolved_value.value
        if mode == "simulated":
            action = "take_red_pill"
        elif mode == "real":
            action = "take_blue_pill"
        else:
            raise DecisionResolutionError(
                "resolved world mode does not map to a declared pill choice"
            )
        return DecisionChoice(
            selected_action=action,
            evidence_refs=(
                () if view.supporting_id is None else (view.supporting_id,)
            ),
            inspected_cells=(_world_cell(),),
        )


def _matrix_model(model_id: str, access: EvidenceAccess) -> DecisionModelSpec:
    return DecisionModelSpec(
        model_id=model_id,
        version="1",
        supported_decision_types=("pill-choice",),
        evidence_access=access,
        parameter_schema=(),
        decision_hook=PillChoiceHook(),
    )


def matrix_direct_model() -> DecisionModelSpec:
    return _matrix_model("matrix-direct", EvidenceAccess.DIRECT_ONLY)


def matrix_epistemic_model() -> DecisionModelSpec:
    return _matrix_model("matrix-epistemic", EvidenceAccess.EPISTEMIC)


def matrix_omniscient_model() -> DecisionModelSpec:
    return _matrix_model("matrix-omniscient", EvidenceAccess.OMNISCIENT)
