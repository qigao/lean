from __future__ import annotations

from dataclasses import fields, replace
import unittest

from narrative_dynamics.contracts import stable_content_hash
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
    Decision,
    Entity,
    EntityRef,
    GenericNarrative,
    NarrativeEvent,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.world import (
    ActionEffectSpec,
    ActionIntent,
    ActionTransitionSpec,
    WorldState,
    WorldStepResult,
    WorldTransitionModelSpec,
    advance_world_step,
    world_state_from_story,
)


_PROJECTION_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.observation_projection import (
        ObservationCapabilitySpec,
        ObservationFact,
        ObservationProjectionError,
        ObservationProjectionModelSpec,
        ObservationProjectionResult,
        ObserverProjectionSpec,
        ProjectedObservation,
        project_world_observations,
    )
except ImportError as error:
    _PROJECTION_IMPORT_ERROR = error


class SeedAgentPhaseHook:
    def __call__(self, prior_state, event):
        agent = event.arguments["agent"].value
        return StateDelta((StateDeltaOp("set", agent.entity_id, "agent.phase", event.arguments["phase"]),))


class SeedAgentLocationHook:
    def __call__(self, prior_state, event):
        agent = event.arguments["agent"].value
        return StateDelta((StateDeltaOp("set", agent.entity_id, "agent.location", event.arguments["location"]),))


class SeedServiceHealthHook:
    def __call__(self, prior_state, event):
        service = event.arguments["service"].value
        return StateDelta((StateDeltaOp("set", service.entity_id, "service.health", event.arguments["health"]),))


class SeedServiceAlertHook:
    def __call__(self, prior_state, event):
        service = event.arguments["service"].value
        return StateDelta((StateDeltaOp("set", service.entity_id, "service.alert", event.arguments["alert"]),))


class SeedRoomLightingHook:
    def __call__(self, prior_state, event):
        room = event.arguments["room"].value
        return StateDelta((StateDeltaOp("set", room.entity_id, "room.lighting", event.arguments["lighting"]),))


class PhaseTransitionHook:
    def __call__(self, prior_state, decision, action):
        return StateDelta((StateDeltaOp("set", decision.actor_id, "agent.phase", action.arguments["phase"]),))


class ClearAlertTransitionHook:
    def __call__(self, prior_state, decision, action):
        service = action.arguments["service"].value
        return StateDelta((StateDeltaOp("clear", service.entity_id, "service.alert", None),))


class NoopTransitionHook:
    def __call__(self, prior_state, decision, action):
        return StateDelta(())


class RecordViewsHook:
    def __init__(self, facts=()):
        self.calls = []
        self.facts = tuple(facts)

    def __call__(self, prior_visible, next_visible, observer, step_index):
        self.calls.append((dict(prior_visible), dict(next_visible), observer, step_index))
        return self.facts


class MutableProbeHook:
    def __init__(self):
        self.prior_is_immutable = False
        self.next_is_immutable = False

    def __call__(self, prior_visible, next_visible, observer, step_index):
        marker = StateCellRef(EntityRef(observer.id, observer.type_name), "agent.location")
        try:
            prior_visible[marker] = TypedValue("LocationState", "hall")
        except TypeError:
            self.prior_is_immutable = True
        try:
            next_visible[marker] = TypedValue("LocationState", "hall")
        except TypeError:
            self.next_is_immutable = True
        return ()


class OwnLocationHook:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        cell = StateCellRef(EntityRef(observer.id, observer.type_name), "agent.location")
        value = next_visible.get(cell)
        return () if value is None else (ObservationFact(cell, "equals", value),)


class OwnLocationHookV2:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        cell = StateCellRef(EntityRef(observer.id, observer.type_name), "agent.location")
        if cell not in next_visible:
            return ()
        return (ObservationFact(cell, "equals", next_visible[cell]),)


class ConstantFactsHook:
    def __init__(self, facts):
        self.facts = tuple(facts)
        self.calls = 0

    def __call__(self, prior_visible, next_visible, observer, step_index):
        self.calls += 1
        return self.facts


class RaisingProjectionHook:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        raise RuntimeError("projection boom")


class ListProjectionHook:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        return []


class GeneratorProjectionHook:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        return (item for item in ())


class MappingProjectionHook:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        return {}


class ScalarProjectionHook:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        return "fact"


class WrongTupleItemProjectionHook:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        return ("fact",)


class DuplicateFactProjectionHook:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        cell = StateCellRef(EntityRef(observer.id, "Agent"), "agent.location")
        value = next_visible[cell]
        fact = ObservationFact(cell, "equals", value)
        return (fact, fact)


def _agent_ref(agent_id: str) -> TypedValue:
    return TypedValue("AgentRef", EntityRef(agent_id, "Agent"))


def _service_ref() -> TypedValue:
    return TypedValue("ServiceRef", EntityRef("svc", "Service"))


def _room_ref() -> TypedValue:
    return TypedValue("RoomRef", EntityRef("room", "Room"))


def _cell(entity_id: str, entity_type: str, variable: str) -> StateCellRef:
    return StateCellRef(EntityRef(entity_id, entity_type), variable)


def _forge(instance, **changes):
    forged = object.__new__(type(instance))
    for item in fields(instance):
        object.__setattr__(forged, item.name, changes.get(item.name, getattr(instance, item.name)))
    return forged


def make_projection_domain() -> DomainSpec:
    return DomainSpec(
        domain_id="projection-test",
        version="1",
        entity_types=(EntityTypeSpec("Agent"), EntityTypeSpec("Room"), EntityTypeSpec("Service")),
        value_types=(
            ValueTypeSpec("AgentRef", "entity_ref", entity_type="Agent"),
            ValueTypeSpec("RoomRef", "entity_ref", entity_type="Room"),
            ValueTypeSpec("ServiceRef", "entity_ref", entity_type="Service"),
            ValueTypeSpec("PhaseState", "enum", allowed_values=("ready", "active")),
            ValueTypeSpec("LocationState", "enum", allowed_values=("hall", "vault")),
            ValueTypeSpec("HealthState", "enum", allowed_values=("degraded", "healthy")),
            ValueTypeSpec("AlertState", "bool"),
            ValueTypeSpec("LightState", "enum", allowed_values=("dark", "bright")),
        ),
        state_variables=(
            StateVariableSpec("agent.phase", "Agent", "PhaseState"),
            StateVariableSpec("agent.location", "Agent", "LocationState"),
            StateVariableSpec("service.health", "Service", "HealthState"),
            StateVariableSpec("service.alert", "Service", "AlertState"),
            StateVariableSpec("room.lighting", "Room", "LightState"),
        ),
        event_types=(
            EventTypeSpec("SeedAgentPhase", None, (ParameterSpec("agent", "AgentRef"), ParameterSpec("phase", "PhaseState")), (StateEffectSpec("agent.phase", "agent"),), "seed_agent_phase"),
            EventTypeSpec("SeedAgentLocation", None, (ParameterSpec("agent", "AgentRef"), ParameterSpec("location", "LocationState")), (StateEffectSpec("agent.location", "agent"),), "seed_agent_location"),
            EventTypeSpec("SeedServiceHealth", None, (ParameterSpec("service", "ServiceRef"), ParameterSpec("health", "HealthState")), (StateEffectSpec("service.health", "service"),), "seed_service_health"),
            EventTypeSpec("SeedServiceAlert", None, (ParameterSpec("service", "ServiceRef"), ParameterSpec("alert", "AlertState")), (StateEffectSpec("service.alert", "service"),), "seed_service_alert"),
            EventTypeSpec("SeedRoomLighting", None, (ParameterSpec("room", "RoomRef"), ParameterSpec("lighting", "LightState")), (StateEffectSpec("room.lighting", "room"),), "seed_room_lighting"),
        ),
        action_types=(
            ActionTypeSpec("actor-phase-action", (ParameterSpec("phase", "PhaseState"),)),
            ActionTypeSpec("service-alert-action", (ParameterSpec("service", "ServiceRef"),)),
            ActionTypeSpec("noop-action", ()),
        ),
        decision_types=(
            DecisionTypeSpec("phase-choice", "Agent", "actor-phase-action"),
            DecisionTypeSpec("alert-choice", "Agent", "service-alert-action"),
            DecisionTypeSpec("noop-choice", "Agent", "noop-action"),
        ),
        semantic_hooks=(
            SemanticHookBinding("seed_agent_phase", "seed agent phase", SeedAgentPhaseHook()),
            SemanticHookBinding("seed_agent_location", "seed agent location", SeedAgentLocationHook()),
            SemanticHookBinding("seed_service_health", "seed service health", SeedServiceHealthHook()),
            SemanticHookBinding("seed_service_alert", "seed service alert", SeedServiceAlertHook()),
            SemanticHookBinding("seed_room_lighting", "seed room lighting", SeedRoomLightingHook()),
        ),
    )


def make_projection_story(domain: DomainSpec) -> GenericNarrative:
    entities = (Entity("a1", "Agent"), Entity("a2", "Agent"), Entity("room", "Room"), Entity("svc", "Service"))
    events = (
        NarrativeEvent("e1", 1, "SeedAgentPhase", None, {"agent": _agent_ref("a1"), "phase": TypedValue("PhaseState", "ready")}),
        NarrativeEvent("e2", 2, "SeedAgentPhase", None, {"agent": _agent_ref("a2"), "phase": TypedValue("PhaseState", "ready")}),
        NarrativeEvent("e3", 3, "SeedAgentLocation", None, {"agent": _agent_ref("a1"), "location": TypedValue("LocationState", "hall")}),
        NarrativeEvent("e4", 4, "SeedAgentLocation", None, {"agent": _agent_ref("a2"), "location": TypedValue("LocationState", "vault")}),
        NarrativeEvent("e5", 5, "SeedServiceHealth", None, {"service": _service_ref(), "health": TypedValue("HealthState", "degraded")}),
        NarrativeEvent("e6", 6, "SeedServiceAlert", None, {"service": _service_ref(), "alert": TypedValue("AlertState", True)}),
        NarrativeEvent("e7", 7, "SeedRoomLighting", None, {"room": _room_ref(), "lighting": TypedValue("LightState", "dark")}),
    )
    decisions = (
        Decision("d-a1-phase", 8, "a1", "phase-choice", (_cell("a1", "Agent", "agent.phase"),), (ActionOption("a1-ready", "actor-phase-action", {"phase": TypedValue("PhaseState", "ready")}), ActionOption("a1-active", "actor-phase-action", {"phase": TypedValue("PhaseState", "active")}))),
        Decision("d-a2-alert", 9, "a2", "alert-choice", (_cell("svc", "Service", "service.alert"),), (ActionOption("a2-clear-alert", "service-alert-action", {"service": _service_ref()}),)),
        Decision("d-a1-noop", 10, "a1", "noop-choice", (_cell("a1", "Agent", "agent.phase"),), (ActionOption("a1-wait", "noop-action", {}),)),
        Decision("d-a2-noop", 11, "a2", "noop-choice", (_cell("a2", "Agent", "agent.phase"),), (ActionOption("a2-wait", "noop-action", {}),)),
    )
    return GenericNarrative(domain.domain_id, domain.version, domain.content_hash, entities, events, (), (), (), decisions)


def make_transition_model(domain: DomainSpec) -> WorldTransitionModelSpec:
    return WorldTransitionModelSpec(
        "world-model", "1", domain.domain_id, domain.version, domain.content_hash,
        (
            ActionTransitionSpec("actor-phase-action", (ActionEffectSpec("agent.phase", "actor"),), PhaseTransitionHook()),
            ActionTransitionSpec("service-alert-action", (ActionEffectSpec("service.alert", "argument", "service"),), ClearAlertTransitionHook()),
            ActionTransitionSpec("noop-action", (), NoopTransitionHook()),
        ),
    )


def make_world_step(*, phase: str = "active", clear_alert: bool = False, source_at_time: int | None = None) -> tuple[DomainSpec, GenericNarrative, WorldStepResult]:
    domain = make_projection_domain()
    story = make_projection_story(domain)
    prior = world_state_from_story(story, domain, at_time=source_at_time)
    first_action = "a1-ready" if phase == "ready" else "a1-active"
    second_decision = "d-a2-alert" if clear_alert else "d-a2-noop"
    second_action = "a2-clear-alert" if clear_alert else "a2-wait"
    result = advance_world_step(
        story, domain, prior, make_transition_model(domain),
        (
            ActionIntent("d-a1-phase", first_action, "selection-model", "sha256:" + "1" * 64),
            ActionIntent(second_decision, second_action, "selection-model", "sha256:" + "2" * 64),
        ),
    )
    return domain, story, result


def make_second_noop_step(domain: DomainSpec, story: GenericNarrative, prior: WorldState) -> WorldStepResult:
    return advance_world_step(
        story, domain, prior, make_transition_model(domain),
        (
            ActionIntent("d-a1-noop", "a1-wait", "selection-model", "sha256:" + "3" * 64),
            ActionIntent("d-a2-noop", "a2-wait", "selection-model", "sha256:" + "4" * 64),
        ),
    )


def projection_model(domain: DomainSpec, *specs):
    return ObservationProjectionModelSpec("projection", "1", domain.domain_id, domain.version, domain.content_hash, tuple(specs))


def _spec(hook, *, channel: str = "vision", read=(), emit=()):
    return ObserverProjectionSpec("Agent", channel, tuple(read), tuple(emit), hook)


class NarrativeObservationProjectionTests(unittest.TestCase):
    def require_projection(self) -> None:
        if _PROJECTION_IMPORT_ERROR is not None:
            self.fail("narrative observation projection boundary is missing: " f"{_PROJECTION_IMPORT_ERROR}")

    def test_record_constructors_fail_closed_and_public_records_are_data_not_certification(self):
        self.require_projection()
        cell = _cell("a1", "Agent", "agent.location")
        fake_hash = "sha256:" + "1" * 64
        invalid_calls = (
            lambda: ObservationCapabilitySpec("", "any"),
            lambda: ObservationCapabilitySpec("agent.location", "missing"),
            lambda: ObservationFact(cell, "not_equals", TypedValue("LocationState", "hall")),
            lambda: ObservationFact(cell, "equals", None),
            lambda: ObservationFact(cell, "clear", TypedValue("LocationState", "hall")),
            lambda: ProjectedObservation("", "vision", ObservationFact(cell, "equals", TypedValue("LocationState", "hall")), 1, fake_hash, fake_hash, (), fake_hash),
            lambda: ProjectedObservation("a1", "vision", ObservationFact(cell, "equals", TypedValue("LocationState", "hall")), -1, fake_hash, fake_hash, (), fake_hash),
            lambda: ProjectedObservation("a1", "vision", ObservationFact(cell, "equals", TypedValue("LocationState", "hall")), 1, "bad-hash", fake_hash, (), fake_hash),
            lambda: ProjectedObservation("a1", "vision", ObservationFact(cell, "equals", TypedValue("LocationState", "hall")), 1, fake_hash, fake_hash, (fake_hash, fake_hash), fake_hash),
            lambda: ObservationProjectionResult("model", fake_hash, fake_hash, fake_hash, 1, []),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)):
                    call()
        fact = ObservationFact(cell, "equals", TypedValue("LocationState", "hall"))
        record = ProjectedObservation("a1", "vision", fact, 1, fake_hash, fake_hash, (), fake_hash)
        self.assertEqual(record.source_world_step_hash, fake_hash)
        self.assertEqual(record.content_hash, stable_content_hash(record.to_dict()))

    def test_model_identity_binds_domain_capabilities_channel_and_hook_not_order(self):
        self.require_projection()
        domain = make_projection_domain()
        read = (ObservationCapabilitySpec("room.lighting", "any"), ObservationCapabilitySpec("agent.location", "any"))
        emit = (ObservationCapabilitySpec("agent.location", "any"),)
        first = _spec(OwnLocationHook(), read=read, emit=emit)
        reordered = _spec(OwnLocationHook(), read=tuple(reversed(read)), emit=emit)
        m1 = projection_model(domain, first)
        m2 = projection_model(domain, reordered)
        self.assertEqual(m1.content_hash, m2.content_hash)
        changed_channel = projection_model(domain, _spec(OwnLocationHook(), channel="status", read=read, emit=emit))
        changed_hook = projection_model(domain, _spec(OwnLocationHookV2(), read=read, emit=emit))
        changed_read = projection_model(domain, _spec(OwnLocationHook(), read=(ObservationCapabilitySpec("agent.location", "any"),), emit=emit))
        changed_emit = projection_model(domain, _spec(OwnLocationHook(), read=read, emit=(ObservationCapabilitySpec("agent.location", "observer"),)))
        self.assertNotEqual(m1.content_hash, changed_channel.content_hash)
        self.assertNotEqual(m1.content_hash, changed_hook.content_hash)
        self.assertNotEqual(m1.content_hash, changed_read.content_hash)
        self.assertNotEqual(m1.content_hash, changed_emit.content_hash)
        self.assertNotEqual(m1.content_hash, replace(m1, domain_spec_hash="sha256:" + "2" * 64).content_hash)

    def test_duplicate_declarations_and_semantic_containment_fail_closed(self):
        self.require_projection()
        cap_any = ObservationCapabilitySpec("agent.location", "any")
        cap_observer = ObservationCapabilitySpec("agent.location", "observer")
        with self.assertRaises((TypeError, ValueError)):
            _spec(OwnLocationHook(), read=(cap_any, cap_any), emit=(cap_observer,))
        _spec(OwnLocationHook(), read=(cap_any,), emit=(cap_any, cap_observer))
        _spec(OwnLocationHook(), read=(cap_observer,), emit=(cap_observer,))
        with self.assertRaises((TypeError, ValueError)):
            _spec(OwnLocationHook(), read=(cap_observer,), emit=(cap_any,))
        domain = make_projection_domain()
        spec = _spec(OwnLocationHook(), read=(cap_any,), emit=(cap_any,))
        with self.assertRaises((TypeError, ValueError)):
            projection_model(domain, spec, spec)

    def test_blackout_model_returns_empty_source_bound_result_without_hook_calls(self):
        self.require_projection()
        domain, story, world_step = make_world_step()
        model = ObservationProjectionModelSpec("blackout", "1", domain.domain_id, domain.version, domain.content_hash, ())
        result = project_world_observations(story, domain, world_step, model)
        self.assertEqual(result.observations, ())
        self.assertEqual(result.model_hash, model.content_hash)
        self.assertEqual(result.source_world_step_hash, world_step.content_hash)
        self.assertEqual(result.source_world_state_hash, world_step.next_state.content_hash)
        self.assertEqual(result.step_index, world_step.next_state.step_index)

    def test_undeclared_observer_variable_and_incompatible_observer_scope_reject_before_hook(self):
        self.require_projection()
        domain, story, world_step = make_world_step()
        specs = (
            ObserverProjectionSpec("MissingAgent", "vision", (ObservationCapabilitySpec("agent.location", "any"),), (ObservationCapabilitySpec("agent.location", "any"),), RecordViewsHook()),
            _spec(RecordViewsHook(), read=(ObservationCapabilitySpec("missing.state", "any"),), emit=(ObservationCapabilitySpec("missing.state", "any"),)),
            _spec(RecordViewsHook(), read=(ObservationCapabilitySpec("service.health", "observer"),), emit=(ObservationCapabilitySpec("service.health", "observer"),)),
        )
        for spec in specs:
            hook = spec.projection_hook
            with self.subTest(spec=spec):
                with self.assertRaises(ObservationProjectionError):
                    project_world_observations(story, domain, world_step, projection_model(domain, spec))
                self.assertEqual(hook.calls, [])

    def test_hook_sees_only_read_capability_cells_and_scope_is_exact(self):
        self.require_projection()
        domain, story, world_step = make_world_step()
        own_hook = RecordViewsHook(())
        own = _spec(own_hook, read=(ObservationCapabilitySpec("agent.location", "observer"),), emit=(ObservationCapabilitySpec("agent.location", "observer"),))
        project_world_observations(story, domain, world_step, projection_model(domain, own))
        self.assertEqual(len(own_hook.calls), 2)
        for prior, next_values, observer, step_index in own_hook.calls:
            expected = _cell(observer.id, "Agent", "agent.location")
            self.assertEqual(set(prior), {expected})
            self.assertEqual(set(next_values), {expected})
            self.assertEqual(step_index, world_step.next_state.step_index)
            self.assertNotIn(_cell("svc", "Service", "service.health"), next_values)
            self.assertNotIn(_cell("room", "Room", "room.lighting"), next_values)
        any_hook = RecordViewsHook(())
        any_spec = _spec(any_hook, read=(ObservationCapabilitySpec("agent.location", "any"),), emit=(ObservationCapabilitySpec("agent.location", "observer"),))
        project_world_observations(story, domain, world_step, projection_model(domain, any_spec))
        for _, next_values, _, _ in any_hook.calls:
            self.assertEqual({cell.subject.entity_id for cell in next_values if cell.state_variable == "agent.location"}, {"a1", "a2"})
        mutable_probe = MutableProbeHook()
        mutable_spec = _spec(mutable_probe, read=(ObservationCapabilitySpec("agent.location", "observer"),), emit=(ObservationCapabilitySpec("agent.location", "observer"),))
        project_world_observations(story, domain, world_step, projection_model(domain, mutable_spec))
        self.assertTrue(mutable_probe.prior_is_immutable)
        self.assertTrue(mutable_probe.next_is_immutable)

    def test_readable_nonemittable_and_bad_hook_return_shapes_reject_typed(self):
        self.require_projection()
        domain, story, world_step = make_world_step()
        lighting = _cell("room", "Room", "room.lighting")
        hook = RecordViewsHook((ObservationFact(lighting, "equals", world_step.next_state.values[lighting]),))
        spec = _spec(hook, read=(ObservationCapabilitySpec("room.lighting", "any"), ObservationCapabilitySpec("agent.location", "any")), emit=(ObservationCapabilitySpec("agent.location", "any"),))
        with self.assertRaises(ObservationProjectionError):
            project_world_observations(story, domain, world_step, projection_model(domain, spec))
        self.assertGreater(len(hook.calls), 0)
        for invalid_hook in (ListProjectionHook(), GeneratorProjectionHook(), MappingProjectionHook(), ScalarProjectionHook(), WrongTupleItemProjectionHook(), DuplicateFactProjectionHook(), RaisingProjectionHook()):
            with self.subTest(hook=type(invalid_hook).__name__):
                bad_spec = _spec(invalid_hook, read=(ObservationCapabilitySpec("agent.location", "observer"),), emit=(ObservationCapabilitySpec("agent.location", "observer"),))
                with self.assertRaises(ObservationProjectionError):
                    project_world_observations(story, domain, world_step, projection_model(domain, bad_spec))

    def test_equals_requires_exact_post_step_typed_truth(self):
        self.require_projection()
        domain, story, world_step = make_world_step()
        cell = _cell("a1", "Agent", "agent.location")
        expected = world_step.next_state.values[cell]
        good = _spec(ConstantFactsHook((ObservationFact(cell, "equals", expected),)), read=(ObservationCapabilitySpec("agent.location", "any"),), emit=(ObservationCapabilitySpec("agent.location", "any"),))
        result = project_world_observations(story, domain, world_step, projection_model(domain, good))
        self.assertEqual(result.observations[0].fact.value, expected)
        for value in (TypedValue("LocationState", "vault"), TypedValue("HealthState", "healthy")):
            with self.subTest(value=value):
                bad = _spec(ConstantFactsHook((ObservationFact(cell, "equals", value),)), read=(ObservationCapabilitySpec("agent.location", "any"),), emit=(ObservationCapabilitySpec("agent.location", "any"),))
                with self.assertRaises(ObservationProjectionError):
                    project_world_observations(story, domain, world_step, projection_model(domain, bad))
        cleared_domain, cleared_story, cleared_step = make_world_step(clear_alert=True)
        absent = _cell("svc", "Service", "service.alert")
        absent_spec = ObserverProjectionSpec("Agent", "vision", (ObservationCapabilitySpec("service.alert", "any"),), (ObservationCapabilitySpec("service.alert", "any"),), ConstantFactsHook((ObservationFact(absent, "equals", TypedValue("AlertState", True)),)))
        with self.assertRaises(ObservationProjectionError):
            project_world_observations(cleared_story, cleared_domain, cleared_step, projection_model(cleared_domain, absent_spec))

    def test_clear_requires_explicit_current_step_clear(self):
        self.require_projection()
        domain, story, world_step = make_world_step(clear_alert=True)
        cell = _cell("svc", "Service", "service.alert")
        spec = ObserverProjectionSpec("Agent", "status", (ObservationCapabilitySpec("service.alert", "any"),), (ObservationCapabilitySpec("service.alert", "any"),), ConstantFactsHook((ObservationFact(cell, "clear", None),)))
        result = project_world_observations(story, domain, world_step, projection_model(domain, spec))
        writing = tuple(record for record in world_step.transitions if any(op.kind == "clear" and op.subject_id == "svc" and op.state_variable == "service.alert" for op in record.delta.operations))
        self.assertEqual(len(writing), 1)
        self.assertTrue(all(observation.source_transition_hashes == (writing[0].content_hash,) for observation in result.observations))
        second_step = make_second_noop_step(domain, story, world_step.next_state)
        with self.assertRaises(ObservationProjectionError):
            project_world_observations(story, domain, second_step, projection_model(domain, spec))

    def test_invalid_fact_cell_subject_type_or_variable_rejects(self):
        self.require_projection()
        domain, story, world_step = make_world_step()
        invalid_cells = (_cell("missing", "Agent", "agent.location"), _cell("a1", "Service", "agent.location"), _cell("a1", "Agent", "missing.state"), _cell("svc", "Service", "agent.location"))
        for cell in invalid_cells:
            with self.subTest(cell=cell):
                spec = _spec(ConstantFactsHook((ObservationFact(cell, "equals", TypedValue("LocationState", "hall")),)), read=(ObservationCapabilitySpec("agent.location", "any"),), emit=(ObservationCapabilitySpec("agent.location", "any"),))
                with self.assertRaises(ObservationProjectionError):
                    project_world_observations(story, domain, world_step, projection_model(domain, spec))

    def test_two_observers_can_receive_different_percepts_from_same_world_step(self):
        self.require_projection()
        domain, story, world_step = make_world_step()
        spec = _spec(OwnLocationHook(), read=(ObservationCapabilitySpec("agent.location", "observer"),), emit=(ObservationCapabilitySpec("agent.location", "observer"),))
        result = project_world_observations(story, domain, world_step, projection_model(domain, spec))
        self.assertEqual({observation.observer_id: observation.fact.value.value for observation in result.observations}, {"a1": "hall", "a2": "vault"})
        self.assertTrue(all(observation.source_world_step_hash == world_step.content_hash for observation in result.observations))
        self.assertTrue(all(observation.source_world_state_hash == world_step.next_state.content_hash for observation in result.observations))

    def test_multiple_channels_and_empty_fact_tuple_are_supported(self):
        self.require_projection()
        domain, story, world_step = make_world_step()
        vision = OwnLocationHook()
        status = RecordViewsHook(())
        model = projection_model(domain, _spec(vision, channel="vision", read=(ObservationCapabilitySpec("agent.location", "observer"),), emit=(ObservationCapabilitySpec("agent.location", "observer"),)), _spec(status, channel="status", read=(ObservationCapabilitySpec("agent.location", "observer"),), emit=(ObservationCapabilitySpec("agent.location", "observer"),)))
        result = project_world_observations(story, domain, world_step, model)
        self.assertEqual({observation.channel for observation in result.observations}, {"vision"})
        self.assertEqual(len(status.calls), 2)

    def test_transition_provenance_binds_written_cells_including_same_value_set(self):
        self.require_projection()
        domain, story, world_step = make_world_step(phase="ready")
        cell = _cell("a1", "Agent", "agent.phase")
        spec = _spec(ConstantFactsHook((ObservationFact(cell, "equals", TypedValue("PhaseState", "ready")),)), read=(ObservationCapabilitySpec("agent.phase", "any"),), emit=(ObservationCapabilitySpec("agent.phase", "any"),))
        result = project_world_observations(story, domain, world_step, projection_model(domain, spec))
        writing = tuple(record for record in world_step.transitions if any(op.subject_id == "a1" and op.state_variable == "agent.phase" for op in record.delta.operations))
        self.assertEqual(len(writing), 1)
        matched = tuple(observation for observation in result.observations if observation.fact.cell == cell)
        self.assertTrue(matched)
        self.assertTrue(all(observation.source_transition_hashes == (writing[0].content_hash,) for observation in matched))

    def test_unwritten_persistent_value_has_empty_transition_provenance_and_exact_source_hashes(self):
        self.require_projection()
        domain, story, world_step = make_world_step()
        cell = _cell("room", "Room", "room.lighting")
        value = world_step.next_state.values[cell]
        spec = _spec(ConstantFactsHook((ObservationFact(cell, "equals", value),)), read=(ObservationCapabilitySpec("room.lighting", "any"),), emit=(ObservationCapabilitySpec("room.lighting", "any"),))
        result = project_world_observations(story, domain, world_step, projection_model(domain, spec))
        self.assertTrue(result.observations)
        for observation in result.observations:
            self.assertEqual(observation.source_transition_hashes, ())
            self.assertEqual(observation.source_world_step_hash, world_step.content_hash)
            self.assertEqual(observation.source_world_state_hash, world_step.next_state.content_hash)

    def test_runtime_derives_step_and_lineage_and_hook_cannot_supply_it(self):
        self.require_projection()
        domain, story, world_step = make_world_step()
        cell = _cell("a1", "Agent", "agent.location")
        value = world_step.next_state.values[cell]
        spec = _spec(ConstantFactsHook((ObservationFact(cell, "equals", value),)), read=(ObservationCapabilitySpec("agent.location", "any"),), emit=(ObservationCapabilitySpec("agent.location", "any"),))
        result = project_world_observations(story, domain, world_step, projection_model(domain, spec))
        self.assertTrue(result.observations)
        authored_time = next(decision.logical_time for decision in story.decisions if decision.id == "d-a1-phase")
        for observation in result.observations:
            self.assertEqual(observation.step_index, world_step.next_state.step_index)
            self.assertNotEqual(observation.step_index, authored_time)
            self.assertEqual(observation.source_world_step_hash, world_step.content_hash)
            self.assertEqual(observation.source_world_state_hash, world_step.next_state.content_hash)
            self.assertEqual(observation.projection_spec_hash, spec.content_hash)

    def test_source_identity_and_state_payload_reject_before_hook(self):
        self.require_projection()
        domain, story, world_step = make_world_step()
        bad_prior_unknown = dict(world_step.prior_state.values)
        bad_prior_unknown[_cell("missing", "Agent", "agent.location")] = TypedValue("LocationState", "hall")
        bad_next_type = dict(world_step.next_state.values)
        bad_next_type[_cell("a1", "Agent", "agent.location")] = TypedValue("HealthState", "healthy")
        forged_cases = (
            _forge(world_step, prior_state=_forge(world_step.prior_state, source_story_hash="sha256:" + "9" * 64)),
            _forge(world_step, next_state=_forge(world_step.next_state, domain_spec_hash="sha256:" + "8" * 64)),
            _forge(world_step, prior_state=_forge(world_step.prior_state, values=bad_prior_unknown)),
            _forge(world_step, next_state=_forge(world_step.next_state, values=bad_next_type)),
            _forge(world_step, next_state=_forge(world_step.next_state, step_index=world_step.prior_state.step_index + 2)),
            _forge(world_step, next_state=_forge(world_step.next_state, parent_state_hash="sha256:" + "7" * 64)),
        )
        for forged in forged_cases:
            hook = RecordViewsHook(())
            spec = _spec(hook, read=(ObservationCapabilitySpec("agent.location", "any"),), emit=(ObservationCapabilitySpec("agent.location", "any"),))
            with self.subTest(forged=forged):
                with self.assertRaises(ObservationProjectionError):
                    project_world_observations(story, domain, forged, projection_model(domain, spec))
                self.assertEqual(hook.calls, [])

    def test_transition_decision_action_actor_and_cutoff_reject_before_hook(self):
        self.require_projection()
        domain, story, world_step = make_world_step()
        first = world_step.transitions[0]
        forged_records = (
            _forge(first, intent=_forge(first.intent, decision_id="missing-decision")),
            _forge(first, intent=_forge(first.intent, selected_action="missing-action")),
            _forge(first, actor_id="a2"),
            _forge(first, action=ActionOption(first.action.id, first.action.type_name, {"phase": TypedValue("PhaseState", "active")})),
        )
        for record in forged_records:
            hook = RecordViewsHook(())
            spec = _spec(hook, read=(ObservationCapabilitySpec("agent.location", "any"),), emit=(ObservationCapabilitySpec("agent.location", "any"),))
            forged_step = _forge(world_step, transitions=(record,) + world_step.transitions[1:])
            with self.subTest(record=record):
                with self.assertRaises(ObservationProjectionError):
                    project_world_observations(story, domain, forged_step, projection_model(domain, spec))
                self.assertEqual(hook.calls, [])
        cutoff = 7
        prior = _forge(world_step.prior_state, source_at_time=cutoff)
        next_state = _forge(world_step.next_state, source_at_time=cutoff, parent_state_hash=prior.content_hash)
        cutoff_step = _forge(world_step, prior_state=prior, next_state=next_state)
        hook = RecordViewsHook(())
        spec = _spec(hook, read=(ObservationCapabilitySpec("agent.location", "any"),), emit=(ObservationCapabilitySpec("agent.location", "any"),))
        with self.assertRaises(ObservationProjectionError):
            project_world_observations(story, domain, cutoff_step, projection_model(domain, spec))
        self.assertEqual(hook.calls, [])

    def test_extensional_forgery_duplicate_actor_and_write_collisions_reject_before_hook(self):
        self.require_projection()
        domain, story, world_step = make_world_step()
        first, second = world_step.transitions
        forged_values = dict(world_step.next_state.values)
        forged_values[_cell("a1", "Agent", "agent.location")] = TypedValue("LocationState", "vault")
        duplicate_write_record = _forge(first, delta=StateDelta((StateDeltaOp("set", "a1", "agent.phase", TypedValue("PhaseState", "active")), StateDeltaOp("set", "a1", "agent.phase", TypedValue("PhaseState", "active")))))
        conflicting_record = _forge(second, actor_id="a2", delta=StateDelta((StateDeltaOp("set", "a1", "agent.phase", TypedValue("PhaseState", "ready")),)))
        duplicate_actor_record = _forge(second, actor_id=first.actor_id)
        forged_cases = (
            _forge(world_step, next_state=_forge(world_step.next_state, values=forged_values)),
            _forge(world_step, transitions=(first, duplicate_actor_record)),
            _forge(world_step, transitions=(duplicate_write_record, second)),
            _forge(world_step, transitions=(first, conflicting_record)),
        )
        for forged in forged_cases:
            hook = RecordViewsHook(())
            spec = _spec(hook, read=(ObservationCapabilitySpec("agent.location", "any"),), emit=(ObservationCapabilitySpec("agent.location", "any"),))
            with self.subTest(forged=forged):
                with self.assertRaises(ObservationProjectionError):
                    project_world_observations(story, domain, forged, projection_model(domain, spec))
                self.assertEqual(hook.calls, [])

    def test_projection_order_hash_and_inputs_are_immutable(self):
        self.require_projection()
        domain, story, world_step = make_world_step()
        a1_cell = _cell("a1", "Agent", "agent.location")
        a2_cell = _cell("a2", "Agent", "agent.location")
        facts = (ObservationFact(a2_cell, "equals", world_step.next_state.values[a2_cell]), ObservationFact(a1_cell, "equals", world_step.next_state.values[a1_cell]))
        first = _spec(ConstantFactsHook(facts), channel="vision", read=(ObservationCapabilitySpec("agent.location", "any"),), emit=(ObservationCapabilitySpec("agent.location", "any"),))
        second = _spec(RecordViewsHook(()), channel="status", read=(ObservationCapabilitySpec("agent.location", "any"),), emit=(ObservationCapabilitySpec("agent.location", "any"),))
        before = (story.to_dict(), domain.to_dict(), world_step.to_dict(), world_step.prior_state.to_dict(), world_step.next_state.to_dict())
        result_one = project_world_observations(story, domain, world_step, projection_model(domain, first, second))
        reversed_first = _spec(ConstantFactsHook(tuple(reversed(facts))), channel="vision", read=(ObservationCapabilitySpec("agent.location", "any"),), emit=(ObservationCapabilitySpec("agent.location", "any"),))
        result_two = project_world_observations(story, domain, world_step, projection_model(domain, second, reversed_first))
        self.assertEqual(result_one.content_hash, result_two.content_hash)
        self.assertEqual(tuple((item.observer_id, item.channel, item.fact.cell.subject.entity_type, item.fact.cell.subject.entity_id, item.fact.cell.state_variable) for item in result_one.observations), tuple((item.observer_id, item.channel, item.fact.cell.subject.entity_type, item.fact.cell.subject.entity_id, item.fact.cell.state_variable) for item in result_two.observations))
        self.assertEqual(before, (story.to_dict(), domain.to_dict(), world_step.to_dict(), world_step.prior_state.to_dict(), world_step.next_state.to_dict()))


if __name__ == "__main__":
    unittest.main()
