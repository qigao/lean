# Narrative World Transition V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a narrative-native, provenance-linked world transition layer that executes one or more canonical selected actions against a shared immutable world snapshot and atomically produces the next typed world state.

**Architecture:** Add a sidecar `narrative_dynamics.narrative.world` module rather than changing `GenericNarrative`, `DomainSpec`, `ActionTypeSpec`, replay, or decision semantics. The new module reuses `StateDelta`, seeds `WorldState` from `objective_state()`, re-resolves canonical actors/actions from `Decision`, validates effect capabilities and typed deltas, evaluates every action against one prior snapshot, rejects overlapping actual write sets, and commits pairwise-disjoint deltas atomically.

**Tech Stack:** Python 3 stdlib (`dataclasses`, `collections.abc.Mapping`, `types.MappingProxyType`), existing `stable_content_hash`, existing `measure_implementation`, existing Generic Narrative IR/domain/replay types, `unittest`, GitHub Actions `proof` workflow, Lean 4.32.0 verification gates.

**Spec:** `docs/superpowers/specs/2026-08-25-narrative-world-transition-v1-design.md`

## Global Constraints

- Design base is exactly `proof/narrative-dynamics-v0` at `fa4df6c7607130f7b5364b0e02ef9ce1dcee16ec`.
- Work only on `work/narrative-world-transition-v1`; do not modify `master` or advance `proof/narrative-dynamics-v0` during implementation.
- Reuse existing `StateDelta` / `StateDeltaOp`; do not add a second mutation representation.
- Do not modify `GenericNarrative`, `DomainSpec`, `ActionTypeSpec`, authored event semantics, or their content hashes.
- `WorldTransitionModelSpec` must bind exact domain ID, version, and `DomainSpec.content_hash`.
- `ActionIntent` is provenance only; canonical actor, action type, and action arguments must always be re-resolved from the story.
- Every supplied `WorldState` must be fully revalidated against the canonical story/domain before any transition hook runs.
- Every transition hook sees the same immutable `X_t`; no action may observe another action's delta in the same step.
- Effect declarations are write-capability upper bounds; a hook may write any subset, including the empty set.
- Conflict detection uses actual returned write sets, not declared capabilities; any cross-intent overlap fails with `WorldTransitionConflictError`, including same-value writes.
- One canonical actor may contribute at most one action intent per step.
- Successful non-conflicting batches must be invariant to input-intent order, including transition order, batch hash, next-state hash, and result hash.
- Core V1 has no RNG, initiative, sequential execution, last-writer-wins, or conflict resolver.
- Export exactly 11 new names from `narrative_dynamics.narrative`; export none from top-level `narrative_dynamics`.
- Keep world-transition fixtures local to `tests/test_narrative_world_transition.py`; do not expand `tests/narrative_test_support.py` solely for this feature.
- Complete implementation stays inside these six paths only:
  - `docs/superpowers/specs/2026-08-25-narrative-world-transition-v1-design.md`
  - `docs/superpowers/plans/2026-08-25-narrative-world-transition-v1.md`
  - `narrative_dynamics/narrative/world.py`
  - `narrative_dynamics/narrative/__init__.py`
  - `tests/test_narrative_world_transition.py`
  - `tests/test_narrative_trust_api.py`
- If correct implementation appears to require IR/domain/replay/decision/uncertain/intention/movie fixture/root package/registry/observational/prison/Lean-source changes, stop and re-review the architecture instead of widening scope.
- Integration into `proof/narrative-dynamics-v0` occurs only after exact-head full CI success and explicit human merge approval.

---

## File Structure

- `narrative_dynamics/narrative/world.py` — owns world-sidecar records, model identity, canonical intent resolution, prior-state validation, effect-capability resolution, transition-hook execution, delta validation, simultaneous conflict detection, atomic commit, and world-step lineage.
- `narrative_dynamics/narrative/__init__.py` — imports and exports the exact 11 approved world-transition public names only after semantic GREEN.
- `tests/test_narrative_world_transition.py` — owns all V1 world fixtures and semantic tests; no shared fixture expansion.
- `tests/test_narrative_trust_api.py` — extends the exact narrative public-surface expectation by 11 names while retaining package-root isolation.
- `docs/superpowers/specs/2026-08-25-narrative-world-transition-v1-design.md` — approved architecture; do not change during implementation unless a genuine architecture contradiction is found.
- `docs/superpowers/plans/2026-08-25-narrative-world-transition-v1.md` — this task-by-task implementation contract.

---

### Task 1: Lock the test-only RED boundary

**Files:**
- Create: `tests/test_narrative_world_transition.py`
- Modify: `tests/test_narrative_trust_api.py`
- Verify unchanged/absent: `narrative_dynamics/narrative/world.py`

**Interfaces:**
- Consumes: existing `GenericNarrative`, `DomainSpec`, `StateDelta`, `StateDeltaOp`, `objective_state`, and canonical IR records.
- Produces: a failing contract for the exact 11 world public names and all World Transition V1 invariants; no production interface exists yet.

- [ ] **Step 1: Add a fail-closed world import gate and local typed fixtures**

Create `tests/test_narrative_world_transition.py` with the world imports behind one import gate so unittest discovery continues and every semantic test can fail for the intended missing module rather than aborting discovery:

```python
from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
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
from narrative_dynamics.narrative.replay import objective_state

_WORLD_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.world import (
        ActionEffectSpec,
        ActionIntent,
        ActionTransitionRecord,
        ActionTransitionSpec,
        WorldState,
        WorldStepResult,
        WorldTransitionConflictError,
        WorldTransitionError,
        WorldTransitionModelSpec,
        advance_world_step,
        world_state_from_story,
    )
except ImportError as error:
    _WORLD_IMPORT_ERROR = error


class SeedPhaseHook:
    def __call__(self, prior_state, event):
        actor = event.arguments["agent"].value
        return StateDelta((
            StateDeltaOp(
                "set",
                actor.entity_id,
                "agent.phase",
                event.arguments["phase"],
            ),
        ))


class SeedHealthHook:
    def __call__(self, prior_state, event):
        service = event.arguments["service"].value
        return StateDelta((
            StateDeltaOp(
                "set",
                service.entity_id,
                "service.health",
                event.arguments["health"],
            ),
        ))


class ActorPhaseTransitionHook:
    def __call__(self, prior_state, decision, action):
        return StateDelta((
            StateDeltaOp(
                "set",
                decision.actor_id,
                "agent.phase",
                action.arguments["phase"],
            ),
        ))


class ActorPhaseTransitionHookV2:
    def __call__(self, prior_state, decision, action):
        phase = action.arguments["phase"]
        return StateDelta((
            StateDeltaOp("set", decision.actor_id, "agent.phase", phase),
        ))


class ServiceHealthTransitionHook:
    def __call__(self, prior_state, decision, action):
        service = action.arguments["service"].value
        return StateDelta((
            StateDeltaOp(
                "set",
                service.entity_id,
                "service.health",
                action.arguments["health"],
            ),
        ))


class NoopTransitionHook:
    def __call__(self, prior_state, decision, action):
        return StateDelta(())


class OutsideCapabilityHook:
    def __call__(self, prior_state, decision, action):
        return StateDelta((
            StateDeltaOp(
                "set",
                "svc",
                "service.health",
                TypedValue("HealthState", "recovered"),
            ),
        ))


class DuplicateWriteHook:
    def __call__(self, prior_state, decision, action):
        value = action.arguments["phase"]
        return StateDelta((
            StateDeltaOp("set", decision.actor_id, "agent.phase", value),
            StateDeltaOp("set", decision.actor_id, "agent.phase", value),
        ))


class BadClearHook:
    def __call__(self, prior_state, decision, action):
        return StateDelta((
            StateDeltaOp(
                "clear",
                decision.actor_id,
                "agent.phase",
                TypedValue("PhaseState", "ready"),
            ),
        ))


class UnknownCellHook:
    def __call__(self, prior_state, decision, action):
        return StateDelta((
            StateDeltaOp(
                "set",
                "missing",
                "missing.state",
                TypedValue("PhaseState", "ready"),
            ),
        ))


class WrongValueHook:
    def __call__(self, prior_state, decision, action):
        return StateDelta((
            StateDeltaOp(
                "set",
                decision.actor_id,
                "agent.phase",
                TypedValue("HealthState", "healthy"),
            ),
        ))


class RaisingHook:
    def __call__(self, prior_state, decision, action):
        raise RuntimeError("boom")


class RecordingPhaseHook:
    def __init__(self):
        self.snapshots = []

    def __call__(self, prior_state, decision, action):
        self.snapshots.append(dict(prior_state))
        return StateDelta((
            StateDeltaOp(
                "set",
                decision.actor_id,
                "agent.phase",
                action.arguments["phase"],
            ),
        ))
```

Add one local domain/story builder that gives the tests two agents, one service, authored seed state, distinct/disjoint actor actions, conflicting service actions, a no-op action, and a later decision for cutoff validation:

```python
def make_world_domain() -> DomainSpec:
    return DomainSpec(
        domain_id="world-test",
        version="1",
        entity_types=(EntityTypeSpec("Agent"), EntityTypeSpec("Service")),
        value_types=(
            ValueTypeSpec("AgentRef", "entity_ref", entity_type="Agent"),
            ValueTypeSpec("ServiceRef", "entity_ref", entity_type="Service"),
            ValueTypeSpec("PhaseState", "enum", allowed_values=("idle", "ready", "done")),
            ValueTypeSpec("HealthState", "enum", allowed_values=("healthy", "failed", "recovered")),
        ),
        state_variables=(
            StateVariableSpec("agent.phase", "Agent", "PhaseState"),
            StateVariableSpec("service.health", "Service", "HealthState"),
        ),
        event_types=(
            EventTypeSpec(
                "SeedPhase",
                None,
                (ParameterSpec("agent", "AgentRef"), ParameterSpec("phase", "PhaseState")),
                (StateEffectSpec("agent.phase", "agent"),),
                "seed_phase",
            ),
            EventTypeSpec(
                "SeedHealth",
                None,
                (ParameterSpec("service", "ServiceRef"), ParameterSpec("health", "HealthState")),
                (StateEffectSpec("service.health", "service"),),
                "seed_health",
            ),
        ),
        action_types=(
            ActionTypeSpec("actor-phase-action", (ParameterSpec("phase", "PhaseState"),)),
            ActionTypeSpec(
                "service-health-action",
                (ParameterSpec("service", "ServiceRef"), ParameterSpec("health", "HealthState")),
            ),
            ActionTypeSpec("noop-action", ()),
        ),
        decision_types=(
            DecisionTypeSpec("phase-choice", "Agent", "actor-phase-action"),
            DecisionTypeSpec("service-choice", "Agent", "service-health-action"),
            DecisionTypeSpec("noop-choice", "Agent", "noop-action"),
        ),
        semantic_hooks=(
            SemanticHookBinding("seed_phase", "seed agent phase", SeedPhaseHook()),
            SemanticHookBinding("seed_health", "seed service health", SeedHealthHook()),
        ),
    )


def _agent_cell(agent_id: str) -> StateCellRef:
    return StateCellRef(EntityRef(agent_id, "Agent"), "agent.phase")


def _service_cell() -> StateCellRef:
    return StateCellRef(EntityRef("svc", "Service"), "service.health")


def make_world_story() -> GenericNarrative:
    domain = make_world_domain()
    return GenericNarrative(
        domain.domain_id,
        domain.version,
        domain.content_hash,
        (Entity("bob", "Agent"), Entity("alice", "Agent"), Entity("svc", "Service")),
        (
            NarrativeEvent("e1", 1, "SeedHealth", None, {
                "service": TypedValue("ServiceRef", EntityRef("svc", "Service")),
                "health": TypedValue("HealthState", "failed"),
            }),
            NarrativeEvent("e2", 2, "SeedPhase", None, {
                "agent": TypedValue("AgentRef", EntityRef("bob", "Agent")),
                "phase": TypedValue("PhaseState", "idle"),
            }),
            NarrativeEvent("e3", 3, "SeedPhase", None, {
                "agent": TypedValue("AgentRef", EntityRef("alice", "Agent")),
                "phase": TypedValue("PhaseState", "idle"),
            }),
        ),
        (),
        (),
        (),
        (
            Decision("d-bob-phase", 4, "bob", "phase-choice", (_agent_cell("bob"),), (
                ActionOption("bob-ready", "actor-phase-action", {"phase": TypedValue("PhaseState", "ready")}),
            )),
            Decision("d-alice-phase", 5, "alice", "phase-choice", (_agent_cell("alice"),), (
                ActionOption("alice-done", "actor-phase-action", {"phase": TypedValue("PhaseState", "done")}),
            )),
            Decision("d-bob-service", 6, "bob", "service-choice", (_service_cell(),), (
                ActionOption("bob-recover", "service-health-action", {
                    "service": TypedValue("ServiceRef", EntityRef("svc", "Service")),
                    "health": TypedValue("HealthState", "recovered"),
                }),
            )),
            Decision("d-alice-service", 7, "alice", "service-choice", (_service_cell(),), (
                ActionOption("alice-fail", "service-health-action", {
                    "service": TypedValue("ServiceRef", EntityRef("svc", "Service")),
                    "health": TypedValue("HealthState", "failed"),
                }),
                ActionOption("alice-recover", "service-health-action", {
                    "service": TypedValue("ServiceRef", EntityRef("svc", "Service")),
                    "health": TypedValue("HealthState", "recovered"),
                }),
            )),
            Decision("d-alice-noop", 8, "alice", "noop-choice", (_agent_cell("alice"),), (
                ActionOption("alice-wait", "noop-action", {}),
            )),
        ),
    )
```

- [ ] **Step 2: Add helpers that construct approved sidecar models and provenance intents**

Add these helpers below the fixtures; they are called only after each test invokes `self.require_world()`:

```python
def make_transition_model(*, phase_hook=None, include_service=True, include_noop=True):
    domain = make_world_domain()
    transitions = [
        ActionTransitionSpec(
            "actor-phase-action",
            (ActionEffectSpec("agent.phase", "actor"),),
            ActorPhaseTransitionHook() if phase_hook is None else phase_hook,
        )
    ]
    if include_service:
        transitions.append(ActionTransitionSpec(
            "service-health-action",
            (ActionEffectSpec("service.health", "argument", "service"),),
            ServiceHealthTransitionHook(),
        ))
    if include_noop:
        transitions.append(ActionTransitionSpec("noop-action", (), NoopTransitionHook()))
    return WorldTransitionModelSpec(
        "world-model",
        "1",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        tuple(transitions),
    )


def intent(decision_id: str, action_id: str, *, marker: str = "1"):
    return ActionIntent(
        decision_id,
        action_id,
        "selection-model",
        "sha256:" + marker * 64,
    )
```

- [ ] **Step 3: Add the RED semantic tests required by the spec**

Create one `NarrativeWorldTransitionTests(unittest.TestCase)` with `require_world()` and the following concrete test methods/assertions. Keep these assertions explicit; do not replace them with broad truthiness checks:

```python
class NarrativeWorldTransitionTests(unittest.TestCase):
    def require_world(self) -> None:
        if _WORLD_IMPORT_ERROR is not None:
            self.fail(f"narrative world transition API is missing: {_WORLD_IMPORT_ERROR}")

    def test_world_state_from_story_matches_objective_state_and_lineage(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        state = world_state_from_story(story, domain, at_time=3)
        self.assertEqual(dict(state.values), dict(objective_state(story, domain, at_time=3)))
        self.assertEqual(state.source_story_hash, story.content_hash)
        self.assertEqual(state.source_at_time, 3)
        self.assertEqual(state.step_index, 0)
        self.assertIsNone(state.parent_state_hash)
        self.assertIsNone(state.transition_batch_hash)

    def test_single_canonical_action_changes_world_and_binds_lineage(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        model = make_transition_model()
        result = advance_world_step(story, domain, prior, model, (intent("d-bob-phase", "bob-ready"),))
        self.assertIsInstance(result, WorldStepResult)
        self.assertEqual(result.prior_state.content_hash, prior.content_hash)
        self.assertEqual(result.next_state.values[_agent_cell("bob")], TypedValue("PhaseState", "ready"))
        self.assertEqual(result.next_state.step_index, 1)
        self.assertEqual(result.next_state.parent_state_hash, prior.content_hash)
        self.assertIsNotNone(result.next_state.transition_batch_hash)
        self.assertEqual(result.transitions[0].actor_id, "bob")
        self.assertEqual(result.transitions[0].action.id, "bob-ready")

    def test_forged_prior_state_rejects_before_hook_execution(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        recording = RecordingPhaseHook()
        model = make_transition_model(phase_hook=recording)
        good = world_state_from_story(story, domain)
        forged_values = dict(good.values)
        forged_values[StateCellRef(EntityRef("svc", "Service"), "agent.phase")] = TypedValue("PhaseState", "ready")
        forged = replace(good, values=forged_values)
        with self.assertRaises(WorldTransitionError):
            advance_world_step(story, domain, forged, model, (intent("d-bob-phase", "bob-ready"),))
        self.assertEqual(recording.snapshots, [])

    def test_all_hooks_read_the_same_prior_snapshot(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        recording = RecordingPhaseHook()
        model = make_transition_model(phase_hook=recording)
        prior = world_state_from_story(story, domain)
        advance_world_step(story, domain, prior, model, (
            intent("d-bob-phase", "bob-ready"),
            intent("d-alice-phase", "alice-done", marker="2"),
        ))
        expected = dict(prior.values)
        self.assertEqual(recording.snapshots, [expected, expected])

    def test_disjoint_batch_is_order_invariant(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior, model = world_state_from_story(story, domain), make_transition_model()
        a = intent("d-bob-phase", "bob-ready")
        b = intent("d-alice-phase", "alice-done", marker="2")
        first = advance_world_step(story, domain, prior, model, (a, b))
        second = advance_world_step(story, domain, prior, model, (b, a))
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(first.next_state.content_hash, second.next_state.content_hash)

    def test_different_value_overlap_is_typed_conflict(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior, model = world_state_from_story(story, domain), make_transition_model()
        with self.assertRaises(WorldTransitionConflictError):
            advance_world_step(story, domain, prior, model, (
                intent("d-bob-service", "bob-recover"),
                intent("d-alice-service", "alice-fail", marker="2"),
            ))

    def test_same_value_overlap_is_also_typed_conflict(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior, model = world_state_from_story(story, domain), make_transition_model()
        with self.assertRaises(WorldTransitionConflictError):
            advance_world_step(story, domain, prior, model, (
                intent("d-bob-service", "bob-recover"),
                intent("d-alice-service", "alice-recover", marker="2"),
            ))

    def test_invalid_delta_is_atomic_and_prior_state_is_unchanged(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        before = prior.to_dict()
        bad_model = WorldTransitionModelSpec(
            "bad-world", "1", domain.domain_id, domain.version, domain.content_hash,
            (ActionTransitionSpec("actor-phase-action", (ActionEffectSpec("agent.phase", "actor"),), OutsideCapabilityHook()),),
        )
        with self.assertRaises(WorldTransitionError):
            advance_world_step(story, domain, prior, bad_model, (intent("d-bob-phase", "bob-ready"),))
        self.assertEqual(prior.to_dict(), before)

    def test_duplicate_actor_is_rejected(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior, model = world_state_from_story(story, domain), make_transition_model()
        with self.assertRaises(WorldTransitionError):
            advance_world_step(story, domain, prior, model, (
                intent("d-bob-phase", "bob-ready"),
                intent("d-bob-service", "bob-recover", marker="2"),
            ))

    def test_missing_decision_action_and_uncovered_action_type_reject(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        model = make_transition_model(include_noop=False)
        for bad in (
            intent("missing", "bob-ready"),
            intent("d-bob-phase", "missing"),
            intent("d-alice-noop", "alice-wait"),
        ):
            with self.subTest(bad=bad.to_dict()):
                with self.assertRaises(WorldTransitionError):
                    advance_world_step(story, domain, prior, model, (bad,))

    def test_actor_and_argument_effect_capabilities_are_typed(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        wrong_actor_target = WorldTransitionModelSpec(
            "wrong-actor", "1", domain.domain_id, domain.version, domain.content_hash,
            (ActionTransitionSpec("actor-phase-action", (ActionEffectSpec("service.health", "actor"),), ActorPhaseTransitionHook()),),
        )
        wrong_argument_target = WorldTransitionModelSpec(
            "wrong-argument", "1", domain.domain_id, domain.version, domain.content_hash,
            (ActionTransitionSpec("service-health-action", (ActionEffectSpec("service.health", "argument", "health"),), ServiceHealthTransitionHook()),),
        )
        with self.assertRaises(WorldTransitionError):
            advance_world_step(story, domain, prior, wrong_actor_target, (intent("d-bob-phase", "bob-ready"),))
        with self.assertRaises(WorldTransitionError):
            advance_world_step(story, domain, prior, wrong_argument_target, (intent("d-bob-service", "bob-recover"),))

    def test_outside_capability_duplicate_write_and_invalid_delta_shapes_reject(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        cases = (
            OutsideCapabilityHook(),
            DuplicateWriteHook(),
            BadClearHook(),
            UnknownCellHook(),
            WrongValueHook(),
            RaisingHook(),
        )
        for hook in cases:
            model = WorldTransitionModelSpec(
                "bad", "1", domain.domain_id, domain.version, domain.content_hash,
                (ActionTransitionSpec("actor-phase-action", (ActionEffectSpec("agent.phase", "actor"),), hook),),
            )
            with self.subTest(hook=type(hook).__name__):
                with self.assertRaises(WorldTransitionError):
                    advance_world_step(story, domain, prior, model, (intent("d-bob-phase", "bob-ready"),))

    def test_explicit_noop_succeeds_and_preserves_extensional_values(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior, model = world_state_from_story(story, domain), make_transition_model()
        result = advance_world_step(story, domain, prior, model, (intent("d-alice-noop", "alice-wait"),))
        self.assertEqual(dict(result.next_state.values), dict(prior.values))
        self.assertEqual(result.next_state.step_index, prior.step_index + 1)
        self.assertEqual(result.next_state.parent_state_hash, prior.content_hash)
        self.assertNotEqual(result.next_state.content_hash, prior.content_hash)

    def test_numeric_source_cutoff_rejects_later_decision(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain, at_time=5)
        with self.assertRaises(WorldTransitionError):
            advance_world_step(story, domain, prior, make_transition_model(), (intent("d-bob-service", "bob-recover"),))

    def test_model_identity_binds_domain_effects_and_hook_implementation_not_order(self):
        self.require_world()
        domain = make_world_domain()
        first_phase = ActionTransitionSpec("actor-phase-action", (ActionEffectSpec("agent.phase", "actor"),), ActorPhaseTransitionHook())
        second_phase = ActionTransitionSpec("actor-phase-action", (ActionEffectSpec("agent.phase", "actor"),), ActorPhaseTransitionHookV2())
        service = ActionTransitionSpec("service-health-action", (ActionEffectSpec("service.health", "argument", "service"),), ServiceHealthTransitionHook())
        base = WorldTransitionModelSpec("m", "1", domain.domain_id, domain.version, domain.content_hash, (first_phase, service))
        reversed_model = WorldTransitionModelSpec("m", "1", domain.domain_id, domain.version, domain.content_hash, (service, first_phase))
        changed_domain = replace(base, domain_spec_hash="sha256:" + "f" * 64)
        changed_effect = WorldTransitionModelSpec("m", "1", domain.domain_id, domain.version, domain.content_hash, (
            ActionTransitionSpec("actor-phase-action", (), ActorPhaseTransitionHook()), service,
        ))
        changed_hook = WorldTransitionModelSpec("m", "1", domain.domain_id, domain.version, domain.content_hash, (second_phase, service))
        self.assertEqual(base.content_hash, reversed_model.content_hash)
        self.assertNotEqual(base.content_hash, changed_domain.content_hash)
        self.assertNotEqual(base.content_hash, changed_effect.content_hash)
        self.assertNotEqual(base.content_hash, changed_hook.content_hash)

    def test_transition_and_result_hashes_bind_prior_intent_delta_and_model(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior, model = world_state_from_story(story, domain), make_transition_model()
        first = advance_world_step(story, domain, prior, model, (intent("d-bob-phase", "bob-ready", marker="1"),))
        second = advance_world_step(story, domain, prior, model, (intent("d-bob-phase", "bob-ready", marker="2"),))
        changed_model = replace(model, version="2")
        third = advance_world_step(story, domain, prior, changed_model, (intent("d-bob-phase", "bob-ready", marker="1"),))
        self.assertNotEqual(first.transitions[0].content_hash, second.transitions[0].content_hash)
        self.assertNotEqual(first.next_state.transition_batch_hash, second.next_state.transition_batch_hash)
        self.assertNotEqual(first.content_hash, second.content_hash)
        self.assertNotEqual(first.model_hash, third.model_hash)
        self.assertNotEqual(first.content_hash, third.content_hash)
```

Also add record-construction validation tests for `ActionEffectSpec`, `ActionIntent`, `WorldState`, `ActionTransitionRecord`, and `WorldStepResult`: malformed strings, malformed SHA-256 hashes, duplicate transition action types, invalid `subject_source`, forbidden/missing `subject_argument`, non-callable hooks, invalid mapping key/value types, negative step index, and structurally inconsistent result lineage must fail at construction rather than being silently normalized.

- [ ] **Step 4: Extend the exact narrative public-surface expectation by 11 names**

In `tests/test_narrative_trust_api.py`, add this block to `_EXPECTED_PUBLIC_API` after the intentional-decision names:

```python
    # World transition.
    "ActionEffectSpec",
    "ActionTransitionSpec",
    "WorldTransitionModelSpec",
    "ActionIntent",
    "WorldState",
    "ActionTransitionRecord",
    "WorldStepResult",
    "WorldTransitionError",
    "WorldTransitionConflictError",
    "world_state_from_story",
    "advance_world_step",
```

Do not add imports or names to top-level `narrative_dynamics`; the existing root-isolation loop must remain unchanged.

- [ ] **Step 5: Run the dedicated test file and full Python suite to verify RED**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_narrative_world_transition.py' -v
python3 -m unittest discover -s tests -v
```

Expected RED:

- dedicated world tests fail because `narrative_dynamics.narrative.world` does not exist;
- `test_exact_public_surface_and_root_isolation` fails because the 11 new narrative exports do not exist;
- existing deterministic decision, uncertain belief, intentional decision, movie conformance, runtime, and scientific-validation tests remain green.

If unrelated existing tests fail, fix the RED fixture/test itself before production work; do not proceed on a noisy RED.

- [ ] **Step 6: Commit the test-only RED atomically**

```bash
git add tests/test_narrative_world_transition.py tests/test_narrative_trust_api.py
git commit -m "test: define narrative world transition v1"
```

Verify the commit contains exactly the two test paths and that `narrative_dynamics/narrative/world.py` is still absent.

- [ ] **Step 7: Open/retain a draft PR against the research branch and observe exact-head CI RED**

Create a draft PR from `work/narrative-world-transition-v1` to `proof/narrative-dynamics-v0` if one does not already exist. Record the exact RED commit SHA and verify its `proof` workflow reaches the Python step with all Lean conformance/build/theorem gates green before failing only at the new world/API boundary.

Do not write production code until this CI RED is observed.

---

### Task 2: Add immutable world/model records and initial world-state projection

**Files:**
- Create: `narrative_dynamics/narrative/world.py`
- Test: `tests/test_narrative_world_transition.py`

**Interfaces:**
- Consumes: `stable_content_hash`, `measure_implementation`, `DomainSpec`, `StateDelta`, `StateDeltaOp`, `GenericNarrative`, `Decision`, `ActionOption`, `StateCellRef`, `TypedValue`, `objective_state`.
- Produces: `ActionEffectSpec`, `ActionTransitionSpec`, `WorldTransitionModelSpec`, `ActionIntent`, `WorldState`, `ActionTransitionRecord`, `WorldStepResult`, `WorldTransitionError`, `WorldTransitionConflictError`, `world_state_from_story()`; `advance_world_step()` remains absent until Task 3.

- [ ] **Step 1: Run the dedicated RED file and note which tests can become GREEN from records/projection only**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_narrative_world_transition.py' -v
```

Expected: all tests still fail at the import gate because `world.py` is absent.

- [ ] **Step 2: Create canonical validation/freeze helpers and typed error classes**

Start `narrative_dynamics/narrative/world.py` with focused helpers; do not import decision/uncertain/intention modules:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import re
from types import MappingProxyType

from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.domain import DomainSpec, StateDelta, StateDeltaOp
from narrative_dynamics.narrative.ir import (
    ActionOption,
    Decision,
    Entity,
    EntityRef,
    GenericNarrative,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.replay import objective_state

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_SUBJECT_SOURCES = frozenset({"actor", "argument"})


class WorldTransitionError(ValueError):
    """A canonical selected action could not be executed safely."""


class WorldTransitionConflictError(WorldTransitionError):
    """Two action deltas in one simultaneous step wrote the same cell."""


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _cell_key(cell: StateCellRef) -> tuple[str, str, str]:
    return (cell.subject.entity_type, cell.subject.entity_id, cell.state_variable)


def _freeze_values(value: object) -> Mapping[StateCellRef, TypedValue]:
    if not isinstance(value, Mapping):
        raise TypeError("world values must be a mapping")
    frozen = dict(value)
    if any(not isinstance(cell, StateCellRef) or not isinstance(item, TypedValue) for cell, item in frozen.items()):
        raise TypeError("world values must map StateCellRef to TypedValue")
    return MappingProxyType(frozen)
```

- [ ] **Step 3: Implement the immutable public declarations with canonical `to_dict()` / `content_hash`**

Implement the records with the exact public signatures from the spec. Key constructor rules must be explicit:

```python
@dataclass(frozen=True)
class ActionEffectSpec:
    state_variable: str
    subject_source: str
    subject_argument: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_variable", _text(self.state_variable, label="action effect state variable"))
        source = _text(self.subject_source, label="action effect subject source")
        if source not in _SUBJECT_SOURCES:
            raise ValueError("action effect subject source must be actor or argument")
        if source == "actor":
            if self.subject_argument is not None:
                raise ValueError("actor action effect cannot declare a subject argument")
        else:
            object.__setattr__(self, "subject_argument", _text(self.subject_argument, label="action effect subject argument"))
        object.__setattr__(self, "subject_source", source)

    def to_dict(self) -> dict[str, object]:
        return {
            "state_variable": self.state_variable,
            "subject_source": self.subject_source,
            "subject_argument": self.subject_argument,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class ActionTransitionSpec:
    action_type: str
    effects: tuple[ActionEffectSpec, ...]
    transition_hook: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_type", _text(self.action_type, label="action transition type"))
        effects = tuple(self.effects)
        if any(not isinstance(effect, ActionEffectSpec) for effect in effects):
            raise TypeError("action transition effects must be ActionEffectSpec values")
        keys = tuple((e.state_variable, e.subject_source, e.subject_argument) for e in effects)
        if len(set(keys)) != len(keys):
            raise ValueError("action transition effects must be unique")
        if not callable(self.transition_hook):
            raise TypeError("action transition hook must be callable")
        object.__setattr__(self, "effects", tuple(sorted(effects, key=lambda e: (e.state_variable, e.subject_source, e.subject_argument or ""))))

    def to_dict(self) -> dict[str, object]:
        return {
            "action_type": self.action_type,
            "effects": [effect.to_dict() for effect in self.effects],
            "implementation_identity": measure_implementation(self.transition_hook).manifest_identity(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())
```

Implement `WorldTransitionModelSpec` so its transition tuple is sorted by `action_type`, duplicate action types reject, and `to_dict()` serializes each `ActionTransitionSpec.to_dict()` rather than only storing hook hashes. Implement `ActionIntent` with four trimmed/hash-validated fields and a stable `content_hash`.

Implement `WorldState` exactly as specified, including structural checks:

```python
if not isinstance(self.step_index, int) or isinstance(self.step_index, bool) or self.step_index < 0:
    raise ValueError("world step index must be a non-negative integer")
if self.source_at_time is not None and (
    not isinstance(self.source_at_time, int)
    or isinstance(self.source_at_time, bool)
    or self.source_at_time < 0
):
    raise ValueError("world source cutoff must be a non-negative integer or None")
if self.step_index == 0:
    if self.parent_state_hash is not None or self.transition_batch_hash is not None:
        raise ValueError("step-zero world state cannot have transition lineage")
else:
    object.__setattr__(self, "parent_state_hash", _hash(self.parent_state_hash, label="parent state hash"))
    object.__setattr__(self, "transition_batch_hash", _hash(self.transition_batch_hash, label="transition batch hash"))
```

Canonical `WorldState.to_dict()` must serialize values as a sorted list of `{cell, value}` pairs, never `str(cell)` keys.

Implement structural `ActionTransitionRecord` and `WorldStepResult` records now, even though Task 3 will create them. Their `__post_init__` methods must validate type/shape/lineage fields available without re-running domain semantics, freeze tuples, and expose `to_dict()` / `content_hash`.

- [ ] **Step 4: Implement `world_state_from_story()` by delegating to existing objective replay**

```python
def world_state_from_story(
    story: GenericNarrative,
    domain: DomainSpec,
    *,
    at_time: int | None = None,
) -> WorldState:
    values = objective_state(story, domain, at_time=at_time)
    return WorldState(
        domain_id=domain.domain_id,
        domain_version=domain.version,
        domain_spec_hash=domain.content_hash,
        source_story_hash=story.content_hash,
        source_at_time=at_time,
        step_index=0,
        parent_state_hash=None,
        transition_batch_hash=None,
        values=values,
    )
```

Do not duplicate `_event_replay()` or `DomainSpec.apply_event()`.

- [ ] **Step 5: Run record/projection tests and inspect remaining failures**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_narrative_world_transition.py' -v
```

Expected at this stage:

- record-construction and `world_state_from_story` tests pass;
- tests that require `advance_world_step` still fail because that function is not defined/exported from `world.py` yet;
- no existing test file is modified.

- [ ] **Step 6: Commit records and initial-state projection**

```bash
git add narrative_dynamics/narrative/world.py
git commit -m "feat: add narrative world state records"
```

The commit must contain only `world.py`.

---

### Task 3: Implement canonical single-action execution and fail-closed validation

**Files:**
- Modify: `narrative_dynamics/narrative/world.py`
- Test: `tests/test_narrative_world_transition.py`

**Interfaces:**
- Consumes: all Task 2 records plus canonical story/domain and `WorldTransitionModelSpec`.
- Produces: `advance_world_step(story, domain, prior_state, model, intents) -> WorldStepResult` with correct single-action/no-op behavior and all pre-hook/type/capability validation; multi-agent conflict/order guarantees are finalized in Task 4.

- [ ] **Step 1: Run the dedicated suite and confirm execution tests are RED**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_narrative_world_transition.py' -v
```

Expected: record/projection tests pass, execution tests fail because `advance_world_step` is missing.

- [ ] **Step 2: Add complete prior-state trust-boundary validation before hook execution**

Implement one helper that calls existing narrative validation indirectly through `objective_state(story, domain, at_time=0)` only if needed for canonical validation, or import `validate_narrative` directly and call it once. Then validate exact model/prior/story identities and every public prior-state cell/value:

```python
def _validate_prior_state(
    story: GenericNarrative,
    domain: DomainSpec,
    prior: WorldState,
    model: WorldTransitionModelSpec,
) -> dict[str, Entity]:
    from narrative_dynamics.narrative.domain import validate_narrative

    validate_narrative(story, domain)
    if not isinstance(prior, WorldState):
        raise WorldTransitionError("world execution requires WorldState")
    if not isinstance(model, WorldTransitionModelSpec):
        raise WorldTransitionError("world execution requires WorldTransitionModelSpec")
    domain_identity = (domain.domain_id, domain.version, domain.content_hash)
    if (prior.domain_id, prior.domain_version, prior.domain_spec_hash) != domain_identity:
        raise WorldTransitionError("world state domain identity does not match DomainSpec")
    if (model.domain_id, model.domain_version, model.domain_spec_hash) != domain_identity:
        raise WorldTransitionError("world transition model domain identity does not match DomainSpec")
    if prior.source_story_hash != story.content_hash:
        raise WorldTransitionError("world state source story does not match canonical narrative")

    entities = {entity.id: entity for entity in story.entities}
    try:
        for cell, value in prior.values.items():
            subject = entities.get(cell.subject.entity_id)
            if subject is None or subject.type_name != cell.subject.entity_type:
                raise ValueError("world state cell subject is not canonical")
            state_variable = domain._state_variable(cell.state_variable)
            if state_variable.subject_type != subject.type_name:
                raise ValueError("world state cell subject type does not match state variable")
            domain._value_type(state_variable.value_type).validate(value, entities)
    except (TypeError, ValueError) as error:
        raise WorldTransitionError("world state contains an invalid canonical value") from error
    return entities
```

The forged-prior test must prove the hook was not called.

- [ ] **Step 3: Resolve every intent back to canonical decision/action/model declarations**

Add a private resolved tuple/record or helper returning `(intent, decision, action, transition_spec)` and enforce empty batch, duplicate decision IDs, cutoff, missing decision/action, and uncovered action type before hooks:

```python
def _resolve_intents(story, prior, model, intents):
    values = tuple(intents)
    if not values:
        raise WorldTransitionError("world step requires at least one action intent")
    if any(not isinstance(item, ActionIntent) for item in values):
        raise WorldTransitionError("world step intents must be ActionIntent values")
    if len({item.decision_id for item in values}) != len(values):
        raise WorldTransitionError("world step decision ids must be unique")

    decisions = {decision.id: decision for decision in story.decisions}
    transitions = {transition.action_type: transition for transition in model.transitions}
    resolved = []
    actors = set()
    for item in values:
        decision = decisions.get(item.decision_id)
        if decision is None:
            raise WorldTransitionError("action intent decision is not declared")
        if prior.source_at_time is not None and decision.logical_time > prior.source_at_time:
            raise WorldTransitionError("action intent decision occurs after world source cutoff")
        if decision.actor_id in actors:
            raise WorldTransitionError("one actor may contribute at most one action per world step")
        actors.add(decision.actor_id)
        action = next((candidate for candidate in decision.actions if candidate.id == item.selected_action), None)
        if action is None:
            raise WorldTransitionError("action intent action is not declared by its decision")
        transition = transitions.get(action.type_name)
        if transition is None:
            raise WorldTransitionError("selected action type is not executable by this world model")
        resolved.append((item, decision, action, transition))
    return tuple(resolved)
```

At runtime also verify every `ActionTransitionSpec.action_type` names a declared `DomainSpec` action type before resolving intents; wrap the private lookup error into `WorldTransitionError`.

- [ ] **Step 4: Resolve effect capabilities from canonical actor/arguments**

Implement `_allowed_cells(...)` using only canonical decision/action/domain/story data:

```python
def _allowed_cells(domain, entities, decision, action, transition):
    action_type = domain._action_type(action.type_name)
    parameters = {parameter.name: parameter for parameter in action_type.parameters}
    allowed = set()
    for effect in transition.effects:
        state_variable = domain._state_variable(effect.state_variable)
        if effect.subject_source == "actor":
            subject = entities[decision.actor_id]
        else:
            parameter = parameters.get(effect.subject_argument)
            if parameter is None:
                raise WorldTransitionError("action effect subject argument is not a declared parameter")
            value = action.arguments[effect.subject_argument].value
            if not isinstance(value, EntityRef):
                raise WorldTransitionError("action effect subject argument must contain EntityRef")
            subject = entities.get(value.entity_id)
            if subject is None or subject.type_name != value.entity_type:
                raise WorldTransitionError("action effect subject argument is not canonical")
        if subject.type_name != state_variable.subject_type:
            raise WorldTransitionError("action effect target type does not match state variable")
        allowed.add(StateCellRef(EntityRef(subject.id, subject.type_name), state_variable.name))
    return frozenset(allowed)
```

Do not treat declared capabilities as actual writes.

- [ ] **Step 5: Execute hooks against an immutable snapshot and validate returned deltas**

Build one `MappingProxyType(dict(prior.values))` and pass that snapshot to all hooks. Wrap normal hook exceptions and reject non-`StateDelta` results. Validate every operation fully:

```python
def _validated_delta(domain, entities, allowed, delta):
    if not isinstance(delta, StateDelta):
        raise WorldTransitionError("action transition hook must return StateDelta")
    seen = set()
    for operation in delta.operations:
        try:
            subject = entities.get(operation.subject_id)
            if subject is None:
                raise ValueError("state delta subject is not declared")
            state_variable = domain._state_variable(operation.state_variable)
            cell = StateCellRef(EntityRef(subject.id, subject.type_name), state_variable.name)
            if subject.type_name != state_variable.subject_type:
                raise ValueError("state delta subject type mismatch")
            if cell not in allowed:
                raise ValueError("state delta wrote outside action effect capability")
            if cell in seen:
                raise ValueError("one action delta cannot write one state cell twice")
            seen.add(cell)
            if operation.kind == "clear":
                if operation.value is not None:
                    raise ValueError("clear state delta cannot contain a value")
            else:
                if operation.value is None:
                    raise ValueError("set state delta requires a value")
                domain._value_type(state_variable.value_type).validate(operation.value, entities)
        except (TypeError, ValueError) as error:
            raise WorldTransitionError("action transition produced an invalid state delta") from error
    return delta
```

Hook wrapper:

```python
try:
    raw_delta = transition.transition_hook(snapshot, decision, action)
except Exception as error:
    raise WorldTransitionError("action transition hook failed") from error
```

- [ ] **Step 6: Build one transition record and a single-action atomic next state**

For each validated action, construct `ActionTransitionRecord` with `prior_state_hash=prior.content_hash` and `transition_spec_hash=transition.content_hash`. For the initial GREEN, build the canonical transition tuple, apply its operations to a fresh `dict(prior.values)`, compute:

```python
batch_payload = [record.to_dict() for record in canonical_records]
batch_hash = stable_content_hash(batch_payload)
```

then construct `WorldState(... step_index=prior.step_index + 1, parent_state_hash=prior.content_hash, transition_batch_hash=batch_hash, values=next_values)` and `WorldStepResult(model.model_id, model.content_hash, prior, canonical_records, next_state)`.

Even at single-action stage, never mutate `prior.values`.

- [ ] **Step 7: Run dedicated tests and confirm single-action/fail-closed semantics are GREEN**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_narrative_world_transition.py' -v
```

Expected at this checkpoint:

- initial-state, record, single-action, forged-prior, cutoff, missing/uncovered action, effect-capability, outside-capability, duplicate-write, invalid-shape/type, hook-exception, no-op, identity, and lineage tests pass;
- multi-agent order/conflict/snapshot tests may still expose incomplete batch semantics and are Task 4's RED.

- [ ] **Step 8: Commit canonical single-action execution**

```bash
git add narrative_dynamics/narrative/world.py
git commit -m "feat: execute canonical narrative actions"
```

Do not modify tests to accommodate implementation behavior; tests already encode the approved spec.

---

### Task 4: Complete simultaneous multi-agent snapshot, conflict, and atomic commit semantics

**Files:**
- Modify: `narrative_dynamics/narrative/world.py`
- Test: `tests/test_narrative_world_transition.py`

**Interfaces:**
- Consumes: Task 3's fully validated per-intent transition records.
- Produces: order-invariant simultaneous multi-agent `advance_world_step()` with one shared snapshot, actual-write conflict detection, canonical record ordering, atomic commit, and identical hashes for input permutations.

- [ ] **Step 1: Run the dedicated file and isolate remaining multi-agent RED tests**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_narrative_world_transition.py' -v
```

Expected RED is limited to snapshot/order/conflict/whole-batch behavior; all Task 2-3 tests must remain green.

- [ ] **Step 2: Ensure every hook in a batch receives the same immutable snapshot object/content**

Create the snapshot once before iterating resolved intents:

```python
snapshot = MappingProxyType(dict(prior_state.values))
records = []
for item, decision, action, transition in resolved:
    allowed = _allowed_cells(domain, entities, decision, action, transition)
    try:
        raw_delta = transition.transition_hook(snapshot, decision, action)
    except Exception as error:
        raise WorldTransitionError("action transition hook failed") from error
    delta = _validated_delta(domain, entities, allowed, raw_delta)
    records.append(ActionTransitionRecord(
        item,
        decision.actor_id,
        action,
        transition.content_hash,
        prior_state.content_hash,
        delta,
    ))
```

No copy of a partially updated state may be created until the loop has finished and all deltas are valid.

- [ ] **Step 3: Canonicalize transition records independently of caller intent order**

Sort by the exact approved key:

```python
canonical_records = tuple(sorted(
    records,
    key=lambda record: (
        record.actor_id,
        record.intent.decision_id,
        record.action.id,
    ),
))
```

`ActionTransitionRecord.to_dict()` must serialize its `StateDelta.operations` in canonical state-cell order for batch hashing; if the original returned operation order is retained in the record object, the serialization layer—not hook execution—must canonicalize it so semantically identical disjoint deltas hash identically.

- [ ] **Step 4: Detect conflicts from actual write sets only**

Compute each record's actual set of written canonical cells from its validated delta. Reject any overlap before commit:

```python
owner_by_cell = {}
for record in canonical_records:
    for operation in record.delta.operations:
        subject = entities[operation.subject_id]
        cell = StateCellRef(
            EntityRef(subject.id, subject.type_name),
            operation.state_variable,
        )
        previous = owner_by_cell.get(cell)
        if previous is not None:
            raise WorldTransitionConflictError(
                f"world step writes {cell.state_variable!r} for {cell.subject.entity_id!r} more than once"
            )
        owner_by_cell[cell] = record
```

Do not compare values; same-value overlap is still a conflict. Do not compare declared capability sets; two actions may declare overlapping possible effects but succeed if the actual deltas are disjoint/empty.

- [ ] **Step 5: Commit all pairwise-disjoint deltas atomically in canonical order**

Only after conflict detection succeeds:

```python
next_values = dict(prior_state.values)
for record in canonical_records:
    operations = sorted(
        record.delta.operations,
        key=lambda op: _cell_key(StateCellRef(
            EntityRef(entities[op.subject_id].id, entities[op.subject_id].type_name),
            op.state_variable,
        )),
    )
    for operation in operations:
        subject = entities[operation.subject_id]
        cell = StateCellRef(EntityRef(subject.id, subject.type_name), operation.state_variable)
        if operation.kind == "clear":
            next_values.pop(cell, None)
        else:
            next_values[cell] = operation.value
```

Then compute `batch_hash` from canonical record payloads, create the lineage-bound next `WorldState`, and return `WorldStepResult`.

- [ ] **Step 6: Run all dedicated world tests to semantic GREEN**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_narrative_world_transition.py' -v
```

Expected: all dedicated world-transition tests pass, including:

- same prior snapshot observed by all hooks;
- input-order invariance;
- different-value and same-value overlap typed conflicts;
- atomic prior preservation on any invalid action/delta;
- duplicate actor rejection;
- no-op success;
- exact model/transition/result lineage.

At this point `tests/test_narrative_trust_api.py` is still expected to fail because public exports have intentionally not been wired yet.

- [ ] **Step 7: Run the full Python suite and establish semantic GREEN / export-only RED**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: every test passes except `NarrativeTrustTests.test_exact_public_surface_and_root_isolation`, which reports only the 11 missing world names. Existing deterministic decision, uncertain belief, intentional decision, Knives Out, Matrix, Memento, Story V2 compatibility, scientific runtime, and model-comparison tests must be green.

- [ ] **Step 8: Commit simultaneous multi-agent semantics**

```bash
git add narrative_dynamics/narrative/world.py
git commit -m "feat: add atomic multi-agent world transitions"
```

Observe exact-head CI if the branch PR is open. The expected CI boundary is Python RED only for the 11 not-yet-exported names, with all Lean gates and dedicated world semantics green.

---

### Task 5: Open the scoped public API and complete exact-head regression verification

**Files:**
- Modify: `narrative_dynamics/narrative/__init__.py`
- Verify already modified: `tests/test_narrative_trust_api.py`
- Verify: `tests/test_narrative_world_transition.py`

**Interfaces:**
- Consumes: semantic-GREEN `world.py` from Task 4.
- Produces: exact 11-name `narrative_dynamics.narrative` public surface, unchanged top-level package, and one exact feature head that passes all Python/Lean/story/testimony gates.

- [ ] **Step 1: Verify the pre-export RED is exactly one public-surface failure**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: only `test_exact_public_surface_and_root_isolation` fails and lists the 11 world names as missing. If any semantic world test fails, stop and fix Task 4 before exporting.

- [ ] **Step 2: Import the exact 11 world names into `narrative_dynamics.narrative`**

Add one import block to `narrative_dynamics/narrative/__init__.py`:

```python
from narrative_dynamics.narrative.world import (
    ActionEffectSpec,
    ActionIntent,
    ActionTransitionRecord,
    ActionTransitionSpec,
    WorldState,
    WorldStepResult,
    WorldTransitionConflictError,
    WorldTransitionError,
    WorldTransitionModelSpec,
    advance_world_step,
    world_state_from_story,
)
```

Do not modify `narrative_dynamics/__init__.py`.

- [ ] **Step 3: Add the exact 11 names to narrative `__all__`**

Add exactly:

```python
    "ActionEffectSpec",
    "ActionTransitionSpec",
    "WorldTransitionModelSpec",
    "ActionIntent",
    "WorldState",
    "ActionTransitionRecord",
    "WorldStepResult",
    "WorldTransitionError",
    "WorldTransitionConflictError",
    "world_state_from_story",
    "advance_world_step",
```

Keep all existing exports unchanged.

- [ ] **Step 4: Run dedicated world tests and the full Python suite**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_narrative_world_transition.py' -v
python3 -m unittest discover -s tests -v
```

Expected: both commands exit 0; the exact public-surface/root-isolation test passes, proving all 11 names exist only in `narrative_dynamics.narrative`.

- [ ] **Step 5: Commit scoped exports atomically**

```bash
git add narrative_dynamics/narrative/__init__.py
git commit -m "feat: export narrative world transition api"
```

This commit must modify only `narrative_dynamics/narrative/__init__.py`.

- [ ] **Step 6: Run/observe the exact full `proof` workflow on the export commit**

The repository workflow contract is `.github/workflows/proof.yml`. Verify all of these exact gates on the same feature head:

```bash
export PATH="$HOME/.elan/bin:$PATH"
lake update
lake build NarrativeDynamics.Core.Belief NarrativeDynamics.Core.Drive NarrativeDynamics.Core.Learning
mkdir -p .generated-conformance
generated=.generated-conformance/lean_reference_vectors.json
lake env lean --run NarrativeDynamics/Conformance/ReferenceVectors.lean > "$generated"
cmp conformance/lean_reference_vectors.json "$generated"
lake build
python3 -m unittest discover -s tests -v
lake env lean NarrativeDynamics/Tests/StoryState.lean
lake env lean NarrativeDynamics/Tests/Testimony.lean
```

And verify the workflow's full theorem block exits 0 for:

```text
Drive
Collective
Learning
Belief
Strategy
Epistemic
VectorStrategy
WorldGraph
TemporalGraph
TypedGraph
TypedHypergraph
TypedInference
Closure
Provenance
TruthMaintenance
BeliefSupport
EpistemicBelief
Interpretation
ObservationAdmission
CognitivePipeline
HypothesisCompetition
GroundedHypothesisSpace
GroundedBeliefUpdate
InterpretationCommitment
InterpretationSelection
EpistemicGoal
EpistemicGoalCompetition
GroundedEpistemicGoal
GroundedGoalCovariance
GroundedGoalScoreCovariance
GroundedGoalRankingReversal
GroundedGoalSoftmax
```

For remote GitHub verification, record the exact feature-head SHA, workflow run number/ID, job ID, and each step conclusion. A green prior run on another SHA is not sufficient.

- [ ] **Step 7: Audit the final diff boundary and implementation identity**

Compare the feature branch against `fa4df6c7607130f7b5364b0e02ef9ce1dcee16ec` and require the changed-file set to be exactly:

```text
docs/superpowers/specs/2026-08-25-narrative-world-transition-v1-design.md
docs/superpowers/plans/2026-08-25-narrative-world-transition-v1.md
narrative_dynamics/narrative/world.py
narrative_dynamics/narrative/__init__.py
tests/test_narrative_world_transition.py
tests/test_narrative_trust_api.py
```

Review `world.py` specifically for:

- no objective-state access inside transition hooks beyond initial `world_state_from_story` seeding;
- no mutation of caller-owned `WorldState.values`;
- no intent-provided actor/action payload;
- prior-state validation before hooks;
- same snapshot for all hooks;
- actual-write conflict detection;
- canonical ordering/hashing;
- exact domain/model/story lineage;
- transition hook identity measured through the callable object, not a bare function that resolves to `builtins` under the existing attestation rules.

- [ ] **Step 8: Update the draft PR body/ready state only after exact-head GREEN**

Replace any RED-only PR text with final implementation and verification evidence, including exact head SHA and proof run. Mark the PR ready for review. Do not merge it into `proof/narrative-dynamics-v0` without explicit human approval.

---

## Plan Self-Review Checklist

Before executing Task 1, verify this plan against the approved spec:

- **Spec coverage:** Tasks 1-5 cover all 21 required test classes, public records, sidecar identity, initial objective projection, public prior-state trust boundary, canonical intent resolution, actor/argument effect capabilities, hook attestation, typed delta validation, snapshot isolation, one-action-per-actor, actual-write conflict detection, atomic commit, order invariance, lineage, no-op semantics, cutoff semantics, exact 11-name exports, root isolation, six-file boundary, and all CI gates.
- **Placeholder scan:** The plan contains no `TBD`, `TODO`, `implement later`, `similar to Task N`, or unspecified error-handling step. Every coding step includes concrete interfaces/code/assertions.
- **Type consistency:** Public signatures match the spec exactly: `ActionEffectSpec`, `ActionTransitionSpec`, `WorldTransitionModelSpec`, `ActionIntent`, `WorldState`, `ActionTransitionRecord`, `WorldStepResult`, `world_state_from_story(...)`, and `advance_world_step(...)` use the same field/function names in every task.
- **Scope:** The feature remains one independently testable subsystem: `X_t + canonical selected actions -> X_t+1`. Observation projection, belief refresh, scheduler, stochastic transition, conflict resolver, POMDP planning, and Lean formalization remain out of scope.
