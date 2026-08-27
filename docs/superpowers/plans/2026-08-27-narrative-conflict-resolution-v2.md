# Narrative Conflict Resolution V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic, attested, capability-limited conflict resolution for simultaneous multi-agent world writes while preserving exact resolver-free V1 behavior and lineage.

**Architecture:** Keep all cognitive models and scheduling unchanged. World Transition continues to execute every action hook against one immutable prior snapshot, then builds connected components from actual overlapping write sets; one optional attested `ConflictResolverSpec` resolves each component into one replacement `StateDelta`, after which a global collision check and one atomic world commit produce the next state. Original `ActionTransitionRecord`s remain in lineage and explicit `ConflictResolutionRecord`s explain the effective world mutation.

**Tech Stack:** Python 3 dataclasses, `unittest`, immutable `MappingProxyType` snapshots, existing `stable_content_hash`, existing implementation attestation via `measure_implementation`, GitHub Actions `proof` workflow.

**Spec:** `docs/superpowers/specs/2026-08-27-narrative-conflict-resolution-v2-design.md`

## Global Constraints

- Integrated base is exactly `1ad91cd3782e1da0154e3354737bbebf3f369197` on `proof/narrative-dynamics-v0`; post-merge `proof #1114` is GREEN.
- First implementation-bearing CI must be test-only RED. No production file may change before that RED is verified on the exact test-only head.
- Preserve one immutable prior world snapshot for every action hook and every conflict resolver hook.
- Conflict detection uses actual validated delta write sets, not declared capabilities alone.
- Conflict units are deterministic connected components, not pairwise sequential arbitration.
- One `WorldTransitionModelSpec` may configure at most one optional `ConflictResolverSpec`.
- Resolver output replaces the whole conflicting component's original deltas for final mutation.
- Resolver writes are limited to `union(participant.allowed_write_cells)`.
- No implicit actor priority, resolver precedence, execution-order fallback, or last-writer-wins behavior.
- Resolver-free `WorldTransitionModelSpec.to_dict()` must remain semantically identical to V1 and must not emit `conflict_resolver: null`.
- Resolver-free, conflict-free `WorldStepResult.to_dict()` must omit `conflict_resolutions` and preserve the V1 transition-batch hash payload.
- Existing same-value and different-value overlap still raise `WorldTransitionConflictError` when no resolver is configured.
- Resolver implementation identity is attested and bound into resolver/world/simulation model identity.
- `ConflictResolutionRecord` is canonical data, not independent proof that its resolver hash belongs to an external world model; `advance_world_step()` owns that runtime certification.
- The package root `narrative_dynamics/__init__.py` remains unchanged.
- Do not modify `runtime_reactive.py`, `runtime_intention.py`, `runtime_planning.py`, `runtime_decision_dispatch.py`, `runtime_cognition.py`, `runtime_perception.py`, `observation_projection.py`, `ir.py`, `domain.py`, model-comparison/calibration modules, or Lean sources.
- No RNG, stochastic conflict semantics, multiple resolver routing, mechanism DSL, or held-out model-comparison work belongs in this feature.
- Use strict RED -> GREEN -> atomic commit -> exact-head CI discipline.

## File Structure

- `narrative_dynamics/narrative/conflict.py` — new focused pure-data boundary for `ConflictParticipant`, `ConflictResolutionContext`, `ConflictResolverSpec`, and `ConflictResolutionRecord`; imports IR/domain primitives and attestation helpers but never imports `world.py`.
- `narrative_dynamics/narrative/world.py` — owns `WorldTransitionConflictResolutionError`, optional resolver integration, conflict graph construction, runtime certification, resolver execution, effective mutation assembly, V2 batch lineage, and atomic commit.
- `narrative_dynamics/narrative/__init__.py` — final narrative-package export only; touched after a dedicated public-surface RED.
- `tests/test_narrative_conflict_resolution.py` — new generic resolver/records/component/reference-scenario contract tests.
- `tests/test_narrative_world_transition.py` — V1 compatibility, no-resolver conflict preservation, atomicity, and final-collision regression tests.
- `tests/test_narrative_simulation.py` — heterogeneous scheduler integration using existing Reactive/Intentional/Planning fixtures; scheduler production must remain unchanged.
- `tests/test_narrative_trust_api.py` — final exact five-name narrative public-surface RED/GREEN; package-root isolation remains locked.

---

### Task 1: Authoritative Conflict Resolution V2 Test-Only RED

**Files:**
- Create: `tests/test_narrative_conflict_resolution.py`
- Modify: `tests/test_narrative_world_transition.py`
- Modify: `tests/test_narrative_simulation.py`
- Do not modify production files.

**Interfaces:**
- Consumes existing `ActionIntent`, `ActionTransitionSpec`, `WorldTransitionModelSpec`, `WorldStepResult`, `advance_world_step`, and scheduler V2 APIs.
- Defines the expected future interfaces exactly:
  - `ConflictParticipant(actor_id: str, decision_id: str, action: ActionOption, transition_record_hash: str, transition_spec_hash: str, original_delta: StateDelta, allowed_write_cells: tuple[StateCellRef, ...])`
  - `ConflictResolutionContext(prior_state_hash: str, participants: tuple[ConflictParticipant, ...], conflict_cells: tuple[StateCellRef, ...])`
  - `ConflictResolverSpec(resolver_id: str, version: str, domain_id: str, domain_version: str, domain_spec_hash: str, supported_action_types: tuple[str, ...], resolver_hook: object)`
  - `ConflictResolutionRecord(resolver_id: str, resolver_hash: str, prior_state_hash: str, context: ConflictResolutionContext, resolved_delta: StateDelta)`
  - `WorldTransitionModelSpec(..., conflict_resolver: ConflictResolverSpec | None = None)`
  - `WorldStepResult(..., conflict_resolutions: tuple[ConflictResolutionRecord, ...] = ())`
  - `WorldTransitionConflictResolutionError(WorldTransitionConflictError)`

- [ ] **Step 1: Add a guarded import boundary in the new conflict test module**

Create `tests/test_narrative_conflict_resolution.py` with imports that allow test discovery to continue before the production module exists:

```python
from __future__ import annotations

from dataclasses import fields, replace
import unittest

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.domain import (
    ActionTypeSpec,
    DecisionTypeSpec,
    ParameterSpec,
    StateDelta,
    StateDeltaOp,
    StateVariableSpec,
    ValueTypeSpec,
)
from narrative_dynamics.narrative.ir import (
    ActionOption,
    Decision,
    Entity,
    EntityRef,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.world import (
    ActionEffectSpec,
    ActionTransitionSpec,
    WorldTransitionConflictError,
    WorldTransitionModelSpec,
    advance_world_step,
    world_state_from_story,
)
from tests.test_narrative_world_transition import (
    make_world_domain,
    make_world_story,
    intent,
)

_CONFLICT_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.conflict import (
        ConflictParticipant,
        ConflictResolutionContext,
        ConflictResolutionRecord,
        ConflictResolverSpec,
    )
    from narrative_dynamics.narrative.world import (
        WorldTransitionConflictResolutionError,
    )
except ImportError as error:
    _CONFLICT_IMPORT_ERROR = error


def _hash(label: str) -> str:
    return stable_content_hash({"conflict-test": label})


def _forge(instance, **changes):
    forged = object.__new__(type(instance))
    for item in fields(instance):
        object.__setattr__(
            forged,
            item.name,
            changes.get(item.name, getattr(instance, item.name)),
        )
    return forged


class NarrativeConflictResolutionTests(unittest.TestCase):
    def require_conflict(self) -> None:
        if _CONFLICT_IMPORT_ERROR is not None:
            self.fail(
                "narrative conflict resolution boundary is missing: "
                f"{_CONFLICT_IMPORT_ERROR}"
            )
```

- [ ] **Step 2: Add direct record/identity tests**

Add concrete tests that lock frozen canonical records, full `ActionOption` payloads, and resolver attestation identity:

```python
    def test_conflict_records_are_frozen_canonical_and_bind_full_action_payload(self):
        self.require_conflict()
        action = ActionOption(
            "bid-a",
            "service-health-action",
            {
                "service": TypedValue(
                    "ServiceRef",
                    EntityRef("svc", "Service"),
                ),
                "health": TypedValue("HealthState", "recovered"),
            },
        )
        cell = StateCellRef(EntityRef("svc", "Service"), "service.health")
        delta = StateDelta(
            (
                StateDeltaOp(
                    "set",
                    "svc",
                    "service.health",
                    TypedValue("HealthState", "recovered"),
                ),
            )
        )
        participant = ConflictParticipant(
            "alice",
            "d-alice-service",
            action,
            _hash("transition"),
            _hash("transition-spec"),
            delta,
            (cell,),
        )
        context = ConflictResolutionContext(
            _hash("prior"),
            (
                participant,
                replace(
                    participant,
                    actor_id="bob",
                    decision_id="d-bob-service",
                    transition_record_hash=_hash("transition-b"),
                ),
            ),
            (cell,),
        )
        self.assertEqual(participant.action.arguments, action.arguments)
        self.assertEqual(
            tuple(item.actor_id for item in context.participants),
            ("alice", "bob"),
        )
        self.assertEqual(context.conflict_cells, (cell,))
        self.assertTrue(participant.content_hash.startswith("sha256:"))
        self.assertTrue(context.content_hash.startswith("sha256:"))
        with self.assertRaises((TypeError, ValueError)):
            replace(participant, transition_record_hash="bad")

    def test_conflict_context_rejects_forged_conflict_cells_and_action_payload(self):
        self.require_conflict()
        # Build two valid participants whose original deltas collide on service.health.
        # Then require the public context constructor to reject an empty/wrong
        # conflict-cell set, while runtime world preflight later rejects a forged
        # ActionOption payload before the resolver hook executes.
        ...
```

Replace the final method body above with the actual construction used in the first test; do not keep the ellipsis in the committed file. The exact assertions are:

```python
        with self.assertRaises(ValueError):
            ConflictResolutionContext(prior_hash, participants, ())
        with self.assertRaises(ValueError):
            ConflictResolutionContext(prior_hash, participants, (wrong_cell,))
```

Add two deterministic resolver hook classes so implementation identity can be compared without invoking the hook:

```python
class PreferLexicalActorResolver:
    def __call__(self, snapshot, context):
        winner = min(context.participants, key=lambda item: item.actor_id)
        return winner.original_delta


class PreferLexicalActorResolverV2:
    def __call__(self, snapshot, context):
        ordered = tuple(sorted(context.participants, key=lambda item: item.actor_id))
        return ordered[0].original_delta
```

Then add:

```python
    def test_resolver_identity_binds_configuration_and_implementation(self):
        self.require_conflict()
        domain = make_world_domain()
        first = ConflictResolverSpec(
            "resolver",
            "1",
            domain.domain_id,
            domain.version,
            domain.content_hash,
            ("service-health-action",),
            PreferLexicalActorResolver(),
        )
        reordered = ConflictResolverSpec(
            "resolver",
            "1",
            domain.domain_id,
            domain.version,
            domain.content_hash,
            ("service-health-action",),
            PreferLexicalActorResolver(),
        )
        changed = replace(first, resolver_hook=PreferLexicalActorResolverV2())
        self.assertEqual(first.content_hash, reordered.content_hash)
        self.assertNotEqual(first.content_hash, changed.content_hash)
```

- [ ] **Step 3: Add V1 compatibility tests to `tests/test_narrative_world_transition.py`**

Import `stable_content_hash` is already present in that file. Add tests that explicitly construct the old V1 payload instead of accepting a new `None` field:

```python
    def test_resolver_free_world_model_payload_and_hash_remain_exact_v1_shape(self):
        self.require_world()
        model = make_transition_model()
        expected = {
            "model_id": model.model_id,
            "version": model.version,
            "domain_id": model.domain_id,
            "domain_version": model.domain_version,
            "domain_spec_hash": model.domain_spec_hash,
            "transitions": [item.to_dict() for item in model.transitions],
        }
        self.assertEqual(model.to_dict(), expected)
        self.assertNotIn("conflict_resolver", model.to_dict())
        self.assertNotIn("conflict_resolver_hash", model.to_dict())
        self.assertEqual(model.content_hash, stable_content_hash(expected))

    def test_resolver_free_conflict_free_result_preserves_v1_batch_payload(self):
        self.require_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        result = advance_world_step(
            story,
            domain,
            prior,
            make_transition_model(),
            (intent("d-bob-phase", "bob-ready"),),
        )
        expected_batch = stable_content_hash(
            [item.to_dict() for item in result.transitions]
        )
        self.assertEqual(result.next_state.transition_batch_hash, expected_batch)
        self.assertNotIn("conflict_resolutions", result.to_dict())
```

Keep the existing `test_different_value_overlap_is_typed_conflict` and `test_same_value_overlap_is_also_typed_conflict` unchanged; they are the exact no-resolver behavior locks.

- [ ] **Step 4: Add connected-component and resolver-safety tests**

Use a recording resolver that receives contexts and returns a deterministic delta:

```python
class RecordingResolver:
    def __init__(self, result: StateDelta):
        self.calls = []
        self.result = result

    def __call__(self, snapshot, context):
        self.calls.append((dict(snapshot), context))
        return self.result


class RaisingResolver:
    def __call__(self, snapshot, context):
        raise RuntimeError("resolver boom")
```

Add exact tests with these names and assertions:

```python
    def test_two_way_conflict_resolves_once_against_exact_prior_snapshot(self): ...
    def test_three_way_transitive_overlap_is_one_connected_component(self): ...
    def test_two_independent_conflicts_produce_two_canonical_components(self): ...
    def test_input_reorder_preserves_context_resolution_and_hash(self): ...
    def test_noop_delta_does_not_create_conflict_component(self): ...
    def test_unsupported_action_type_rejects_before_action_or_resolver_hook(self): ...
    def test_resolver_hook_exception_is_typed_and_preserves_exact_cause(self): ...
    def test_resolver_cannot_write_outside_union_capability(self): ...
    def test_resolver_invalid_value_clear_set_and_duplicate_output_reject(self): ...
    def test_forged_action_payload_rejects_before_resolver_hook(self): ...
    def test_bad_component_aborts_entire_step_and_preserves_prior(self): ...
    def test_resolver_vs_nonconflicting_record_collision_rejects(self): ...
    def test_resolution_vs_resolution_collision_rejects(self): ...
```

The committed methods must contain concrete fixture creation and assertions, not ellipses. Use the same `make_conflict_domain()` / `make_conflict_story()` fixture described in Step 5 so every assertion runs through public `advance_world_step()` rather than private helpers.

- [ ] **Step 5: Add one self-contained conflict reference domain/story fixture**

Build the reference fixture by extending the existing world test domain rather than changing production `DomainSpec`:

```python
def make_conflict_domain():
    base = make_world_domain()
    return replace(
        base,
        value_types=base.value_types
        + (
            ValueTypeSpec(
                "BidLevel",
                "enum",
                allowed_values=("8", "11"),
            ),
            ValueTypeSpec(
                "CombatHealth",
                "enum",
                allowed_values=("healthy", "injured"),
            ),
            ValueTypeSpec(
                "CombatStatus",
                "enum",
                allowed_values=(
                    "idle",
                    "attack-success",
                    "attack-failed",
                    "defense-success",
                ),
            ),
        ),
        state_variables=base.state_variables
        + (
            StateVariableSpec("service.owner", "Service", "AgentRef"),
            StateVariableSpec("agent.health", "Agent", "CombatHealth"),
            StateVariableSpec("agent.combat", "Agent", "CombatStatus"),
        ),
        action_types=base.action_types
        + (
            ActionTypeSpec(
                "claim-service-action",
                (ParameterSpec("service", "ServiceRef"),),
            ),
            ActionTypeSpec(
                "bid-service-action",
                (
                    ParameterSpec("service", "ServiceRef"),
                    ParameterSpec("bid", "BidLevel"),
                ),
            ),
            ActionTypeSpec(
                "attack-action",
                (ParameterSpec("target", "AgentRef"),),
            ),
            ActionTypeSpec("defend-action", ()),
        ),
        decision_types=base.decision_types
        + (
            DecisionTypeSpec("claim-choice", "Agent", "claim-service-action"),
            DecisionTypeSpec("bid-choice", "Agent", "bid-service-action"),
            DecisionTypeSpec("attack-choice", "Agent", "attack-action"),
            DecisionTypeSpec("defend-choice", "Agent", "defend-action"),
        ),
    )
```

Define test transition hooks whose write capabilities exactly match their `ActionEffectSpec`s:

```python
class ClaimServiceTransition:
    def __call__(self, snapshot, decision, action):
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    action.arguments["service"].value.entity_id,
                    "service.owner",
                    TypedValue("AgentRef", EntityRef(decision.actor_id, "Agent")),
                ),
            )
        )


class BidServiceTransition(ClaimServiceTransition):
    pass


class AttackTransition:
    def __call__(self, snapshot, decision, action):
        target = action.arguments["target"].value.entity_id
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    target,
                    "agent.health",
                    TypedValue("CombatHealth", "injured"),
                ),
                StateDeltaOp(
                    "set",
                    decision.actor_id,
                    "agent.combat",
                    TypedValue("CombatStatus", "attack-success"),
                ),
            )
        )


class DefendTransition:
    def __call__(self, snapshot, decision, action):
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    decision.actor_id,
                    "agent.health",
                    TypedValue("CombatHealth", "healthy"),
                ),
                StateDeltaOp(
                    "set",
                    decision.actor_id,
                    "agent.combat",
                    TypedValue("CombatStatus", "defense-success"),
                ),
            )
        )
```

`make_conflict_story(domain)` must reuse the base story's `bob`, `alice`, and `svc`, add `carol: Agent`, and append authored decisions for Alice/Bob/Carol claim/bid plus Alice attack Bob and Bob defend. Every action id, argument, and logical time must be explicit and deterministic.

- [ ] **Step 6: Add the three reference-scenario tests**

Add deterministic test-only resolvers:

```python
class ClaimResolver:
    def __call__(self, snapshot, context):
        winner = min(context.participants, key=lambda item: item.actor_id)
        service = winner.action.arguments["service"].value.entity_id
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    service,
                    "service.owner",
                    TypedValue("AgentRef", EntityRef(winner.actor_id, "Agent")),
                ),
            )
        )


class AuctionResolver:
    def __call__(self, snapshot, context):
        ranked = sorted(
            context.participants,
            key=lambda item: (
                -int(item.action.arguments["bid"].value),
                item.actor_id,
            ),
        )
        winner = ranked[0]
        service = winner.action.arguments["service"].value.entity_id
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    service,
                    "service.owner",
                    TypedValue("AgentRef", EntityRef(winner.actor_id, "Agent")),
                ),
            )
        )


class AttackDefenseResolver:
    def __call__(self, snapshot, context):
        attack = next(
            item for item in context.participants
            if item.action.type_name == "attack-action"
        )
        defense = next(
            item for item in context.participants
            if item.action.type_name == "defend-action"
        )
        target = attack.action.arguments["target"].value.entity_id
        if defense.actor_id != target:
            return attack.original_delta
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    target,
                    "agent.health",
                    TypedValue("CombatHealth", "healthy"),
                ),
                StateDeltaOp(
                    "set",
                    attack.actor_id,
                    "agent.combat",
                    TypedValue("CombatStatus", "attack-failed"),
                ),
                StateDeltaOp(
                    "set",
                    defense.actor_id,
                    "agent.combat",
                    TypedValue("CombatStatus", "defense-success"),
                ),
            )
        )
```

Then add:

```python
    def test_competing_claims_preserve_losing_attempt_and_explicit_resolution_lineage(self): ...
    def test_three_party_auction_highest_bid_with_attested_lexical_tie_break(self): ...
    def test_attack_defense_uses_whole_component_replacement_not_partial_raw_delta(self): ...
```

Committed tests must assert original transition count, resolution count, participant hashes, final owner/health/status values, input-order invariance, and that the attack-success raw delta is not partially applied when defense wins.

- [ ] **Step 7: Extend scheduler tests without changing scheduler production**

In `tests/test_narrative_simulation.py`, add a separate guarded conflict import so existing scheduler tests still discover before `conflict.py` exists:

```python
_CONFLICT_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.conflict import ConflictResolverSpec
    from narrative_dynamics.narrative.world import (
        WorldTransitionConflictResolutionError,
    )
except ImportError as error:
    _CONFLICT_IMPORT_ERROR = error
```

Add `ConflictAlertResolver` whose output sets `service.alert=True` and records only `(snapshot, context)`; do not let it see cognition objects.

Extend `make_conflict_case()` with an optional resolver argument and pass it as the final `WorldTransitionModelSpec` argument. Keep the existing `test_world_failure_blocks_projection_and_preserves_prior_state` unchanged when resolver is omitted.

Add:

```python
    def test_heterogeneous_conflict_resolution_preserves_shared_prior_and_projects_resolved_world(self): ...
    def test_two_round_heterogeneous_conflict_resolution_replays_exactly(self): ...
```

The fixture must use all three model kinds already established by `make_heterogeneous_case()` or a directly adapted copy: `a1=intentional`, `a2=reactive`, `a3=planning`. Configure at least two selected actions to write `service.alert`, assert every decision result binds the same prior ledger hash, assert one explicit conflict resolution exists, assert projection binds that resolved `WorldStepResult`, and compare two independently constructed two-round trajectory hashes.

- [ ] **Step 8: Run the focused test modules before committing**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution \
  tests.test_narrative_world_transition \
  tests.test_narrative_simulation -v
```

Expected on the untouched production tree:

- all pre-existing world/simulation tests remain GREEN;
- new V1 resolver-free compatibility tests remain GREEN;
- new conflict tests fail only because `narrative_dynamics.narrative.conflict` and the optional resolver/result fields/error are absent;
- new scheduler conflict-resolution tests fail for the same missing Conflict V2 boundary;
- there must be no unrelated import, syntax, baseline, Lean, cognition, or observation failure.

- [ ] **Step 9: Commit the complete test-only RED**

```bash
git add \
  tests/test_narrative_conflict_resolution.py \
  tests/test_narrative_world_transition.py \
  tests/test_narrative_simulation.py
git commit -m "test: define narrative conflict resolution v2 contract"
```

- [ ] **Step 10: Open/update a Draft PR and obtain authoritative exact-head RED**

Push the branch and open a Draft PR titled `feat: add narrative conflict resolution v2` against `proof/narrative-dynamics-v0`, tracking issue #27. Wait for the PR-triggered `proof` run on the exact test-only head. Record the run number/id and verify the Python failure set is confined to the newly added Conflict V2 tests; do not write production before this gate is accepted.

---

### Task 2: Conflict Records and Resolver Specification GREEN

**Files:**
- Create: `narrative_dynamics/narrative/conflict.py`
- Test: `tests/test_narrative_conflict_resolution.py`

**Interfaces:**
- Consumes: `ActionOption`, `StateCellRef`, `StateDelta`, `measure_implementation`, `stable_content_hash`.
- Produces exactly:
  - `ConflictParticipant`
  - `ConflictResolutionContext`
  - `ConflictResolverSpec`
  - `ConflictResolutionRecord`
- `conflict.py` must not import `narrative_dynamics.narrative.world`.

- [ ] **Step 1: Implement shared validation/payload helpers in `conflict.py`**

```python
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re

from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.domain import StateDelta
from narrative_dynamics.narrative.ir import ActionOption, StateCellRef

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _cell_key(cell: StateCellRef) -> tuple[str, str, str]:
    return (
        cell.subject.entity_type,
        cell.subject.entity_id,
        cell.state_variable,
    )


def _delta_payload(delta: StateDelta) -> dict[str, object]:
    return {
        "operations": [
            operation.to_dict()
            for operation in sorted(
                delta.operations,
                key=lambda item: (item.subject_id, item.state_variable),
            )
        ]
    }
```

- [ ] **Step 2: Implement `ConflictParticipant` with structural self-validation**

```python
@dataclass(frozen=True)
class ConflictParticipant:
    actor_id: str
    decision_id: str
    action: ActionOption
    transition_record_hash: str
    transition_spec_hash: str
    original_delta: StateDelta
    allowed_write_cells: tuple[StateCellRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_id", _text(self.actor_id, label="conflict actor id"))
        object.__setattr__(self, "decision_id", _text(self.decision_id, label="conflict decision id"))
        if not isinstance(self.action, ActionOption):
            raise TypeError("conflict participant action must be ActionOption")
        object.__setattr__(
            self,
            "transition_record_hash",
            _hash(self.transition_record_hash, label="conflict transition record hash"),
        )
        object.__setattr__(
            self,
            "transition_spec_hash",
            _hash(self.transition_spec_hash, label="conflict transition spec hash"),
        )
        if not isinstance(self.original_delta, StateDelta):
            raise TypeError("conflict participant original_delta must be StateDelta")
        cells = tuple(self.allowed_write_cells)
        if any(not isinstance(cell, StateCellRef) for cell in cells):
            raise TypeError("conflict participant allowed cells must be StateCellRef values")
        if len(set(cells)) != len(cells):
            raise ValueError("conflict participant allowed cells must be unique")
        object.__setattr__(self, "allowed_write_cells", tuple(sorted(cells, key=_cell_key)))

        by_operation_key = {
            (cell.subject.entity_id, cell.state_variable): cell
            for cell in self.allowed_write_cells
        }
        for operation in self.original_delta.operations:
            if (operation.subject_id, operation.state_variable) not in by_operation_key:
                raise ValueError("conflict participant original delta exceeds allowed cells")
```

`to_dict()` must include the complete `action.to_dict()`, both hashes, `_delta_payload(original_delta)`, and canonical `allowed_write_cells`; `content_hash` is `stable_content_hash(to_dict())`.

- [ ] **Step 3: Implement exact conflict-cell derivation and `ConflictResolutionContext`**

Derive actual participant writes by matching each original operation to its participant's typed allowed cell. Count cells across participants; conflict cells are exactly those with count >= 2.

```python
def _participant_write_cells(participant: ConflictParticipant) -> tuple[StateCellRef, ...]:
    by_key = {
        (cell.subject.entity_id, cell.state_variable): cell
        for cell in participant.allowed_write_cells
    }
    return tuple(
        sorted(
            {
                by_key[(operation.subject_id, operation.state_variable)]
                for operation in participant.original_delta.operations
            },
            key=_cell_key,
        )
    )


@dataclass(frozen=True)
class ConflictResolutionContext:
    prior_state_hash: str
    participants: tuple[ConflictParticipant, ...]
    conflict_cells: tuple[StateCellRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prior_state_hash",
            _hash(self.prior_state_hash, label="conflict prior state hash"),
        )
        participants = tuple(self.participants)
        if len(participants) < 2:
            raise ValueError("conflict context requires at least two participants")
        if any(not isinstance(item, ConflictParticipant) for item in participants):
            raise TypeError("conflict context participants must be ConflictParticipant values")
        participants = tuple(
            sorted(
                participants,
                key=lambda item: (item.actor_id, item.decision_id, item.action.id),
            )
        )
        keys = tuple((item.actor_id, item.decision_id, item.action.id) for item in participants)
        if len(set(keys)) != len(keys):
            raise ValueError("conflict context participant keys must be unique")

        counts = Counter(
            cell
            for participant in participants
            for cell in _participant_write_cells(participant)
        )
        expected = tuple(sorted((cell for cell, count in counts.items() if count >= 2), key=_cell_key))
        supplied = tuple(sorted(tuple(self.conflict_cells), key=_cell_key))
        if not expected or supplied != expected:
            raise ValueError("conflict context cells must equal exact overlapping writes")
        object.__setattr__(self, "participants", participants)
        object.__setattr__(self, "conflict_cells", supplied)
```

`to_dict()` and `content_hash` must be canonical.

- [ ] **Step 4: Implement `ConflictResolverSpec`**

```python
@dataclass(frozen=True)
class ConflictResolverSpec:
    resolver_id: str
    version: str
    domain_id: str
    domain_version: str
    domain_spec_hash: str
    supported_action_types: tuple[str, ...]
    resolver_hook: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        # validate five textual/hash identity fields using _text/_hash
        action_types = tuple(self.supported_action_types)
        if not action_types:
            raise ValueError("conflict resolver requires supported action types")
        if any(not isinstance(item, str) or not item.strip() or item != item.strip() for item in action_types):
            raise ValueError("conflict resolver action types must be trimmed strings")
        if len(set(action_types)) != len(action_types):
            raise ValueError("conflict resolver action types must be unique")
        if not callable(self.resolver_hook):
            raise TypeError("conflict resolver hook must be callable")
        object.__setattr__(self, "supported_action_types", tuple(sorted(action_types)))

    def to_dict(self) -> dict[str, object]:
        return {
            "resolver_id": self.resolver_id,
            "version": self.version,
            "domain_id": self.domain_id,
            "domain_version": self.domain_version,
            "domain_spec_hash": self.domain_spec_hash,
            "supported_action_types": list(self.supported_action_types),
            "implementation_identity": measure_implementation(
                self.resolver_hook
            ).manifest_identity(),
        }
```

Use `_text`/`_hash` explicitly for every identity field rather than leaving the comment in committed code.

- [ ] **Step 5: Implement `ConflictResolutionRecord` as canonical data**

```python
@dataclass(frozen=True)
class ConflictResolutionRecord:
    resolver_id: str
    resolver_hash: str
    prior_state_hash: str
    context: ConflictResolutionContext
    resolved_delta: StateDelta

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolver_id", _text(self.resolver_id, label="conflict resolution resolver id"))
        object.__setattr__(self, "resolver_hash", _hash(self.resolver_hash, label="conflict resolution resolver hash"))
        object.__setattr__(self, "prior_state_hash", _hash(self.prior_state_hash, label="conflict resolution prior state hash"))
        if not isinstance(self.context, ConflictResolutionContext):
            raise TypeError("conflict resolution context must be ConflictResolutionContext")
        if self.context.prior_state_hash != self.prior_state_hash:
            raise ValueError("conflict resolution context must bind exact prior state")
        if not isinstance(self.resolved_delta, StateDelta):
            raise TypeError("conflict resolution resolved_delta must be StateDelta")
```

Do not attempt to prove external world-model identity here.

- [ ] **Step 6: Run direct record tests**

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_conflict_records_are_frozen_canonical_and_bind_full_action_payload \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_conflict_context_rejects_forged_conflict_cells_and_action_payload \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_resolver_identity_binds_configuration_and_implementation -v
```

Expected: PASS. World integration/reference/scheduler Conflict V2 tests remain RED.

- [ ] **Step 7: Commit records/spec sidecar**

```bash
git add narrative_dynamics/narrative/conflict.py
git commit -m "feat: add narrative conflict resolver records"
```

---

### Task 3: World Model Resolver Configuration and V1 Compatibility GREEN

**Files:**
- Modify: `narrative_dynamics/narrative/world.py`
- Test: `tests/test_narrative_world_transition.py`
- Test: `tests/test_narrative_conflict_resolution.py`

**Interfaces:**
- Consumes `ConflictResolverSpec` from Task 2.
- Produces:
  - `WorldTransitionConflictResolutionError`
  - `WorldTransitionModelSpec.conflict_resolver: ConflictResolverSpec | None = None`
  - resolver declaration/domain validation before any action hook.
- Does not yet apply valid resolver output to create a successful resolved world step.

- [ ] **Step 1: Import conflict records one-way and add the typed error**

In `world.py`:

```python
from narrative_dynamics.narrative.conflict import (
    ConflictParticipant,
    ConflictResolutionContext,
    ConflictResolutionRecord,
    ConflictResolverSpec,
)


class WorldTransitionConflictResolutionError(WorldTransitionConflictError):
    """A declared conflict could not be resolved safely."""
```

Keep the class in `world.py`; do not import `world.py` from `conflict.py`.

- [ ] **Step 2: Append the optional resolver field to `WorldTransitionModelSpec`**

```python
@dataclass(frozen=True)
class WorldTransitionModelSpec:
    model_id: str
    version: str
    domain_id: str
    domain_version: str
    domain_spec_hash: str
    transitions: tuple[ActionTransitionSpec, ...]
    conflict_resolver: ConflictResolverSpec | None = None
```

In `__post_init__`, if configured, require `ConflictResolverSpec` and exact `(domain_id, domain_version, domain_spec_hash)` match with the world model.

- [ ] **Step 3: Preserve the exact V1 `to_dict()` shape when resolver is absent**

Build the existing payload first, then conditionally add only a configured resolver hash:

```python
    def to_dict(self) -> dict[str, object]:
        payload = {
            "model_id": self.model_id,
            "version": self.version,
            "domain_id": self.domain_id,
            "domain_version": self.domain_version,
            "domain_spec_hash": self.domain_spec_hash,
            "transitions": [item.to_dict() for item in self.transitions],
        }
        if self.conflict_resolver is not None:
            payload["conflict_resolver_hash"] = self.conflict_resolver.content_hash
        return payload
```

Never emit a null key.

- [ ] **Step 4: Extend declaration preflight before action hooks**

Extend `_validate_transition_declarations(domain, model)` so it also validates every configured resolver `supported_action_types` through `domain._action_type(...)`. Domain mismatch has already been rejected by the model constructor; undeclared supported types must raise `WorldTransitionError` before `_execute_one()` runs.

Do not attest the resolver hook here; implementation attestation is required immediately before resolver execution, after conflicts are known to exist.

- [ ] **Step 5: Add `_attested_resolver_hash()`**

```python
def _attested_resolver_hash(resolver: ConflictResolverSpec) -> str:
    try:
        return resolver.content_hash
    except ImplementationAttestationUnavailable as error:
        raise WorldTransitionConflictResolutionError(
            "conflict resolver implementation attestation is unavailable"
        ) from error
```

- [ ] **Step 6: Run compatibility/configuration tests**

```bash
python3 -m unittest \
  tests.test_narrative_world_transition.NarrativeWorldTransitionTests.test_resolver_free_world_model_payload_and_hash_remain_exact_v1_shape \
  tests.test_narrative_world_transition.NarrativeWorldTransitionTests.test_resolver_free_conflict_free_result_preserves_v1_batch_payload \
  tests.test_narrative_world_transition.NarrativeWorldTransitionTests.test_different_value_overlap_is_typed_conflict \
  tests.test_narrative_world_transition.NarrativeWorldTransitionTests.test_same_value_overlap_is_also_typed_conflict \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_unsupported_action_type_rejects_before_action_or_resolver_hook -v
```

Expected: PASS. Valid configured conflicts still do not complete successfully yet.

- [ ] **Step 7: Commit resolver configuration boundary**

```bash
git add narrative_dynamics/narrative/world.py
git commit -m "feat: bind conflict resolver to world model"
```

---

### Task 4: Connected Conflict Components and Resolver Execution GREEN

**Files:**
- Modify: `narrative_dynamics/narrative/world.py`
- Test: `tests/test_narrative_conflict_resolution.py`

**Interfaces:**
- Consumes: validated `ActionTransitionRecord`s and configured `ConflictResolverSpec`.
- Produces private deterministic helpers:
  - `_conflict_components(records, entities) -> tuple[tuple[ActionTransitionRecord, ...], ...]`
  - `_build_conflict_context(...) -> ConflictResolutionContext`
  - `_run_conflict_resolver(...) -> ConflictResolutionRecord`
- Resolver hooks receive the same immutable `snapshot` used by action hooks.
- A valid resolution record may be produced, but Task 5 owns effective-delta application and successful resolved `WorldStepResult` construction.

- [ ] **Step 1: Implement deterministic actual-write overlap graph construction**

Reuse `_record_write_cells()` and `_transition_record_key()`; do not inspect declared capabilities to create edges.

```python
def _conflict_components(
    records: tuple[ActionTransitionRecord, ...],
    entities: Mapping[str, Entity],
) -> tuple[tuple[ActionTransitionRecord, ...], ...]:
    canonical = tuple(sorted(records, key=_transition_record_key))
    writes = {
        record.content_hash: frozenset(_record_write_cells(record, entities))
        for record in canonical
    }
    adjacency = {record.content_hash: set() for record in canonical}
    by_hash = {record.content_hash: record for record in canonical}
    for index, left in enumerate(canonical):
        for right in canonical[index + 1:]:
            if writes[left.content_hash] & writes[right.content_hash]:
                adjacency[left.content_hash].add(right.content_hash)
                adjacency[right.content_hash].add(left.content_hash)

    components = []
    seen = set()
    for record in canonical:
        root = record.content_hash
        if root in seen or not adjacency[root]:
            continue
        stack = [root]
        members = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            members.append(by_hash[current])
            stack.extend(sorted(adjacency[current], reverse=True))
        component = tuple(sorted(members, key=_transition_record_key))
        components.append(component)
    return tuple(
        sorted(
            components,
            key=lambda component: tuple(
                _transition_record_key(item) for item in component
            ),
        )
    )
```

No-op records have empty write sets and therefore never get an edge.

- [ ] **Step 2: Build participants by recomputing canonical authored action/effect capability**

Add helpers to locate the canonical `Decision` by `record.intent.decision_id`, assert its actor/action match the transition record, locate the corresponding `ActionTransitionSpec` by `record.action.type_name`, and call the existing `_allowed_cells(...)` again.

Construct each participant exactly from runtime data:

```python
ConflictParticipant(
    actor_id=record.actor_id,
    decision_id=record.intent.decision_id,
    action=record.action,
    transition_record_hash=record.content_hash,
    transition_spec_hash=record.transition_spec_hash,
    original_delta=record.delta,
    allowed_write_cells=tuple(sorted(allowed, key=_cell_key)),
)
```

Reject any mismatch with `WorldTransitionConflictResolutionError` before resolver hook invocation.

- [ ] **Step 3: Build exact `conflict_cells` and context**

For one connected component, count `_record_write_cells(record, entities)` across members and select cells with count >= 2. Create `ConflictResolutionContext(prior.content_hash, participants, conflict_cells)` and reconstruct it once from its public fields to reject constructor-bypassing forgeries before the hook.

- [ ] **Step 4: Refactor delta validation into one pure canonical validator with two typed wrappers**

Extract the body of current `_validated_delta()` into a private helper that raises only `TypeError`/`ValueError`:

```python
def _canonical_validated_delta(
    domain: DomainSpec,
    entities: Mapping[str, Entity],
    allowed: frozenset[StateCellRef],
    delta: object,
) -> StateDelta:
    # same structural/type/capability checks and canonical sorting as V1
```

Then keep the existing action wrapper behavior exactly:

```python
def _validated_delta(...):
    try:
        return _canonical_validated_delta(...)
    except (TypeError, ValueError) as error:
        raise WorldTransitionError(
            "action transition produced an invalid state delta"
        ) from error
```

Add resolver wrapper:

```python
def _validated_resolver_delta(...):
    try:
        return _canonical_validated_delta(...)
    except (TypeError, ValueError) as error:
        raise WorldTransitionConflictResolutionError(
            "conflict resolver produced an invalid state delta"
        ) from error
```

This preserves V1 action error typing while giving resolver failures their required subtype/cause.

- [ ] **Step 5: Execute one resolver hook per component**

```python
def _run_conflict_resolver(
    snapshot,
    domain,
    entities,
    prior,
    resolver,
    context,
) -> ConflictResolutionRecord:
    if any(
        participant.action.type_name not in resolver.supported_action_types
        for participant in context.participants
    ):
        raise WorldTransitionConflictResolutionError(
            "conflict resolver does not support every participant action type"
        )
    resolver_hash = _attested_resolver_hash(resolver)
    try:
        raw_delta = resolver.resolver_hook(snapshot, context)
    except Exception as error:
        raise WorldTransitionConflictResolutionError(
            "conflict resolver hook failed"
        ) from error
    allowed = frozenset(
        cell
        for participant in context.participants
        for cell in participant.allowed_write_cells
    )
    delta = _validated_resolver_delta(
        domain,
        entities,
        allowed,
        raw_delta,
    )
    return ConflictResolutionRecord(
        resolver.resolver_id,
        resolver_hash,
        prior.content_hash,
        context,
        delta,
    )
```

Catch `Exception`, never `BaseException`.

- [ ] **Step 6: Invoke conflict discovery/resolvers from `advance_world_step()` without changing the no-conflict path**

After all action records are built against the shared snapshot:

```python
canonical_records = tuple(sorted(records, key=_transition_record_key))
components = _conflict_components(canonical_records, entities)
if not components:
    return _atomic_result(prior_state, model, canonical_records, entities)
if model.conflict_resolver is None:
    _reject_write_conflicts(canonical_records, entities)
    raise AssertionError("unreachable: unresolved conflict was not rejected")

resolutions = tuple(
    _run_conflict_resolver(
        snapshot,
        domain,
        entities,
        prior_state,
        model.conflict_resolver,
        _build_conflict_context(...),
    )
    for component in components
)
```

At this task boundary it is acceptable for valid resolver cases to remain RED at final commit construction; do not invent partial application. Task 5 will consume `resolutions` atomically.

- [ ] **Step 7: Run component/preflight/error tests**

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_two_way_conflict_resolves_once_against_exact_prior_snapshot \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_three_way_transitive_overlap_is_one_connected_component \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_two_independent_conflicts_produce_two_canonical_components \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_input_reorder_preserves_context_resolution_and_hash \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_noop_delta_does_not_create_conflict_component \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_resolver_hook_exception_is_typed_and_preserves_exact_cause \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_resolver_cannot_write_outside_union_capability \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_resolver_invalid_value_clear_set_and_duplicate_output_reject \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_forged_action_payload_rejects_before_resolver_hook -v
```

Expected: component/context/preflight/typed failure tests PASS. Tests requiring a successful resolved next state/lineage remain RED.

- [ ] **Step 8: Commit component/resolver execution**

```bash
git add narrative_dynamics/narrative/world.py
git commit -m "feat: execute canonical conflict components"
```

---

### Task 5: Atomic Resolved World, Lineage, Reference Scenarios, and Scheduler GREEN

**Files:**
- Modify: `narrative_dynamics/narrative/world.py`
- Test: `tests/test_narrative_conflict_resolution.py`
- Test: `tests/test_narrative_world_transition.py`
- Test: `tests/test_narrative_simulation.py`
- No scheduler production changes expected.

**Interfaces:**
- Consumes: canonical action records, connected components, validated `ConflictResolutionRecord`s from Task 4.
- Produces complete Conflict Resolution V2 behavior:
  - successful resolved `WorldStepResult.conflict_resolutions`
  - V2 transition batch hash only when resolution exists
  - whole-component replacement effective mutations
  - final global collision validation
  - atomic next-state construction
  - exact runtime certification against configured resolver and original transitions.

- [ ] **Step 1: Append `conflict_resolutions` to `WorldStepResult` with empty-field omission**

```python
@dataclass(frozen=True)
class WorldStepResult:
    model_id: str
    model_hash: str
    prior_state: WorldState
    transitions: tuple[ActionTransitionRecord, ...]
    next_state: WorldState
    conflict_resolutions: tuple[ConflictResolutionRecord, ...] = ()
```

In `__post_init__`:

- require every item is `ConflictResolutionRecord`;
- canonicalize by a resolution key derived from the ordered participant transition-record hashes/keys;
- require every resolution `prior_state_hash == prior_state.content_hash`;
- require every participant transition hash exists in `transitions`;
- require participant actor/decision/action/spec/delta payload equals the referenced transition record;
- require no transition appears in more than one resolution;
- require all non-empty resolutions have one common `(resolver_id, resolver_hash)` internal identity;
- validate `next_state.transition_batch_hash` with the updated batch hash function.

`to_dict()` must retain the V1 payload and only add:

```python
if self.conflict_resolutions:
    payload["conflict_resolutions"] = [
        item.to_dict() for item in self.conflict_resolutions
    ]
```

- [ ] **Step 2: Extend `_transition_batch_hash()` without changing the empty-resolution path**

```python
def _transition_batch_hash(
    transitions: tuple[ActionTransitionRecord, ...],
    conflict_resolutions: tuple[ConflictResolutionRecord, ...] = (),
) -> str:
    canonical_transitions = tuple(sorted(transitions, key=_transition_record_key))
    if not conflict_resolutions:
        return stable_content_hash(
            [item.to_dict() for item in canonical_transitions]
        )
    canonical_resolutions = tuple(
        sorted(conflict_resolutions, key=_conflict_resolution_key)
    )
    return stable_content_hash(
        {
            "transitions": [item.to_dict() for item in canonical_transitions],
            "conflict_resolutions": [
                item.to_dict() for item in canonical_resolutions
            ],
        }
    )
```

The first branch must be byte-for-byte the V1 hashing payload.

- [ ] **Step 3: Compute the effective mutation set by dropping every conflicting raw delta**

Build a set of transition hashes participating in any resolution. Effective sources are:

1. original delta for every transition whose hash is absent from that set;
2. one `resolved_delta` per resolution.

Never apply an original delta from a conflicting participant after resolution.

- [ ] **Step 4: Add final global collision validation over effective deltas**

Create a helper that converts each effective delta operation to `StateCellRef` using canonical entities and raises `WorldTransitionConflictResolutionError` whenever two effective sources write the same cell.

This check runs after every resolver output validates and before `next_values` is changed.

- [ ] **Step 5: Runtime-certify every resolution against the configured world model**

Immediately before successful next-state construction, verify:

```python
resolver = model.conflict_resolver
assert resolver is not None
resolver_hash = _attested_resolver_hash(resolver)
for resolution, component in zip(resolutions, components, strict=True):
    if resolution.resolver_id != resolver.resolver_id:
        raise WorldTransitionConflictResolutionError(...)
    if resolution.resolver_hash != resolver_hash:
        raise WorldTransitionConflictResolutionError(...)
    if resolution.prior_state_hash != prior.content_hash:
        raise WorldTransitionConflictResolutionError(...)
    if tuple(
        participant.transition_record_hash
        for participant in resolution.context.participants
    ) != tuple(record.content_hash for record in component):
        raise WorldTransitionConflictResolutionError(...)
```

Use explicit messages in committed code; do not keep ellipses.

- [ ] **Step 6: Replace `_atomic_result()` with a backward-compatible optional-resolution path**

Keep the no-resolution behavior exactly V1. For non-empty resolutions:

- skip `_reject_write_conflicts()` on raw records because the overlaps are explicitly resolved;
- validate effective mutations globally;
- apply effective mutations in canonical source order;
- compute V2 batch hash with original transitions + resolutions;
- build exactly one `WorldState`;
- return `WorldStepResult(..., conflict_resolutions=resolutions)`.

No intermediate `WorldState` may be constructed.

- [ ] **Step 7: Make `advance_world_step()` call the completed atomic path**

The final structure must be:

```python
records = tuple(...all action executions...)
components = _conflict_components(records, entities)
if not components:
    return _atomic_result(prior_state, model, records, entities)
if model.conflict_resolver is None:
    _reject_write_conflicts(records, entities)
    raise AssertionError("unreachable")
resolutions = tuple(...one resolver execution per component...)
return _atomic_result(
    prior_state,
    model,
    records,
    entities,
    conflict_resolutions=resolutions,
)
```

- [ ] **Step 8: Run all Conflict V2 world/reference tests**

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution \
  tests.test_narrative_world_transition -v
```

Expected: all tests in both modules PASS, including:

- V1 payload/hash preservation;
- same/different no-resolver conflict rejection;
- two-way/transitive/independent components;
- input order invariance;
- exact ActionOption argument visibility;
- unsupported type preflight;
- typed/capability/output failures;
- entire-step atomic abort;
- final collision rejection;
- competing claims;
- three-party auction with explicit lexical tie-break;
- attack/defense whole-component replacement.

- [ ] **Step 9: Run scheduler integration tests and prove production scheduler is unchanged**

```bash
python3 -m unittest tests.test_narrative_simulation -v
```

Expected: all simulation tests PASS, including the two new heterogeneous conflict-resolution tests. Verify `git diff 1ad91cd3782e1da0154e3354737bbebf3f369197 -- narrative_dynamics/narrative/simulation.py` is empty.

- [ ] **Step 10: Run the full Python suite before committing the core GREEN**

```bash
python3 -m unittest discover -s tests -v
```

Expected: `OK`. If the environment cannot execute the full suite locally, push only after focused suites pass and use exact-head GitHub Actions as authoritative evidence; do not claim GREEN from static inspection.

- [ ] **Step 11: Commit atomic resolution integration**

```bash
git add narrative_dynamics/narrative/world.py
git commit -m "feat: resolve simultaneous world conflicts atomically"
```

- [ ] **Step 12: Obtain exact-head core GREEN**

Wait for the PR-triggered `proof` run on this exact commit. Require Lean dependency/conformance/build/theorem, Python numerical tests, Story, and Testimony all success. Record the Python test count and run id. Do not proceed to exports if any unrelated suite regresses.

---

### Task 6: Exact Narrative Public-Surface Test-Only RED

**Files:**
- Modify: `tests/test_narrative_trust_api.py`
- Do not modify `narrative_dynamics/narrative/__init__.py` yet.

**Interfaces:**
- Adds exactly five expected names to the narrative package surface:
  1. `ConflictParticipant`
  2. `ConflictResolutionContext`
  3. `ConflictResolverSpec`
  4. `ConflictResolutionRecord`
  5. `WorldTransitionConflictResolutionError`
- Package root remains isolated.

- [ ] **Step 1: Extend only `_EXPECTED_PUBLIC_API` in the narrative trust test**

Insert the five names next to existing world-transition names:

```python
    "ConflictParticipant",
    "ConflictResolutionContext",
    "ConflictResolverSpec",
    "ConflictResolutionRecord",
    "WorldTransitionConflictResolutionError",
```

Do not relax the root-isolation assertions and do not add any other public name.

- [ ] **Step 2: Run only the trust API test**

```bash
python3 -m unittest \
  tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation -v
```

Expected: exactly one failure whose missing set is exactly the five Conflict V2 names. No production import should exist yet.

- [ ] **Step 3: Commit the test-only public-surface RED**

```bash
git add tests/test_narrative_trust_api.py
git commit -m "test: lock narrative conflict resolution surface"
```

- [ ] **Step 4: Obtain authoritative exact-head public-surface RED**

Wait for PR-triggered `proof` on this exact test-only head. Require all non-trust Python tests and Lean gates to remain GREEN and the sole Python failure to be `test_exact_public_surface_and_root_isolation` with exactly the five missing names.

---

### Task 7: Export Five Names and Final Exact-Head GREEN

**Files:**
- Modify: `narrative_dynamics/narrative/__init__.py`
- Test: `tests/test_narrative_trust_api.py`

**Interfaces:**
- Exports exactly the four data/spec records from `conflict.py` and the one error from `world.py`.
- Does not export anything from package root.

- [ ] **Step 1: Add the conflict import block to `narrative_dynamics/narrative/__init__.py`**

Place before/near the world import block:

```python
from narrative_dynamics.narrative.conflict import (
    ConflictParticipant,
    ConflictResolutionContext,
    ConflictResolutionRecord,
    ConflictResolverSpec,
)
```

`WorldTransitionConflictResolutionError` must be imported from the existing `world` import block, not from `conflict.py`.

- [ ] **Step 2: Add exactly five `__all__` entries**

```python
    "ConflictParticipant",
    "ConflictResolutionContext",
    "ConflictResolverSpec",
    "ConflictResolutionRecord",
    "WorldTransitionConflictResolutionError",
```

Do not touch `narrative_dynamics/__init__.py`.

- [ ] **Step 3: Run focused public/feature tests**

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution \
  tests.test_narrative_world_transition \
  tests.test_narrative_simulation \
  tests.test_narrative_trust_api -v
```

Expected: PASS.

- [ ] **Step 4: Run full Python suite**

```bash
python3 -m unittest discover -s tests -v
```

Expected: `OK`.

- [ ] **Step 5: Commit exports atomically**

```bash
git add narrative_dynamics/narrative/__init__.py
git commit -m "feat: export narrative conflict resolution surface"
```

- [ ] **Step 6: Verify final scope before accepting CI**

Compare integrated base `1ad91cd3782e1da0154e3354737bbebf3f369197` to final feature head. The only allowed feature paths are:

```text
docs/superpowers/specs/2026-08-27-narrative-conflict-resolution-v2-design.md
docs/superpowers/plans/2026-08-27-narrative-conflict-resolution-v2.md
narrative_dynamics/narrative/conflict.py
narrative_dynamics/narrative/world.py
narrative_dynamics/narrative/__init__.py
tests/test_narrative_conflict_resolution.py
tests/test_narrative_world_transition.py
tests/test_narrative_simulation.py
tests/test_narrative_trust_api.py
```

Reject any drift in family cognition, dispatch, scheduler production, observation/perception, IR/domain, model comparison, package root, or Lean sources.

- [ ] **Step 7: Obtain final authoritative exact-head PR GREEN**

Wait for the PR-triggered `proof` run on the exact final feature head. Require:

- Resolve Lean dependencies: success
- Lean Python conformance vectors: success
- Build Lean library: success
- Lean theorem tests: success
- Python numerical tests: success, with final test count recorded
- Narrative story theorem tests: success
- Narrative testimony theorem tests: success

Do not mark the PR ready or merge before this exact-head run is complete/success.

- [ ] **Step 8: Update PR evidence and roadmap only after final GREEN**

Update the Draft PR body with authoritative RED/GREEN run numbers and exact commit SHAs. In issue #27, mark all seven Conflict Resolution V2 checklist items complete only after the final exact-head GREEN. Do not close issue #27 because held-out cognitive model comparison, stochastic world/observation, and P2 identification work remain open.

- [ ] **Step 9: Present integration choice**

Use `superpowers:finishing-a-development-branch`. The human chooses whether to merge; never infer merge authorization from test success.
