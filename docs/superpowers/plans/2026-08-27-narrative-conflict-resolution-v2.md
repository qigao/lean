# Narrative Conflict Resolution V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic, attested, capability-limited resolution for simultaneous multi-agent world conflicts while preserving resolver-free World Transition V1 behavior and lineage exactly.

**Architecture:** Every action hook still runs against one immutable prior world snapshot and yields one validated `ActionTransitionRecord`. The world layer then builds connected components from actual overlapping delta write sets, invokes one optional attested `ConflictResolverSpec` once per component, validates one whole-component replacement `StateDelta`, checks the final effective mutation set globally for collisions, and constructs one atomic next `WorldState`. Original transition records remain in lineage; explicit resolution records explain why effective world mutation differs from attempted deltas.

**Tech Stack:** Python 3 dataclasses, `unittest`, existing `stable_content_hash`, existing `measure_implementation` attestation, immutable `MappingProxyType` snapshots, GitHub Actions `proof` workflow.

**Spec:** `docs/superpowers/specs/2026-08-27-narrative-conflict-resolution-v2-design.md`

## Global Constraints

- Integrated base is exactly `1ad91cd3782e1da0154e3354737bbebf3f369197` on `proof/narrative-dynamics-v0`; post-merge `proof #1114` is GREEN.
- The first implementation-bearing CI must be test-only RED. No production file changes before that exact-head RED is verified.
- Preserve one immutable prior world snapshot for every action hook and every resolver hook.
- Detect conflicts from actual validated delta writes, not declared capabilities alone.
- Resolve deterministic connected components, never sequential pairs.
- A world model may configure one optional `ConflictResolverSpec` only.
- A resolver returns one replacement `StateDelta` for the whole connected component.
- Resolver output is limited to `union(participant.allowed_write_cells)`.
- No implicit actor priority, resolver precedence, execution-order fallback, or last-writer-wins behavior.
- Resolver-free `WorldTransitionModelSpec.to_dict()` must not emit any resolver field and must preserve the V1 hash payload.
- Resolver-free, conflict-free `WorldStepResult.to_dict()` must omit `conflict_resolutions` and preserve the V1 transition-batch hash payload.
- Existing same-value and different-value overlaps still raise `WorldTransitionConflictError` without a resolver.
- Resolver implementation identity is attested and bound into resolver, world-model, and transitive simulation identity.
- `ConflictResolutionRecord` is canonical data. `advance_world_step()` owns cross-object runtime certification against the configured resolver and exact transition records.
- Keep `narrative_dynamics/__init__.py` unchanged.
- Do not modify `runtime_reactive.py`, `runtime_intention.py`, `runtime_planning.py`, `runtime_decision_dispatch.py`, `runtime_cognition.py`, `runtime_perception.py`, `observation_projection.py`, `ir.py`, `domain.py`, model-comparison/calibration modules, or Lean sources.
- Do not add RNG, stochastic conflict rules, multiple resolver routing, mechanism DSL, or held-out comparison work.
- Use RED -> GREEN -> atomic commit -> exact-head CI for every implementation slice.

## File Map

- Create `narrative_dynamics/narrative/conflict.py`: pure conflict records and resolver specification; it must not import `world.py`.
- Modify `narrative_dynamics/narrative/world.py`: resolver error type, optional resolver field, component construction, resolver execution, effective mutation validation, lineage, atomic commit.
- Modify `narrative_dynamics/narrative/__init__.py`: final five-name export only after a dedicated public-surface RED.
- Create `tests/test_narrative_conflict_resolution.py`: generic records, component, safety, and reference-scenario tests.
- Modify `tests/test_narrative_world_transition.py`: V1 compatibility and no-resolver regression locks.
- Modify `tests/test_narrative_simulation.py`: heterogeneous conflict-resolution acceptance tests only; scheduler production remains unchanged.
- Modify `tests/test_narrative_trust_api.py`: exact five-name narrative surface RED/GREEN.

---

### Task 1: Authoritative Test-Only Conflict Resolution RED

**Files:**
- Create: `tests/test_narrative_conflict_resolution.py`
- Modify: `tests/test_narrative_world_transition.py`
- Modify: `tests/test_narrative_simulation.py`
- Production: no changes

**Interfaces locked by tests:**

```python
ConflictParticipant(
    actor_id: str,
    decision_id: str,
    action: ActionOption,
    transition_record_hash: str,
    transition_spec_hash: str,
    original_delta: StateDelta,
    allowed_write_cells: tuple[StateCellRef, ...],
)

ConflictResolutionContext(
    prior_state_hash: str,
    participants: tuple[ConflictParticipant, ...],
    conflict_cells: tuple[StateCellRef, ...],
)

ConflictResolverSpec(
    resolver_id: str,
    version: str,
    domain_id: str,
    domain_version: str,
    domain_spec_hash: str,
    supported_action_types: tuple[str, ...],
    resolver_hook: object,
)

ConflictResolutionRecord(
    resolver_id: str,
    resolver_hash: str,
    prior_state_hash: str,
    context: ConflictResolutionContext,
    resolved_delta: StateDelta,
)
```

`WorldTransitionModelSpec` gains final optional field `conflict_resolver=None`; `WorldStepResult` gains final optional field `conflict_resolutions=()`; `WorldTransitionConflictResolutionError` subclasses `WorldTransitionConflictError`.

- [ ] **Step 1: Add guarded Conflict V2 imports so discovery remains precise before production exists**

At the top of `tests/test_narrative_conflict_resolution.py`:

```python
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


class NarrativeConflictResolutionTests(unittest.TestCase):
    def require_conflict(self) -> None:
        if _CONFLICT_IMPORT_ERROR is not None:
            self.fail(
                "narrative conflict resolution boundary is missing: "
                f"{_CONFLICT_IMPORT_ERROR}"
            )
```

Use normal imports for `StateDelta`, `StateDeltaOp`, `ActionOption`, `EntityRef`, `StateCellRef`, `TypedValue`, `WorldTransitionModelSpec`, `advance_world_step`, and the existing world-test helpers `make_world_domain`, `make_world_story`, and `intent`.

- [ ] **Step 2: Add a direct canonical record test with full `ActionOption.arguments`**

```python
def _hash(label: str) -> str:
    return stable_content_hash({"conflict-test": label})


def service_health_cell() -> StateCellRef:
    return StateCellRef(EntityRef("svc", "Service"), "service.health")


class NarrativeConflictResolutionTests(unittest.TestCase):
    def test_records_bind_full_action_payload_and_exact_conflict_cells(self):
        self.require_conflict()
        cell = service_health_cell()
        action = ActionOption(
            "alice-recover",
            "service-health-action",
            {
                "service": TypedValue("ServiceRef", EntityRef("svc", "Service")),
                "health": TypedValue("HealthState", "recovered"),
            },
        )
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
        alice = ConflictParticipant(
            "alice",
            "d-alice-service",
            action,
            _hash("alice-transition"),
            _hash("service-transition-spec"),
            delta,
            (cell,),
        )
        bob = ConflictParticipant(
            "bob",
            "d-bob-service",
            replace(action, id="bob-recover"),
            _hash("bob-transition"),
            _hash("service-transition-spec"),
            delta,
            (cell,),
        )
        context = ConflictResolutionContext(
            _hash("prior"),
            (bob, alice),
            (cell,),
        )
        self.assertEqual(
            tuple(item.actor_id for item in context.participants),
            ("alice", "bob"),
        )
        self.assertEqual(context.conflict_cells, (cell,))
        self.assertEqual(
            context.participants[0].action.arguments["service"],
            TypedValue("ServiceRef", EntityRef("svc", "Service")),
        )
        with self.assertRaises(ValueError):
            ConflictResolutionContext(_hash("prior"), (alice, bob), ())
```

- [ ] **Step 3: Add resolver identity/attestation test**

```python
class PreferLexicalActorResolver:
    def __call__(self, snapshot, context):
        return min(context.participants, key=lambda item: item.actor_id).original_delta


class PreferLexicalActorResolverV2:
    def __call__(self, snapshot, context):
        ordered = tuple(sorted(context.participants, key=lambda item: item.actor_id))
        return ordered[0].original_delta


def make_service_resolver(domain, hook):
    return ConflictResolverSpec(
        "service-resolver",
        "1",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        ("service-health-action",),
        hook,
    )


class NarrativeConflictResolutionTests(unittest.TestCase):
    def test_resolver_identity_binds_hook_and_supported_action_types(self):
        self.require_conflict()
        domain = make_world_domain()
        first = make_service_resolver(domain, PreferLexicalActorResolver())
        second = make_service_resolver(domain, PreferLexicalActorResolver())
        changed_hook = make_service_resolver(domain, PreferLexicalActorResolverV2())
        changed_types = replace(
            first,
            supported_action_types=("actor-phase-action", "service-health-action"),
        )
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertNotEqual(first.content_hash, changed_hook.content_hash)
        self.assertNotEqual(first.content_hash, changed_types.content_hash)
```

- [ ] **Step 4: Add exact V1 payload/hash locks to `tests/test_narrative_world_transition.py`**

```python
    def test_resolver_free_world_model_keeps_exact_v1_payload(self):
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
        self.assertEqual(model.content_hash, stable_content_hash(expected))
        self.assertNotIn("conflict_resolver_hash", model.to_dict())

    def test_resolver_free_world_step_keeps_exact_v1_batch_payload(self):
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        result = advance_world_step(
            story,
            domain,
            prior,
            make_transition_model(),
            (intent("d-bob-phase", "bob-ready"),),
        )
        self.assertEqual(
            result.next_state.transition_batch_hash,
            stable_content_hash([item.to_dict() for item in result.transitions]),
        )
        self.assertNotIn("conflict_resolutions", result.to_dict())
```

Do not alter the existing same-value and different-value conflict rejection tests.

- [ ] **Step 5: Create a self-contained reference conflict domain/story fixture**

Extend `make_world_domain()` with these exact additions:

```python
value_types += (
    ValueTypeSpec("BidLevel", "enum", allowed_values=("8", "11")),
    ValueTypeSpec("CombatHealth", "enum", allowed_values=("healthy", "injured")),
    ValueTypeSpec(
        "CombatStatus",
        "enum",
        allowed_values=("idle", "attack-success", "attack-failed", "defense-success"),
    ),
)
state_variables += (
    StateVariableSpec("service.owner", "Service", "AgentRef"),
    StateVariableSpec("agent.health", "Agent", "CombatHealth"),
    StateVariableSpec("agent.combat", "Agent", "CombatStatus"),
)
action_types += (
    ActionTypeSpec("claim-service-action", (ParameterSpec("service", "ServiceRef"),)),
    ActionTypeSpec(
        "bid-service-action",
        (ParameterSpec("service", "ServiceRef"), ParameterSpec("bid", "BidLevel")),
    ),
    ActionTypeSpec("attack-action", (ParameterSpec("target", "AgentRef"),)),
    ActionTypeSpec("defend-action", ()),
    ActionTypeSpec("taunt-action", (ParameterSpec("target", "AgentRef"),)),
)
decision_types += (
    DecisionTypeSpec("claim-choice", "Agent", "claim-service-action"),
    DecisionTypeSpec("bid-choice", "Agent", "bid-service-action"),
    DecisionTypeSpec("attack-choice", "Agent", "attack-action"),
    DecisionTypeSpec("defend-choice", "Agent", "defend-action"),
    DecisionTypeSpec("taunt-choice", "Agent", "taunt-action"),
)
```

Create `make_conflict_story(domain)` by taking `base = make_world_story()`, replacing its domain hash with `domain.content_hash`, adding `carol`, `dave`, and `svc2`, then appending authored decisions with these exact identities:

| decision | actor | type | action | arguments |
| --- | --- | --- | --- | --- |
| `d-alice-claim-svc` | alice | claim-choice | `alice-claim-svc` | service=svc |
| `d-bob-claim-svc` | bob | claim-choice | `bob-claim-svc` | service=svc |
| `d-carol-claim-svc2` | carol | claim-choice | `carol-claim-svc2` | service=svc2 |
| `d-dave-claim-svc2` | dave | claim-choice | `dave-claim-svc2` | service=svc2 |
| `d-alice-bid` | alice | bid-choice | `alice-bid-11` | service=svc, bid=11 |
| `d-bob-bid` | bob | bid-choice | `bob-bid-11` | service=svc, bid=11 |
| `d-carol-bid` | carol | bid-choice | `carol-bid-8` | service=svc, bid=8 |
| `d-alice-attack-bob` | alice | attack-choice | `alice-attack-bob` | target=bob |
| `d-bob-defend` | bob | defend-choice | `bob-defend` | no arguments |
| `d-carol-taunt-bob` | carol | taunt-choice | `carol-taunt-bob` | target=bob |

Use consecutive logical times after the base story's final authored decision.

Define exact test transition hooks:

```python
class ClaimTransition:
    def __call__(self, snapshot, decision, action):
        service = action.arguments["service"].value.entity_id
        return StateDelta((StateDeltaOp(
            "set",
            service,
            "service.owner",
            TypedValue("AgentRef", EntityRef(decision.actor_id, "Agent")),
        ),))


class BidTransition(ClaimTransition):
    pass


class AttackTransition:
    def __call__(self, snapshot, decision, action):
        target = action.arguments["target"].value.entity_id
        return StateDelta((
            StateDeltaOp("set", target, "agent.health", TypedValue("CombatHealth", "injured")),
            StateDeltaOp(
                "set",
                decision.actor_id,
                "agent.combat",
                TypedValue("CombatStatus", "attack-success"),
            ),
        ))


class DefendTransition:
    def __call__(self, snapshot, decision, action):
        return StateDelta((
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
        ))


class TauntTransition:
    def __call__(self, snapshot, decision, action):
        target = action.arguments["target"].value.entity_id
        return StateDelta((StateDeltaOp(
            "set",
            target,
            "agent.combat",
            TypedValue("CombatStatus", "attack-failed"),
        ),))
```

`make_conflict_world_model(domain, resolver)` must declare capabilities that match those hooks exactly: claim/bid write `service.owner` through argument `service`; attack writes target `agent.health` and actor `agent.combat`; defend writes actor `agent.health` and actor `agent.combat`; taunt writes target `agent.combat`.

- [ ] **Step 6: Add deterministic test resolvers for the three roadmap scenarios**

```python
class ClaimResolver:
    def __call__(self, snapshot, context):
        winner = min(context.participants, key=lambda item: item.actor_id)
        service = winner.action.arguments["service"].value.entity_id
        return StateDelta((StateDeltaOp(
            "set",
            service,
            "service.owner",
            TypedValue("AgentRef", EntityRef(winner.actor_id, "Agent")),
        ),))


class AuctionResolver:
    def __call__(self, snapshot, context):
        winner = min(
            context.participants,
            key=lambda item: (-int(item.action.arguments["bid"].value), item.actor_id),
        )
        service = winner.action.arguments["service"].value.entity_id
        return StateDelta((StateDeltaOp(
            "set",
            service,
            "service.owner",
            TypedValue("AgentRef", EntityRef(winner.actor_id, "Agent")),
        ),))


class AttackDefenseResolver:
    def __call__(self, snapshot, context):
        attack = next(item for item in context.participants if item.action.type_name == "attack-action")
        defense = next(item for item in context.participants if item.action.type_name == "defend-action")
        target = attack.action.arguments["target"].value.entity_id
        self.assert_target = target
        if defense.actor_id != target:
            return attack.original_delta
        return StateDelta((
            StateDeltaOp("set", target, "agent.health", TypedValue("CombatHealth", "healthy")),
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
        ))
```

The lexical tie-break lives inside `AuctionResolver`; the engine must not provide one.

- [ ] **Step 7: Add concrete component/reference tests**

Use public `advance_world_step()` only. Add these exact test names:

```text
test_two_way_claim_conflict_resolves_once_against_prior_snapshot
test_three_way_transitive_attack_defend_taunt_is_one_component
test_two_independent_claim_components_are_canonical_and_order_invariant
test_noop_delta_does_not_create_component
test_unsupported_action_type_rejects_before_action_and_resolver_hooks
test_resolver_exception_is_typed_and_preserves_runtime_error_cause
test_resolver_output_cannot_exceed_union_capability
test_invalid_resolver_value_clear_set_and_duplicate_output_are_typed
test_one_bad_component_aborts_entire_world_step
test_resolver_vs_nonconflicting_write_collision_rejects
test_resolution_vs_resolution_write_collision_rejects
test_competing_claims_keep_losing_attempt_in_transition_lineage
test_three_party_auction_uses_bid_argument_and_explicit_lexical_tie_break
test_attack_defense_replaces_entire_conflicting_component_delta
```

For the three reference tests, assert these exact outcomes:

```python
self.assertEqual(len(result.transitions), 2)          # claims / attack-defense
self.assertEqual(len(result.conflict_resolutions), 1)
self.assertEqual(
    {item.transition_record_hash for item in result.conflict_resolutions[0].context.participants},
    {item.content_hash for item in result.transitions},
)
```

Claim winner: `service.owner == AgentRef("alice", "Agent")` under `ClaimResolver`.

Auction winner: Alice and Bob both bid 11, so explicit resolver tie-break chooses Alice; Carol's bid 8 loses. Reversing input intent order must preserve `result.to_dict()` and `result.content_hash`.

Attack/defense outcome: Bob health is healthy, Alice combat is attack-failed, Bob combat is defense-success, and no final applied value equals Alice's raw attack-success attempt.

For transitive component construction, use Alice attack Bob, Bob defend, Carol taunt Bob. Alice/Bob overlap on `bob.agent.health`; Bob/Carol overlap on `bob.agent.combat`; Alice/Carol do not overlap directly. Assert one resolver context with participants `(alice, bob, carol)`.

For independent components, use Alice/Bob claims on `svc` and Carol/Dave claims on `svc2`; assert two resolution records in canonical component order and input-order invariance.

- [ ] **Step 8: Add heterogeneous scheduler acceptance tests without changing scheduler production**

In `tests/test_narrative_simulation.py`, add a second guarded import block for `ConflictResolverSpec` and `WorldTransitionConflictResolutionError` so existing tests still discover before Conflict V2 exists.

Add `make_heterogeneous_conflict_case()` by adapting existing scheduler helpers with these exact assignments:

```text
a1: intentional, scheduler-alert-choice, selects raise alert
a2: reactive, scheduler-alert-choice, score hook selects raise alert
a3: planning, scheduler-alert-choice, reward hook selects raise alert
```

All three authored actions target the same `service.alert` cell through `alert-control-action`. The configured resolver returns `service.alert=True` and records only the immutable world snapshot plus `ConflictResolutionContext`.

Add:

```python
    def test_heterogeneous_conflict_resolution_keeps_shared_prior_and_projects_resolved_world(self):
        domain, story, model = make_heterogeneous_conflict_case()
        prior = simulation_state_from_story(story, domain, model, at_time=10)
        result = simulate_step(story, domain, prior, model)
        self.assertEqual(
            {item.decision_result.model_kind for item in result.agent_steps},
            {"reactive", "intentional", "planning"},
        )
        for step in result.agent_steps:
            self.assertEqual(step.decision_result.ledger_hash, prior.evidence_ledger.content_hash)
        self.assertEqual(len(result.world_step.conflict_resolutions), 1)
        self.assertEqual(
            result.world_step.next_state.values[alert_cell()],
            TypedValue("AlertState", True),
        )
        self.assertEqual(
            result.admission_result.projection_result.source_world_step_hash,
            result.world_step.content_hash,
        )

    def test_two_round_heterogeneous_conflict_resolution_replays_exactly(self):
        domain1, story1, model1 = make_heterogeneous_conflict_case()
        domain2, story2, model2 = make_heterogeneous_conflict_case()
        initial1 = simulation_state_from_story(story1, domain1, model1, at_time=10)
        initial2 = simulation_state_from_story(story2, domain2, model2, at_time=10)
        first = simulate_trajectory(story1, domain1, initial1, model1, rounds=2)
        second = simulate_trajectory(story2, domain2, initial2, model2, rounds=2)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertTrue(all(step.world_step.conflict_resolutions for step in first.steps))
```

Keep the existing `test_world_failure_blocks_projection_and_preserves_prior_state` unchanged for the no-resolver case.

- [ ] **Step 9: Run focused tests and verify the expected RED shape**

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution \
  tests.test_narrative_world_transition \
  tests.test_narrative_simulation -v
```

Expected: all existing tests plus the two V1 compatibility tests remain GREEN. New Conflict V2 tests fail only because `conflict.py`, optional resolver/result fields, and the new typed error do not exist. No cognition, observation, world-V1, or syntax regression is acceptable.

- [ ] **Step 10: Commit only tests**

```bash
git add \
  tests/test_narrative_conflict_resolution.py \
  tests/test_narrative_world_transition.py \
  tests/test_narrative_simulation.py
git commit -m "test: define narrative conflict resolution v2 contract"
```

- [ ] **Step 11: Open a Draft PR and obtain authoritative exact-head RED**

Open Draft PR `feat: add narrative conflict resolution v2` against `proof/narrative-dynamics-v0`, tracking #27. Accept RED only from the PR-triggered `proof` run on the exact test-only head. Record run id/number and confirm every failure belongs to the new Conflict V2 contract.

---

### Task 2: Conflict Records and Resolver Specification GREEN

**Files:**
- Create: `narrative_dynamics/narrative/conflict.py`
- Test: `tests/test_narrative_conflict_resolution.py`

**Produces:** `ConflictParticipant`, `ConflictResolutionContext`, `ConflictResolverSpec`, `ConflictResolutionRecord`.

- [ ] **Step 1: Add focused validation helpers**

```python
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
    return (cell.subject.entity_type, cell.subject.entity_id, cell.state_variable)


def _delta_payload(delta: StateDelta) -> dict[str, object]:
    operations = tuple(sorted(delta.operations, key=lambda item: (item.subject_id, item.state_variable)))
    return {"operations": [item.to_dict() for item in operations]}
```

- [ ] **Step 2: Implement `ConflictParticipant` exactly**

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
        object.__setattr__(self, "transition_record_hash", _hash(self.transition_record_hash, label="conflict transition record hash"))
        object.__setattr__(self, "transition_spec_hash", _hash(self.transition_spec_hash, label="conflict transition spec hash"))
        if not isinstance(self.original_delta, StateDelta):
            raise TypeError("conflict participant original delta must be StateDelta")
        cells = tuple(self.allowed_write_cells)
        if any(not isinstance(cell, StateCellRef) for cell in cells):
            raise TypeError("conflict participant allowed cells must be StateCellRef values")
        if len(set(cells)) != len(cells):
            raise ValueError("conflict participant allowed cells must be unique")
        cells = tuple(sorted(cells, key=_cell_key))
        allowed_keys = {(cell.subject.entity_id, cell.state_variable) for cell in cells}
        for operation in self.original_delta.operations:
            if (operation.subject_id, operation.state_variable) not in allowed_keys:
                raise ValueError("conflict participant original delta exceeds allowed cells")
        object.__setattr__(self, "allowed_write_cells", cells)

    def to_dict(self) -> dict[str, object]:
        return {
            "actor_id": self.actor_id,
            "decision_id": self.decision_id,
            "action": self.action.to_dict(),
            "transition_record_hash": self.transition_record_hash,
            "transition_spec_hash": self.transition_spec_hash,
            "original_delta": _delta_payload(self.original_delta),
            "allowed_write_cells": [cell.to_dict() for cell in self.allowed_write_cells],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())
```

- [ ] **Step 3: Implement exact conflict-cell derivation and `ConflictResolutionContext`**

```python
def _participant_write_cells(participant: ConflictParticipant) -> tuple[StateCellRef, ...]:
    by_key = {
        (cell.subject.entity_id, cell.state_variable): cell
        for cell in participant.allowed_write_cells
    }
    return tuple(sorted({
        by_key[(operation.subject_id, operation.state_variable)]
        for operation in participant.original_delta.operations
    }, key=_cell_key))


@dataclass(frozen=True)
class ConflictResolutionContext:
    prior_state_hash: str
    participants: tuple[ConflictParticipant, ...]
    conflict_cells: tuple[StateCellRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "prior_state_hash", _hash(self.prior_state_hash, label="conflict prior state hash"))
        participants = tuple(self.participants)
        if len(participants) < 2:
            raise ValueError("conflict context requires at least two participants")
        if any(not isinstance(item, ConflictParticipant) for item in participants):
            raise TypeError("conflict context participants must be ConflictParticipant values")
        participants = tuple(sorted(participants, key=lambda item: (item.actor_id, item.decision_id, item.action.id)))
        keys = tuple((item.actor_id, item.decision_id, item.action.id) for item in participants)
        if len(set(keys)) != len(keys):
            raise ValueError("conflict context participant keys must be unique")
        counts = Counter(cell for participant in participants for cell in _participant_write_cells(participant))
        expected = tuple(sorted((cell for cell, count in counts.items() if count >= 2), key=_cell_key))
        supplied = tuple(sorted(tuple(self.conflict_cells), key=_cell_key))
        if not expected or supplied != expected:
            raise ValueError("conflict context cells must equal exact overlapping writes")
        object.__setattr__(self, "participants", participants)
        object.__setattr__(self, "conflict_cells", supplied)
```

`to_dict()` serializes prior hash, participant payloads, and conflict cell payloads in that canonical order; `content_hash` hashes `to_dict()`.

- [ ] **Step 4: Implement `ConflictResolverSpec` with implementation attestation in identity**

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
        object.__setattr__(self, "resolver_id", _text(self.resolver_id, label="conflict resolver id"))
        object.__setattr__(self, "version", _text(self.version, label="conflict resolver version"))
        object.__setattr__(self, "domain_id", _text(self.domain_id, label="conflict resolver domain id"))
        object.__setattr__(self, "domain_version", _text(self.domain_version, label="conflict resolver domain version"))
        object.__setattr__(self, "domain_spec_hash", _hash(self.domain_spec_hash, label="conflict resolver domain spec hash"))
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
            "implementation_identity": measure_implementation(self.resolver_hook).manifest_identity(),
        }
```

- [ ] **Step 5: Implement canonical `ConflictResolutionRecord` without external-model claims**

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
            raise TypeError("conflict resolution resolved delta must be StateDelta")
```

`to_dict()` includes resolver id/hash, prior hash, full context payload, and canonical resolved-delta payload; `content_hash` hashes it.

- [ ] **Step 6: Run direct record tests**

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_records_bind_full_action_payload_and_exact_conflict_cells \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_resolver_identity_binds_hook_and_supported_action_types -v
```

Expected: PASS. World execution/reference/scheduler Conflict V2 tests remain RED.

- [ ] **Step 7: Commit sidecar records/spec**

```bash
git add narrative_dynamics/narrative/conflict.py
git commit -m "feat: add narrative conflict resolver records"
```

---

### Task 3: World Model Resolver Configuration and V1 Compatibility GREEN

**Files:**
- Modify: `narrative_dynamics/narrative/world.py`

**Produces:** `WorldTransitionConflictResolutionError`, optional `WorldTransitionModelSpec.conflict_resolver`, resolver declaration validation, exact resolver-free payload compatibility.

- [ ] **Step 1: Add one-way imports and the typed error in `world.py`**

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

- [ ] **Step 2: Append `conflict_resolver` to `WorldTransitionModelSpec`**

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

When non-null, require `ConflictResolverSpec` and exact resolver/world `(domain_id, domain_version, domain_spec_hash)` equality.

- [ ] **Step 3: Preserve exact resolver-free `to_dict()` shape**

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

- [ ] **Step 4: Validate resolver-supported action declarations before any action hook**

Extend `_validate_transition_declarations(domain, model)` so each configured `resolver.supported_action_types` entry is resolved through `domain._action_type()`. Wrap undeclared types in `WorldTransitionError("conflict resolver names an undeclared action type")` and preserve the domain error as cause.

- [ ] **Step 5: Add resolver attestation helper**

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
  tests.test_narrative_world_transition.NarrativeWorldTransitionTests.test_resolver_free_world_model_keeps_exact_v1_payload \
  tests.test_narrative_world_transition.NarrativeWorldTransitionTests.test_resolver_free_world_step_keeps_exact_v1_batch_payload \
  tests.test_narrative_world_transition.NarrativeWorldTransitionTests.test_different_value_overlap_is_typed_conflict \
  tests.test_narrative_world_transition.NarrativeWorldTransitionTests.test_same_value_overlap_is_also_typed_conflict \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_unsupported_action_type_rejects_before_action_and_resolver_hooks -v
```

Expected: PASS. Configured valid conflicts still remain RED because no resolver execution path exists yet.

- [ ] **Step 7: Commit resolver configuration**

```bash
git add narrative_dynamics/narrative/world.py
git commit -m "feat: bind conflict resolver to world model"
```

---

### Task 4: Connected Components, Canonical Context, and Resolver Execution GREEN

**Files:**
- Modify: `narrative_dynamics/narrative/world.py`

**Produces private helpers:** `_conflict_components`, `_build_conflict_context`, `_validated_resolver_delta`, `_run_conflict_resolver`.

- [ ] **Step 1: Build deterministic connected components from actual record writes**

```python
def _conflict_components(records, entities):
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
    seen = set()
    components = []
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
        components.append(tuple(sorted(members, key=_transition_record_key)))
    return tuple(sorted(
        components,
        key=lambda component: tuple(_transition_record_key(item) for item in component),
    ))
```

- [ ] **Step 2: Recompute every participant capability from canonical story/model data**

For each record in a component:

1. find canonical decision by `record.intent.decision_id`;
2. require `decision.actor_id == record.actor_id`;
3. find canonical action in that decision and require `action == record.action`;
4. find `ActionTransitionSpec` by `record.action.type_name` and require `transition.content_hash == record.transition_spec_hash`;
5. recompute `_allowed_cells(domain, entities, decision, action, transition)`;
6. construct `ConflictParticipant` from the exact record/action/capability values.

Any mismatch raises `WorldTransitionConflictResolutionError("conflict participant does not bind canonical transition data")` before resolver invocation.

- [ ] **Step 3: Build `ConflictResolutionContext` from exact overlap cells**

Count `_record_write_cells()` across one component, keep cells with count at least 2, sort by `_cell_key`, and construct the context with `prior.content_hash`. Reconstruct one fresh `ConflictResolutionContext` from the public fields and compare `to_dict()` to reject constructor-bypassing forgeries.

- [ ] **Step 4: Extract one pure canonical delta validator and preserve the existing action wrapper**

Move the structural body of `_validated_delta()` into `_canonical_validated_delta(domain, entities, allowed, delta) -> StateDelta`, raising only `TypeError`/`ValueError`. Keep the existing action wrapper message and cause type unchanged. Add:

```python
def _validated_resolver_delta(domain, entities, allowed, delta):
    try:
        return _canonical_validated_delta(domain, entities, allowed, delta)
    except (TypeError, ValueError) as error:
        raise WorldTransitionConflictResolutionError(
            "conflict resolver produced an invalid state delta"
        ) from error
```

- [ ] **Step 5: Execute one resolver per component**

```python
def _run_conflict_resolver(snapshot, domain, entities, prior, resolver, context):
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
    delta = _validated_resolver_delta(domain, entities, allowed, raw_delta)
    return ConflictResolutionRecord(
        resolver.resolver_id,
        resolver_hash,
        prior.content_hash,
        context,
        delta,
    )
```

Never catch `BaseException`.

- [ ] **Step 6: Wire component discovery/resolver invocation into `advance_world_step()` without partial world application**

After all `ActionTransitionRecord`s are built, canonicalize them and derive components. If there are no components, return the unchanged V1 `_atomic_result()` path. If components exist without a resolver, call the existing `_reject_write_conflicts()` and preserve `WorldTransitionConflictError`. If a resolver exists, build contexts and resolution records but do not construct an intermediate world state; Task 5 consumes those records atomically.

- [ ] **Step 7: Run component/preflight/safety tests**

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_three_way_transitive_attack_defend_taunt_is_one_component \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_two_independent_claim_components_are_canonical_and_order_invariant \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_noop_delta_does_not_create_component \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_resolver_exception_is_typed_and_preserves_runtime_error_cause \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_resolver_output_cannot_exceed_union_capability \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_invalid_resolver_value_clear_set_and_duplicate_output_are_typed -v
```

Expected: component/context and typed failure tests PASS. Successful resolved next-state/lineage tests remain RED.

- [ ] **Step 8: Commit component/resolver execution**

```bash
git add narrative_dynamics/narrative/world.py
git commit -m "feat: execute canonical conflict components"
```

---

### Task 5: Atomic Resolved World, Explicit Lineage, Reference Scenarios, Scheduler GREEN

**Files:**
- Modify: `narrative_dynamics/narrative/world.py`
- Production scheduler: no changes

**Produces:** complete resolved `WorldStepResult`, V2 batch lineage, global effective-collision safety, atomic final commit.

- [ ] **Step 1: Append optional `conflict_resolutions` to `WorldStepResult`**

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

In `__post_init__`, canonicalize resolution records by ordered participant transition hashes; require each prior hash equals `prior_state.content_hash`; require every participant transition hash exists in `transitions`; compare participant actor/decision/action/spec/delta with its referenced transition; require one transition belongs to at most one resolution; require non-empty resolutions share one resolver id/hash; validate `next_state.transition_batch_hash` with Step 2.

`to_dict()` must add `conflict_resolutions` only when non-empty.

- [ ] **Step 2: Extend `_transition_batch_hash()` with an exact V1 empty-resolution branch**

```python
def _transition_batch_hash(transitions, conflict_resolutions=()):
    canonical_transitions = tuple(sorted(transitions, key=_transition_record_key))
    if not conflict_resolutions:
        return stable_content_hash([item.to_dict() for item in canonical_transitions])
    canonical_resolutions = tuple(sorted(conflict_resolutions, key=_conflict_resolution_key))
    return stable_content_hash({
        "transitions": [item.to_dict() for item in canonical_transitions],
        "conflict_resolutions": [item.to_dict() for item in canonical_resolutions],
    })
```

- [ ] **Step 3: Build the final effective mutation set**

Compute `resolved_transition_hashes` from every resolution participant. Keep original deltas only for transitions whose `content_hash` is absent from that set. Add exactly one `resolved_delta` for each resolution. Sort original sources by `_transition_record_key` and resolution sources by `_conflict_resolution_key`.

- [ ] **Step 4: Reject every collision in the final effective mutation set**

Convert every effective operation to canonical `StateCellRef`. If any cell is written by two effective sources, raise `WorldTransitionConflictResolutionError("resolved world step still contains overlapping effective writes")`. This catches resolver-vs-nonconflicting and resolution-vs-resolution collisions. There is no fallback order.

- [ ] **Step 5: Runtime-certify resolution lineage against the configured resolver before next-state construction**

For each resolution/component pair require:

```text
resolution.resolver_id == model.conflict_resolver.resolver_id
resolution.resolver_hash == _attested_resolver_hash(model.conflict_resolver)
resolution.prior_state_hash == prior.content_hash
ordered participant transition hashes == ordered component transition hashes
participant ActionOption payload == exact transition ActionOption payload
participant original_delta == exact transition delta
```

Any mismatch raises `WorldTransitionConflictResolutionError` before the next state exists.

- [ ] **Step 6: Extend `_atomic_result()` with optional resolutions while preserving V1 no-resolution behavior**

When resolutions are empty, keep the current `_reject_write_conflicts()`, next-value loop, batch hash, and `WorldStepResult` payload exactly. When resolutions are non-empty, skip raw conflict rejection, validate effective mutations globally, apply only effective mutations to a fresh `dict(prior.values)`, compute the V2 batch hash, build one next `WorldState`, then one `WorldStepResult` with the canonical resolution tuple.

- [ ] **Step 7: Complete `advance_world_step()`**

The final branch structure is exactly:

```text
all action hooks -> canonical records -> conflict components
no components -> exact V1 atomic result
components + no resolver -> existing typed conflict rejection
components + resolver -> one resolution per component -> runtime certification -> one atomic resolved result
```

No temporary world state is visible to resolver hooks or other components.

- [ ] **Step 8: Run all world/reference tests**

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution \
  tests.test_narrative_world_transition -v
```

Expected: PASS, including V1 exact payload locks, connected components, capability/schema failures, global collision failures, claims, auction, attack/defense, input-order invariance, and prior-state atomicity.

- [ ] **Step 9: Run scheduler acceptance and prove scheduler production unchanged**

```bash
python3 -m unittest tests.test_narrative_simulation -v
git diff 1ad91cd3782e1da0154e3354737bbebf3f369197 -- narrative_dynamics/narrative/simulation.py
```

Expected: all simulation tests PASS and the production scheduler diff is empty.

- [ ] **Step 10: Run full Python suite**

```bash
python3 -m unittest discover -s tests -v
```

Expected: `OK`.

- [ ] **Step 11: Commit atomic world integration**

```bash
git add narrative_dynamics/narrative/world.py
git commit -m "feat: resolve simultaneous world conflicts atomically"
```

- [ ] **Step 12: Require exact-head core GREEN**

Wait for the PR-triggered `proof` run on this exact commit. Require dependency, conformance, Lean build, theorem tests, Python, Story, and Testimony all success. Record the run id/number and Python test count. Do not proceed to public exports on a failing core head.

---

### Task 6: Exact Five-Name Public-Surface Test-Only RED

**Files:**
- Modify: `tests/test_narrative_trust_api.py`
- Production exports: no changes

- [ ] **Step 1: Add exactly five expected narrative names**

```python
    "ConflictParticipant",
    "ConflictResolutionContext",
    "ConflictResolverSpec",
    "ConflictResolutionRecord",
    "WorldTransitionConflictResolutionError",
```

Keep package-root isolation assertions unchanged.

- [ ] **Step 2: Run only the trust surface test**

```bash
python3 -m unittest \
  tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation -v
```

Expected: one failure with exactly those five names missing from the narrative package surface.

- [ ] **Step 3: Commit the public-surface test only**

```bash
git add tests/test_narrative_trust_api.py
git commit -m "test: lock narrative conflict resolution surface"
```

- [ ] **Step 4: Require authoritative exact-head surface RED**

Wait for PR-triggered `proof` on this exact test-only head. All non-trust tests and Lean gates must remain GREEN; the sole Python failure must be the exact public-surface test with exactly five missing names.

---

### Task 7: Export Five Names and Final Exact-Head GREEN

**Files:**
- Modify: `narrative_dynamics/narrative/__init__.py`

- [ ] **Step 1: Import the four conflict records/specs**

```python
from narrative_dynamics.narrative.conflict import (
    ConflictParticipant,
    ConflictResolutionContext,
    ConflictResolutionRecord,
    ConflictResolverSpec,
)
```

Add `WorldTransitionConflictResolutionError` to the existing import block from `narrative_dynamics.narrative.world`.

- [ ] **Step 2: Add exactly five `__all__` entries**

```python
    "ConflictParticipant",
    "ConflictResolutionContext",
    "ConflictResolverSpec",
    "ConflictResolutionRecord",
    "WorldTransitionConflictResolutionError",
```

Do not modify `narrative_dynamics/__init__.py`.

- [ ] **Step 3: Run focused feature/public tests**

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution \
  tests.test_narrative_world_transition \
  tests.test_narrative_simulation \
  tests.test_narrative_trust_api -v
```

Expected: PASS.

- [ ] **Step 4: Run the full Python suite**

```bash
python3 -m unittest discover -s tests -v
```

Expected: `OK`.

- [ ] **Step 5: Commit exports only**

```bash
git add narrative_dynamics/narrative/__init__.py
git commit -m "feat: export narrative conflict resolution surface"
```

- [ ] **Step 6: Verify final scope**

Allowed base-to-feature paths are exactly:

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

Any production diff in cognition families, dispatch, scheduler, observation/perception, IR/domain, model comparison, package root, or Lean is a scope failure.

- [ ] **Step 7: Require final authoritative exact-head PR GREEN**

Wait for the PR-triggered `proof` run on the exact final feature head. Require every workflow step success and record the final Python test count. Do not mark ready or merge from an older GREEN head.

- [ ] **Step 8: Update evidence after GREEN only**

Update the Draft PR body with exact RED/GREEN commit SHAs and proof run ids/numbers. In issue #27, check all seven Conflict Resolution V2 items only after final GREEN. Keep issue #27 open because held-out model comparison, stochastic world/observation, and P2 identification remain unfinished.

- [ ] **Step 9: Hand off integration choice**

Invoke `superpowers:finishing-a-development-branch`. The human decides whether to merge; test success alone is not merge authorization.
