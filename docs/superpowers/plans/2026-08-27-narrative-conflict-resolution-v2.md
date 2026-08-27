# Narrative Conflict Resolution V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic, attested, capability-limited resolution for simultaneous multi-agent world conflicts while preserving resolver-free World Transition V1 behavior and lineage exactly.

**Architecture:** Every action hook still runs against one immutable prior world snapshot and yields one validated `ActionTransitionRecord`. The world layer builds connected components from actual overlapping delta write sets, invokes one optional attested `ConflictResolverSpec` once per component, validates one whole-component replacement `StateDelta`, checks the final effective mutation set globally for collisions, and constructs one atomic next `WorldState`. Original transition records remain in lineage; explicit resolution records explain why effective world mutation differs from attempted deltas.

**Tech Stack:** Python 3 dataclasses, `unittest`, `unittest.mock`, existing `stable_content_hash`, existing `measure_implementation` attestation, immutable `MappingProxyType` snapshots, GitHub Actions `proof` workflow.

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
- A configured resolver changes model identity, but if one step has no conflict its hook is not called and that step records no conflict-resolution lineage.
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
- Create `tests/test_narrative_conflict_resolution.py`: records, component, safety, forgery, lineage, and reference-scenario tests.
- Modify `tests/test_narrative_world_transition.py`: resolver-free V1 compatibility and configured-but-unused resolver locks only.
- Modify `tests/test_narrative_simulation.py`: heterogeneous conflict-resolution and resolver-identity acceptance tests only; scheduler production remains unchanged.
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

- [ ] **Step 1: Add guarded imports and test-forgery helpers**

At the top of `tests/test_narrative_conflict_resolution.py` import `fields`, `replace`, `unittest`, and `patch` from `unittest.mock`, plus the existing world/domain/IR helpers. Add:

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
        WorldStepResult,
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

- [ ] **Step 2: Add a complete direct participant/context fixture and canonical identity tests**

```python
def make_direct_participant_context_fixture():
    cell = StateCellRef(EntityRef("svc", "Service"), "service.health")
    action = ActionOption(
        "alice-recover",
        "service-health-action",
        {
            "service": TypedValue("ServiceRef", EntityRef("svc", "Service")),
            "health": TypedValue("HealthState", "recovered"),
        },
    )
    delta = StateDelta((StateDeltaOp(
        "set",
        "svc",
        "service.health",
        TypedValue("HealthState", "recovered"),
    ),))
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
    context = ConflictResolutionContext(_hash("prior"), (bob, alice), (cell,))
    return alice, bob, context
```

Add exact tests:

```python
    def test_records_bind_full_action_payload_and_exact_conflict_cells(self):
        self.require_conflict()
        alice, bob, context = make_direct_participant_context_fixture()
        self.assertEqual(tuple(item.actor_id for item in context.participants), ("alice", "bob"))
        self.assertEqual(
            context.participants[0].action.arguments["service"],
            TypedValue("ServiceRef", EntityRef("svc", "Service")),
        )
        self.assertEqual(context.conflict_cells, (StateCellRef(EntityRef("svc", "Service"), "service.health"),))
        with self.assertRaises(ValueError):
            ConflictResolutionContext(context.prior_state_hash, (alice, bob), ())

    def test_action_argument_change_changes_participant_and_context_identity(self):
        self.require_conflict()
        alice, bob, context = make_direct_participant_context_fixture()
        changed_action = replace(
            alice.action,
            arguments={
                "service": TypedValue("ServiceRef", EntityRef("svc", "Service")),
                "health": TypedValue("HealthState", "failed"),
            },
        )
        changed = replace(alice, action=changed_action)
        changed_context = ConflictResolutionContext(
            context.prior_state_hash,
            (changed, bob),
            context.conflict_cells,
        )
        self.assertNotEqual(alice.content_hash, changed.content_hash)
        self.assertNotEqual(context.content_hash, changed_context.content_hash)

    def test_resolution_record_rejects_prior_context_mismatch(self):
        self.require_conflict()
        alice, _, context = make_direct_participant_context_fixture()
        with self.assertRaises(ValueError):
            ConflictResolutionRecord(
                "resolver",
                _hash("resolver"),
                _hash("different-prior"),
                context,
                alice.original_delta,
            )
```

Add two implementation-distinct resolver hooks and assert resolver hash changes with hook bytes and supported action types:

```python
class PreferLexicalActorResolver:
    def __call__(self, snapshot, context):
        return min(context.participants, key=lambda item: item.actor_id).original_delta


class PreferLexicalActorResolverV2:
    def __call__(self, snapshot, context):
        ordered = tuple(sorted(context.participants, key=lambda item: item.actor_id))
        return ordered[0].original_delta
```

- [ ] **Step 3: Add resolver-free V1 and configured-but-unused world locks**

In `tests/test_narrative_world_transition.py`, guarded-import `ConflictResolverSpec` so old tests still discover before production exists. Add:

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

    def test_resolver_free_world_step_keeps_exact_v1_payload_and_batch_hash(self):
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        result = advance_world_step(
            story,
            domain,
            prior,
            make_transition_model(),
            (intent("d-bob-phase", "bob-ready"),),
        )
        expected_batch = stable_content_hash([item.to_dict() for item in result.transitions])
        expected_result = {
            "model_id": result.model_id,
            "model_hash": result.model_hash,
            "prior_state": result.prior_state.to_dict(),
            "transitions": [item.to_dict() for item in result.transitions],
            "next_state": result.next_state.to_dict(),
        }
        self.assertEqual(result.next_state.transition_batch_hash, expected_batch)
        self.assertEqual(result.to_dict(), expected_result)
        self.assertEqual(result.content_hash, stable_content_hash(expected_result))
```

Define:

```python
class RecordingNoopConflictResolver:
    def __init__(self):
        self.calls = []

    def __call__(self, snapshot, context):
        self.calls.append((dict(snapshot), context))
        return StateDelta(())
```

Add `test_configured_resolver_is_not_called_on_conflict_free_step`: configured world-model hash differs from resolver-free model; hook calls remain empty; result has empty `conflict_resolutions`; result payload omits the field; transition batch still equals the old list-of-transitions hash.

Keep existing `test_different_value_overlap_is_typed_conflict` and `test_same_value_overlap_is_also_typed_conflict` unchanged.

- [ ] **Step 4: Build one complete conflict reference domain/story**

Extend `make_world_domain()` in the new test module with:

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

`make_conflict_story(domain)` takes `base = make_world_story()`, replaces its domain hash with `domain.content_hash`, adds `carol`, `dave`, and `svc2`, and appends these decisions at consecutive logical times:

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

Define transition hooks:

```python
class ClaimTransition:
    def __call__(self, snapshot, decision, action):
        service = action.arguments["service"].value.entity_id
        return StateDelta((StateDeltaOp(
            "set", service, "service.owner",
            TypedValue("AgentRef", EntityRef(decision.actor_id, "Agent")),
        ),))


class BidTransition(ClaimTransition):
    pass


class AttackTransition:
    def __call__(self, snapshot, decision, action):
        target = action.arguments["target"].value.entity_id
        return StateDelta((
            StateDeltaOp("set", target, "agent.health", TypedValue("CombatHealth", "injured")),
            StateDeltaOp("set", decision.actor_id, "agent.combat", TypedValue("CombatStatus", "attack-success")),
        ))


class DefendTransition:
    def __call__(self, snapshot, decision, action):
        return StateDelta((
            StateDeltaOp("set", decision.actor_id, "agent.health", TypedValue("CombatHealth", "healthy")),
            StateDeltaOp("set", decision.actor_id, "agent.combat", TypedValue("CombatStatus", "defense-success")),
        ))


class TauntTransition:
    def __call__(self, snapshot, decision, action):
        target = action.arguments["target"].value.entity_id
        return StateDelta((StateDeltaOp(
            "set", target, "agent.combat", TypedValue("CombatStatus", "attack-failed"),
        ),))
```

`make_conflict_world_model(domain, resolver)` uses matching capabilities: claim/bid write `service.owner` via argument `service`; attack writes target `agent.health` plus actor `agent.combat`; defend writes actor `agent.health` plus actor `agent.combat`; taunt writes target `agent.combat`.

- [ ] **Step 5: Add deterministic resolver hooks for claims, auction, attack/defense, recording, and failure cases**

```python
class ClaimResolver:
    def __init__(self):
        self.calls = []

    def __call__(self, snapshot, context):
        self.calls.append((dict(snapshot), context))
        winner = min(context.participants, key=lambda item: item.actor_id)
        service = winner.action.arguments["service"].value.entity_id
        return StateDelta((StateDeltaOp(
            "set", service, "service.owner",
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
            "set", service, "service.owner",
            TypedValue("AgentRef", EntityRef(winner.actor_id, "Agent")),
        ),))


class AttackDefenseResolver:
    def __call__(self, snapshot, context):
        attack = next(item for item in context.participants if item.action.type_name == "attack-action")
        defense = next(item for item in context.participants if item.action.type_name == "defend-action")
        target = attack.action.arguments["target"].value.entity_id
        if defense.actor_id != target:
            return attack.original_delta
        return StateDelta((
            StateDeltaOp("set", target, "agent.health", TypedValue("CombatHealth", "healthy")),
            StateDeltaOp("set", attack.actor_id, "agent.combat", TypedValue("CombatStatus", "attack-failed")),
            StateDeltaOp("set", defense.actor_id, "agent.combat", TypedValue("CombatStatus", "defense-success")),
        ))


class RaisingResolver:
    def __init__(self):
        self.calls = 0

    def __call__(self, snapshot, context):
        self.calls += 1
        raise RuntimeError("resolver boom")
```

The auction lexical tie-break exists only inside `AuctionResolver`.

- [ ] **Step 6: Add every generic component/safety/reference test**

Add these exact test names:

```text
test_two_way_claim_conflict_resolves_once_against_prior_snapshot
test_three_way_transitive_attack_defend_taunt_is_one_component
test_two_independent_claim_components_are_canonical_and_order_invariant
test_noop_delta_does_not_create_component
test_undeclared_resolver_supported_type_rejects_before_action_hooks
test_conflict_participant_unsupported_type_rejects_before_resolver_hook
test_resolver_attestation_unavailable_rejects_before_resolver_hook
test_forged_action_payload_rejects_before_resolver_hook
test_resolver_exception_is_typed_and_preserves_runtime_error_cause
test_resolver_output_cannot_exceed_union_capability
test_invalid_resolver_value_clear_set_and_duplicate_output_are_typed
test_one_bad_component_aborts_entire_world_step
test_resolver_vs_nonconflicting_write_collision_rejects
test_resolution_vs_resolution_write_collision_rejects
test_competing_claims_keep_losing_attempt_in_transition_lineage
test_three_party_auction_uses_bid_argument_and_explicit_lexical_tie_break
test_attack_defense_replaces_entire_conflicting_component_delta
test_resolved_batch_hash_binds_original_transitions_and_resolution_records
```

Required assertions and fixture mechanics:

- Two-way claim: `ClaimResolver.calls == [(dict(prior.values), context)]`; participants `(alice,bob)`; final `service.owner` is Alice; one resolution.
- Transitive component: Alice attack Bob overlaps Bob defend on `bob.agent.health`; Bob defend overlaps Carol taunt Bob on `bob.agent.combat`; Alice and Carol do not directly overlap; one `(alice,bob,carol)` context.
- Independent components: Alice/Bob claim `svc`, Carol/Dave claim `svc2`; exactly two canonical resolution records; reversed intent order preserves `to_dict()` and content hash.
- No-op: append existing authored no-op intent to a conflict batch and assert it remains a normal non-conflicting transition, never a participant.
- Undeclared resolver supported type: configure resolver with `supported_action_types=("missing-action-type",)`; action hook call lists and resolver call list remain empty because declaration preflight fails before actions.
- Unsupported participant type: resolver declares only `claim-service-action`, but use an attack/defend conflict; action hooks run to produce deltas, resolver hook call list remains empty, and `WorldTransitionConflictResolutionError` is raised before hook invocation.
- Attestation unavailable: patch `narrative_dynamics.narrative.conflict.measure_implementation` to raise `ImplementationAttestationUnavailable("resolver unavailable")`; action hooks may complete, resolver hook call list stays empty, and exact attestation error is the cause chain.
- Forged action payload: after Task 4 helper exists, patch `narrative_dynamics.narrative.world._build_conflict_context` with `create=True` so it returns a forged context whose first participant has a changed `ActionOption.arguments` but original transition hashes; `advance_world_step()` must reject during runtime context/transition certification before resolver hook. Do not inject a forged `WorldState` or bypass `advance_world_step()`.
- Resolver exception: `caught.exception.__cause__` is `RuntimeError` with text `resolver boom`.
- Schema/capability cases: outside-union cell, wrong TypedValue type, clear carrying value, set missing value, duplicate output cell all raise `WorldTransitionConflictResolutionError`.
- Bad component: prior state payload remains exactly unchanged and no successful `WorldStepResult` is returned.
- Claim lineage: losing raw attempt remains in `result.transitions`; resolution context references both transition hashes.
- Auction: Alice/Bob bid 11, Carol 8; explicit resolver tie-break picks Alice; reversed input order yields identical result/hash.
- Attack/defense: Bob health healthy, Alice combat attack-failed, Bob combat defense-success; raw attack-success write is not applied.
- Resolved batch hash equals:

```python
stable_content_hash({
    "transitions": [item.to_dict() for item in result.transitions],
    "conflict_resolutions": [item.to_dict() for item in result.conflict_resolutions],
})
```

For the two final-collision tests, add one test-only `claim-with-audit-action` variant with parameters `service: ServiceRef` and `audit: ServiceRef`. Its declared effects are `service.owner` on `service` plus `service.health` on `audit`, but its action hook writes only `service.owner`. This makes `audit` health a legal resolver capability that was not an original conflict write.

- Resolver-vs-nonconflicting case: Alice/Bob form a claim component on `svc`, both use `audit=svc2`; a separate Dave `service-health-action` writes `svc2.service.health`; resolver returns winner owner plus `svc2.service.health=recovered`; final global collision must reject.
- Resolution-vs-resolution case: Alice/Bob claim `svc`, Carol/Dave claim `svc2`, all four use `audit=svc3`; resolver for each component writes its owner plus `svc3.service.health=recovered`; the two resolution deltas collide and must reject.

- [ ] **Step 7: Add `WorldStepResult` internal-forgery tests in the new conflict test module**

Create `run_valid_claim_result()` from the public claim fixture. Reconstruct `WorldStepResult` with `replace()`/`_forge()` and assert `ValueError` for each payload-internal forgery:

```text
resolution.prior_state_hash differs from prior_state.content_hash
participant.transition_record_hash absent from transitions
one transition hash appears in two resolution records
participant.action differs from referenced transition.action
participant.original_delta differs from referenced transition.delta
next_state.transition_batch_hash does not bind transitions + resolutions
```

Keep these tests in `test_narrative_conflict_resolution.py` to avoid circular imports with the world-transition fixture module.

- [ ] **Step 8: Add heterogeneous scheduler Conflict V2 acceptance with all three model families**

In `tests/test_narrative_simulation.py`, guarded-import `ConflictResolverSpec` and `WorldTransitionConflictResolutionError`.

Add:

```python
class ConflictAlertReactiveScoreHook:
    def __call__(self, context):
        return {"a2-raise-alert": 3.0, "a2-wait-alert": 0.0}


class ConflictAlertPlanningRewardHook:
    def __call__(self, context):
        if context.action.id == "a3-raise-alert":
            return 3.0
        if context.action.id == "a3-wait-alert":
            return 0.0
        raise AssertionError("unexpected conflict planning action")


class ConflictAlertResolver:
    def __init__(self):
        self.calls = []

    def __call__(self, snapshot, context):
        self.calls.append((dict(snapshot), context))
        return StateDelta((StateDeltaOp(
            "set", "svc", "service.alert", TypedValue("AlertState", True),
        ),))
```

`make_heterogeneous_conflict_case()` uses the existing scheduler domain/story helpers but replaces/extends decisions so all three scheduled actors have `scheduler-alert-choice` with actor-specific action ids:

```text
a1-raise-alert / a1-wait
a2-raise-alert / a2-wait-alert
a3-raise-alert / a3-wait-alert
```

Use `a1` Intentional with existing alert intentional goal/choice setup, `a2` Reactive with `ConflictAlertReactiveScoreHook`, and `a3` Planning with the existing alert hidden states/coupling/transition/observation hooks plus `ConflictAlertPlanningRewardHook`. All three selected raise actions target `service.alert` through the existing `AlertControlTransition`. Configure the world model with `ConflictAlertResolver` supporting only `alert-control-action`.

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
        self.assertEqual(result.world_step.next_state.values[alert_cell()], TypedValue("AlertState", True))
        self.assertEqual(
            result.admission_result.projection_result.source_world_step_hash,
            result.world_step.content_hash,
        )
        context = result.world_step.conflict_resolutions[0].context
        self.assertFalse(hasattr(context, "belief_state"))
        self.assertFalse(hasattr(context, "evidence_ledger"))
        self.assertFalse(hasattr(context, "planning_trace"))

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

Add `test_simulation_model_identity_binds_conflict_resolver_implementation`: build two otherwise identical heterogeneous conflict models whose resolver hooks are implementation-distinct but semantically equivalent and assert different resolver, world-model, and `SimulationModelSpec.content_hash` values. This test belongs in the initial test-only RED, not a later production commit.

Keep existing no-resolver `test_world_failure_blocks_projection_and_preserves_prior_state` unchanged.

- [ ] **Step 9: Run focused tests and verify expected RED shape**

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution \
  tests.test_narrative_world_transition \
  tests.test_narrative_simulation -v
```

Expected: existing tests and resolver-free V1 locks stay GREEN. New Conflict V2 tests fail only because `conflict.py`, optional resolver/result fields, and typed resolution error are absent. No unrelated regression is acceptable.

- [ ] **Step 10: Commit only tests**

```bash
git add \
  tests/test_narrative_conflict_resolution.py \
  tests/test_narrative_world_transition.py \
  tests/test_narrative_simulation.py
git commit -m "test: define narrative conflict resolution v2 contract"
```

- [ ] **Step 11: Open Draft PR and obtain authoritative exact-head RED**

Open Draft PR `feat: add narrative conflict resolution v2` against `proof/narrative-dynamics-v0`, tracking #27. Accept RED only from the PR-triggered `proof` run on the exact test-only head. Record run id/number and verify all failures belong to new Conflict V2 tests.

---

### Task 2: Conflict Records and Resolver Specification GREEN

**Files:**
- Create: `narrative_dynamics/narrative/conflict.py`

**Produces:** `ConflictParticipant`, `ConflictResolutionContext`, `ConflictResolverSpec`, `ConflictResolutionRecord`.

- [ ] **Step 1: Add validation/canonical helpers**

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


def _canonical_delta(delta: StateDelta, *, label: str) -> StateDelta:
    if not isinstance(delta, StateDelta):
        raise TypeError(f"{label} must be StateDelta")
    operations = tuple(sorted(delta.operations, key=lambda item: (item.subject_id, item.state_variable)))
    keys = tuple((item.subject_id, item.state_variable) for item in operations)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} cannot write one cell twice")
    return StateDelta(operations)


def _delta_payload(delta: StateDelta) -> dict[str, object]:
    return {"operations": [item.to_dict() for item in delta.operations]}
```

- [ ] **Step 2: Implement `ConflictParticipant` as canonical public data**

Validate ids/hashes and `ActionOption`; canonicalize `original_delta` with `_canonical_delta`; canonicalize unique allowed cells by `_cell_key`; require every original operation key exists in allowed cells; set canonical values back onto the frozen dataclass. `to_dict()` includes full `action.to_dict()`, both hashes, canonical delta payload, and allowed cells; `content_hash` hashes the payload.

- [ ] **Step 3: Implement exact conflict-cell derivation plus connected `ConflictResolutionContext`**

Derive each participant's actual write cells by mapping canonical operation keys into typed allowed cells. Context constructor must:

```text
require at least two participants
sort by actor_id, decision_id, action.id
require unique participant keys
compute exact conflict_cells as cells written by >=2 participants
require supplied conflict_cells equals exact derived set
build participant overlap graph from actual write cells
require the participant graph is connected
```

This rejects an extra disconnected participant even if the supplied conflict-cell set itself is correct. `to_dict()` and `content_hash` are canonical.

- [ ] **Step 4: Implement `ConflictResolverSpec`**

Validate resolver id/version/domain identity; require non-empty unique sorted supported action types and callable hook; bind `measure_implementation(self.resolver_hook).manifest_identity()` into `to_dict()` and `content_hash`.

- [ ] **Step 5: Implement `ConflictResolutionRecord`**

Validate resolver id/hash and prior hash; require context; require `context.prior_state_hash == prior_state_hash`; canonicalize `resolved_delta` with `_canonical_delta`; serialize resolver identity, prior hash, context, and canonical resolved delta. Do not claim external model certification.

- [ ] **Step 6: Run direct record tests**

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_records_bind_full_action_payload_and_exact_conflict_cells \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_action_argument_change_changes_participant_and_context_identity \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_resolution_record_rejects_prior_context_mismatch -v
```

Expected: PASS; world execution tests remain RED.

- [ ] **Step 7: Commit sidecar records/spec**

```bash
git add narrative_dynamics/narrative/conflict.py
git commit -m "feat: add narrative conflict resolver records"
```

---

### Task 3: World Model Resolver Configuration and V1 Compatibility GREEN

**Files:**
- Modify: `narrative_dynamics/narrative/world.py`

**Produces:** `WorldTransitionConflictResolutionError`, optional `WorldTransitionModelSpec.conflict_resolver`, declaration preflight, resolver-free payload compatibility.

- [ ] **Step 1: Add one-way imports and typed error in `world.py`**

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

- [ ] **Step 2: Append optional resolver field**

```python
conflict_resolver: ConflictResolverSpec | None = None
```

When configured, require exact type and exact resolver/world `(domain_id, domain_version, domain_spec_hash)` equality.

- [ ] **Step 3: Preserve exact resolver-free `to_dict()`**

Build the existing V1 payload first. Only when resolver exists add `conflict_resolver_hash = resolver.content_hash`. Never emit a null key.

- [ ] **Step 4: Split declaration failure timing correctly**

In `_validate_transition_declarations(domain, model)`, validate that every resolver `supported_action_types` name exists in `DomainSpec`; undeclared configured names fail before any action hook. Do not reject a declared-but-not-supported actual participant action here because conflict participation is not known until action deltas exist; Task 4 rejects that after action hooks but before resolver hook.

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

- [ ] **Step 6: Run Task 3 gates**

```bash
python3 -m unittest \
  tests.test_narrative_world_transition.NarrativeWorldTransitionTests.test_resolver_free_world_model_keeps_exact_v1_payload \
  tests.test_narrative_world_transition.NarrativeWorldTransitionTests.test_resolver_free_world_step_keeps_exact_v1_payload_and_batch_hash \
  tests.test_narrative_world_transition.NarrativeWorldTransitionTests.test_different_value_overlap_is_typed_conflict \
  tests.test_narrative_world_transition.NarrativeWorldTransitionTests.test_same_value_overlap_is_also_typed_conflict \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_undeclared_resolver_supported_type_rejects_before_action_hooks -v
```

Expected: PASS. Valid configured conflicts remain RED.

- [ ] **Step 7: Commit resolver configuration**

```bash
git add narrative_dynamics/narrative/world.py
git commit -m "feat: bind conflict resolver to world model"
```

---

### Task 4: Connected Components, Canonical Runtime Certification, Resolver Execution GREEN

**Files:**
- Modify: `narrative_dynamics/narrative/world.py`

**Produces:** private `_conflict_components`, `_build_conflict_context`, `_validated_resolver_delta`, `_run_conflict_resolver`.

- [ ] **Step 1: Build deterministic components from actual record writes**

Canonicalize records by `_transition_record_key`; make an undirected edge only when actual `_record_write_cells()` intersect; walk each connected component once; discard singletons; sort members and components by transition key. No-op deltas create no edges.

- [ ] **Step 2: Recompute exact participant data from canonical story/model state**

For each component record, locate canonical `Decision`, require actor equality, locate canonical `ActionOption` and require full equality, locate `ActionTransitionSpec` by action type, require `_attested_transition_hash(transition) == record.transition_spec_hash`, recompute `_allowed_cells()`, and construct exact `ConflictParticipant`.

Any mismatch raises `WorldTransitionConflictResolutionError("conflict participant does not bind canonical transition data")` before resolver hook.

- [ ] **Step 3: Build/reconstruct exact context and certify against the component**

Derive conflict cells by actual write counts. Construct `ConflictResolutionContext(prior.content_hash, participants, conflict_cells)`, reconstruct it from public fields, require identical payload, and require ordered participant transition hashes equal ordered component record hashes. This is the check the forged-context test targets.

- [ ] **Step 4: Extract pure world delta validation and add resolver wrapper**

Move existing action-delta structural/type/capability checks into `_canonical_validated_delta()` raising `TypeError`/`ValueError`. Keep existing `_validated_delta()` error class/message unchanged. Add `_validated_resolver_delta()` that wraps raw validation failures as `WorldTransitionConflictResolutionError("conflict resolver produced an invalid state delta")` with exact cause.

- [ ] **Step 5: Execute one resolver per component**

Before hook invocation:

```text
all participant action types must belong to resolver.supported_action_types
resolver attestation must succeed
context/component certification must already have passed
```

Then call `resolver_hook(snapshot, context)` with the same immutable prior snapshot used by action hooks. Catch `Exception`, not `BaseException`. Validate output against union participant capabilities and return one `ConflictResolutionRecord`.

- [ ] **Step 6: Wire resolver invocation without partial world application**

After all action records exist: no components use exact V1 `_atomic_result()`; components with no resolver use existing `_reject_write_conflicts()`; components with resolver produce one resolution record each but no intermediate `WorldState`. Task 5 owns final application.

- [ ] **Step 7: Run Task 4 gates**

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_three_way_transitive_attack_defend_taunt_is_one_component \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_two_independent_claim_components_are_canonical_and_order_invariant \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_noop_delta_does_not_create_component \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_conflict_participant_unsupported_type_rejects_before_resolver_hook \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_resolver_attestation_unavailable_rejects_before_resolver_hook \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_forged_action_payload_rejects_before_resolver_hook \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_resolver_exception_is_typed_and_preserves_runtime_error_cause \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_resolver_output_cannot_exceed_union_capability \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_invalid_resolver_value_clear_set_and_duplicate_output_are_typed -v
```

Expected: component/context/preflight/typed failure tests PASS. Successful resolved next-state/lineage tests remain RED.

- [ ] **Step 8: Commit component/resolver execution**

```bash
git add narrative_dynamics/narrative/world.py
git commit -m "feat: execute canonical conflict components"
```

---

### Task 5: Atomic Resolved World, Explicit Lineage, Reference and Scheduler GREEN

**Files:**
- Modify: `narrative_dynamics/narrative/world.py`
- Production scheduler: no changes

**Produces:** successful resolved `WorldStepResult`, V2 batch lineage, configured-but-unused behavior, final global collision safety, one atomic world commit.

- [ ] **Step 1: Append optional `WorldStepResult.conflict_resolutions`**

```python
conflict_resolutions: tuple[ConflictResolutionRecord, ...] = ()
```

Constructor canonicalizes resolution records and verifies only payload-internal facts: prior hash equals `prior_state.content_hash`; every participant transition hash exists in `transitions`; participant actor/decision/action/spec/delta equals referenced transition; one transition appears in at most one resolution; non-empty resolutions share one resolver id/hash; next-state batch hash matches transitions + resolutions. It does not independently certify an external world model.

`to_dict()` omits the field when empty.

- [ ] **Step 2: Extend transition-batch hash with exact V1 empty-resolution branch**

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

- [ ] **Step 3: Build effective mutations and final global collision validation**

Collect all transition hashes appearing in resolutions. Keep raw deltas only for transitions outside that set; add one resolved delta per component; sort effective sources canonically. Convert every operation to canonical `StateCellRef`; if any cell has two effective sources, raise `WorldTransitionConflictResolutionError("resolved world step still contains overlapping effective writes")` before next-state construction.

- [ ] **Step 4: Runtime-certify each resolution against the configured model**

Before successful next-state construction require exact configured resolver id/hash, exact prior hash, exact component transition hashes, exact ActionOption payloads, and exact original deltas. This is stronger than `ConflictResolutionRecord.__post_init__` and belongs only in `advance_world_step()`.

- [ ] **Step 5: Extend `_atomic_result()` while preserving V1 branch exactly**

Empty resolutions keep current raw conflict rejection, V1 mutation loop, V1 batch hash, and omitted-resolution payload. Non-empty resolutions skip raw overlap rejection, validate effective collisions, apply only effective mutations to a fresh prior-values copy, compute V2 batch hash, build one next `WorldState`, then one `WorldStepResult`.

- [ ] **Step 6: Complete `advance_world_step()`**

```text
all action hooks -> records -> components
no components -> exact V1 atomic result
components + no resolver -> existing WorldTransitionConflictError
components + resolver -> one resolution per component -> runtime certification -> one atomic resolved result
```

Resolvers never observe another resolver output or temporary state.

- [ ] **Step 7: Run full world/reference/forgery suite**

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution \
  tests.test_narrative_world_transition -v
```

Expected: PASS, including claims, auction, attack/defense, transitive/independent components, V1 locks, configured-but-unused resolver, resolved batch hash, internal result forgery rejection, final collision tests, and prior-state atomicity.

- [ ] **Step 8: Run scheduler acceptance and prove scheduler production unchanged**

```bash
python3 -m unittest tests.test_narrative_simulation -v
git diff 1ad91cd3782e1da0154e3354737bbebf3f369197 -- narrative_dynamics/narrative/simulation.py
```

Expected: all simulation tests PASS; scheduler production diff is empty; resolver implementation change alters world and simulation model hashes; heterogeneous two-round replay is exact.

- [ ] **Step 9: Run full Python suite**

```bash
python3 -m unittest discover -s tests -v
```

Expected: `OK`.

- [ ] **Step 10: Commit atomic world integration only**

```bash
git add narrative_dynamics/narrative/world.py
git commit -m "feat: resolve simultaneous world conflicts atomically"
```

All behavior-driving tests were already committed in Task 1.

- [ ] **Step 11: Require exact-head core GREEN**

Wait for PR-triggered `proof` on this exact commit. Require dependency, conformance, Lean build, theorem tests, Python, Story, and Testimony all success. Record run id/number and Python test count.

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

- [ ] **Step 2: Run only trust surface test**

```bash
python3 -m unittest \
  tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation -v
```

Expected: one failure with exactly those five names missing from narrative package surface.

- [ ] **Step 3: Commit surface test only**

```bash
git add tests/test_narrative_trust_api.py
git commit -m "test: lock narrative conflict resolution surface"
```

- [ ] **Step 4: Require authoritative exact-head surface RED**

Wait for PR-triggered `proof`. All non-trust tests and Lean gates remain GREEN; sole Python failure is exact public-surface test with exactly five missing names.

---

### Task 7: Export Five Names and Final Exact-Head GREEN

**Files:**
- Modify: `narrative_dynamics/narrative/__init__.py`

- [ ] **Step 1: Import four conflict data/spec names**

```python
from narrative_dynamics.narrative.conflict import (
    ConflictParticipant,
    ConflictResolutionContext,
    ConflictResolutionRecord,
    ConflictResolverSpec,
)
```

Add `WorldTransitionConflictResolutionError` to the existing import block from `world.py`.

- [ ] **Step 2: Add exactly five `__all__` entries**

```python
    "ConflictParticipant",
    "ConflictResolutionContext",
    "ConflictResolverSpec",
    "ConflictResolutionRecord",
    "WorldTransitionConflictResolutionError",
```

Do not modify package root.

- [ ] **Step 3: Run focused feature/public tests**

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

Wait for PR-triggered `proof` on exact final head. Require every workflow step success and record final Python test count. Do not mark ready or merge from older GREEN evidence.

- [ ] **Step 8: Update evidence after GREEN only**

Update Draft PR body with exact RED/GREEN SHAs and proof runs. In issue #27, check all seven Conflict Resolution V2 items only after final GREEN. Keep issue open because held-out comparison, stochastic world/observation, and P2 identification remain unfinished.

- [ ] **Step 9: Hand off integration choice**

Invoke `superpowers:finishing-a-development-branch`. The human decides whether to merge; test success alone is not merge authorization.
