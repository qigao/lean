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
- Create `tests/test_narrative_conflict_resolution.py`: generic records, component, safety, and reference-scenario tests.
- Modify `tests/test_narrative_world_transition.py`: V1 compatibility, configured-but-unused resolver behavior, no-resolver conflict locks, and result-forgery locks.
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

- [ ] **Step 1: Add guarded Conflict V2 imports so test discovery stays precise**

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

Use normal imports for `StateDelta`, `StateDeltaOp`, `ActionOption`, `EntityRef`, `StateCellRef`, `TypedValue`, `WorldTransitionModelSpec`, `advance_world_step`, and existing world-test helpers `make_world_domain`, `make_world_story`, and `intent`.

- [ ] **Step 2: Add canonical record/identity tests**

```python
def _hash(label: str) -> str:
    return stable_content_hash({"conflict-test": label})


def service_health_cell() -> StateCellRef:
    return StateCellRef(EntityRef("svc", "Service"), "service.health")


class PreferLexicalActorResolver:
    def __call__(self, snapshot, context):
        return min(context.participants, key=lambda item: item.actor_id).original_delta


class PreferLexicalActorResolverV2:
    def __call__(self, snapshot, context):
        ordered = tuple(sorted(context.participants, key=lambda item: item.actor_id))
        return ordered[0].original_delta
```

Add these tests:

```python
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
        self.assertEqual(tuple(item.actor_id for item in context.participants), ("alice", "bob"))
        self.assertEqual(context.conflict_cells, (cell,))
        self.assertEqual(
            context.participants[0].action.arguments["service"],
            TypedValue("ServiceRef", EntityRef("svc", "Service")),
        )
        with self.assertRaises(ValueError):
            ConflictResolutionContext(_hash("prior"), (alice, bob), ())

    def test_action_argument_change_changes_participant_and_context_identity(self):
        self.require_conflict()
        cell = service_health_cell()
        base_action = ActionOption(
            "alice-recover",
            "service-health-action",
            {
                "service": TypedValue("ServiceRef", EntityRef("svc", "Service")),
                "health": TypedValue("HealthState", "recovered"),
            },
        )
        changed_action = replace(
            base_action,
            arguments={
                "service": TypedValue("ServiceRef", EntityRef("svc", "Service")),
                "health": TypedValue("HealthState", "failed"),
            },
        )
        delta = StateDelta((StateDeltaOp("set", "svc", "service.health", TypedValue("HealthState", "recovered")),))
        first = ConflictParticipant("alice", "d1", base_action, _hash("t1"), _hash("s1"), delta, (cell,))
        second = ConflictParticipant("alice", "d1", changed_action, _hash("t1"), _hash("s1"), delta, (cell,))
        other = ConflictParticipant("bob", "d2", replace(base_action, id="bob-recover"), _hash("t2"), _hash("s1"), delta, (cell,))
        self.assertNotEqual(first.content_hash, second.content_hash)
        self.assertNotEqual(
            ConflictResolutionContext(_hash("p"), (first, other), (cell,)).content_hash,
            ConflictResolutionContext(_hash("p"), (second, other), (cell,)).content_hash,
        )

    def test_resolver_identity_binds_hook_and_supported_action_types(self):
        self.require_conflict()
        domain = make_world_domain()
        first = ConflictResolverSpec(
            "resolver", "1", domain.domain_id, domain.version, domain.content_hash,
            ("service-health-action",), PreferLexicalActorResolver(),
        )
        same = ConflictResolverSpec(
            "resolver", "1", domain.domain_id, domain.version, domain.content_hash,
            ("service-health-action",), PreferLexicalActorResolver(),
        )
        changed_hook = replace(first, resolver_hook=PreferLexicalActorResolverV2())
        changed_types = replace(first, supported_action_types=("actor-phase-action", "service-health-action"))
        self.assertEqual(first.content_hash, same.content_hash)
        self.assertNotEqual(first.content_hash, changed_hook.content_hash)
        self.assertNotEqual(first.content_hash, changed_types.content_hash)

    def test_resolution_record_rejects_prior_context_mismatch(self):
        self.require_conflict()
        participant, context = make_direct_participant_context_fixture()
        with self.assertRaises(ValueError):
            ConflictResolutionRecord(
                "resolver",
                _hash("resolver"),
                _hash("different-prior"),
                context,
                participant.original_delta,
            )
```

`make_direct_participant_context_fixture()` is a test helper that returns the same canonical participant/context shape used in the first test; its body must contain the explicit `ActionOption`, `StateDelta`, `StateCellRef`, two participants, and context construction shown above.

- [ ] **Step 3: Add exact V1 compatibility and configured-but-unused locks to `tests/test_narrative_world_transition.py`**

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

Add a `RecordingNoopConflictResolver` with `calls=[]` and an unused configured-resolver test:

```python
    def test_configured_resolver_is_not_called_on_conflict_free_step(self):
        domain, story = make_world_domain(), make_world_story()
        hook = RecordingNoopConflictResolver()
        resolver = ConflictResolverSpec(
            "resolver", "1", domain.domain_id, domain.version, domain.content_hash,
            ("actor-phase-action",), hook,
        )
        base_model = make_transition_model()
        model = replace(base_model, conflict_resolver=resolver)
        self.assertNotEqual(model.content_hash, base_model.content_hash)
        prior = world_state_from_story(story, domain)
        result = advance_world_step(
            story,
            domain,
            prior,
            model,
            (intent("d-bob-phase", "bob-ready"),),
        )
        self.assertEqual(hook.calls, [])
        self.assertEqual(result.conflict_resolutions, ())
        self.assertNotIn("conflict_resolutions", result.to_dict())
```

Keep existing same-value and different-value no-resolver conflict rejection tests unchanged.

- [ ] **Step 4: Build a self-contained conflict fixture for connected components and the three roadmap scenarios**

Extend `make_world_domain()` in test code with:

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

`make_conflict_story(domain)` takes `base = make_world_story()`, replaces its domain hash with `domain.content_hash`, adds `carol`, `dave`, and `svc2`, and appends these exact decisions after the base story's final logical time:

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

Define test transition hooks with exact writes:

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

`make_conflict_world_model(domain, resolver)` declares these matching capabilities:

```text
claim/bid: service.owner via argument service
attack: target agent.health + actor agent.combat
defend: actor agent.health + actor agent.combat
taunt: target agent.combat
```

- [ ] **Step 5: Add deterministic test resolvers**

```python
class ClaimResolver:
    def __call__(self, snapshot, context):
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
```

The lexical tie-break exists only inside `AuctionResolver`.

- [ ] **Step 6: Add all generic component/safety/reference tests**

Add these exact test names in `NarrativeConflictResolutionTests`:

```text
test_two_way_claim_conflict_resolves_once_against_prior_snapshot
test_three_way_transitive_attack_defend_taunt_is_one_component
test_two_independent_claim_components_are_canonical_and_order_invariant
test_noop_delta_does_not_create_component
test_unsupported_action_type_rejects_before_action_and_resolver_hooks
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

Required concrete assertions:

- Two-way claim: resolver called once; recorded snapshot equals `dict(prior.values)`; context participants are `(alice,bob)`; final owner is Alice under `ClaimResolver`.
- Transitive component: Alice attack Bob and Bob defend overlap on `bob.agent.health`; Bob defend and Carol taunt Bob overlap on `bob.agent.combat`; Alice/Carol do not directly overlap; resolver sees one `(alice,bob,carol)` context.
- Independent components: Alice/Bob claim `svc`, Carol/Dave claim `svc2`; exactly two resolution records, canonical order, exact input-order invariance.
- No-op: an empty-delta action never appears in a component.
- Unsupported type: action hooks and resolver hook call lists remain empty.
- Forged action payload: use `_forge()` to replace one transition record's `action` while retaining original hashes; resolver call list remains empty and `WorldTransitionConflictResolutionError` is raised during runtime participant certification.
- Resolver exception: `caught.exception.__cause__` is the exact `RuntimeError("resolver boom")`.
- Capability/schema cases: outside-union cell, wrong TypedValue type, clear with value, set without value, duplicate output cell all raise `WorldTransitionConflictResolutionError`.
- Bad component: prior `WorldState.to_dict()` is byte-for-byte unchanged and no successful result exists.
- Final collisions: resolver-vs-nonconflicting and resolution-vs-resolution collisions both reject; there is no fallback order.
- Claim lineage: losing attempt remains in `result.transitions`; resolution context references both transition hashes.
- Auction: Alice/Bob bid 11, Carol bids 8; explicit resolver tie-break chooses Alice; reversed intent order preserves full result/hash.
- Attack/defense: Bob health healthy, Alice combat attack-failed, Bob combat defense-success; raw Alice attack-success write is not applied.
- Resolved batch hash: assert exactly

```python
expected = stable_content_hash({
    "transitions": [item.to_dict() for item in result.transitions],
    "conflict_resolutions": [item.to_dict() for item in result.conflict_resolutions],
})
self.assertEqual(result.next_state.transition_batch_hash, expected)
```

- [ ] **Step 7: Add `WorldStepResult` forged-resolution constructor locks to `tests/test_narrative_world_transition.py`**

After a valid resolved result is available from the conflict fixture, add tests that use `replace()`/`_forge()` to assert constructor rejection when:

```text
resolution.prior_state_hash != result.prior_state.content_hash
participant.transition_record_hash is absent from result.transitions
the same transition hash appears in two resolution records
participant.action != referenced ActionTransitionRecord.action
participant.original_delta != referenced ActionTransitionRecord.delta
next_state.transition_batch_hash does not bind transitions + resolutions
```

Each case must raise `ValueError` when reconstructing `WorldStepResult`; these are payload-internal consistency checks, not external resolver-model certification.

- [ ] **Step 8: Add heterogeneous scheduler acceptance tests with zero scheduler production changes**

In `tests/test_narrative_simulation.py`, add guarded imports for `ConflictResolverSpec` and `WorldTransitionConflictResolutionError`.

Build `make_heterogeneous_conflict_case()` with these exact model assignments:

```text
a1: intentional, scheduler-alert-choice, selects raise alert
a2: reactive, scheduler-alert-choice, score hook selects raise alert
a3: planning, scheduler-alert-choice, reward hook selects raise alert
```

All three authored actions write the same `service.alert` cell through `alert-control-action`. Configure a resolver that returns `service.alert=True` and records only `(dict(snapshot), context)`.

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

Also assert the conflict resolver receives no `RuntimeEvidenceLedger`, dispatch result, belief, goal, or planning record through its public context fields.

- [ ] **Step 9: Run focused tests and verify the expected RED shape**

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution \
  tests.test_narrative_world_transition \
  tests.test_narrative_simulation -v
```

Expected: existing tests and V1 resolver-free compatibility locks stay GREEN. New Conflict V2 tests fail only because `conflict.py`, optional resolver/result fields, and the typed error are absent. No unrelated regression is acceptable.

- [ ] **Step 10: Commit only tests**

```bash
git add \
  tests/test_narrative_conflict_resolution.py \
  tests/test_narrative_world_transition.py \
  tests/test_narrative_simulation.py
git commit -m "test: define narrative conflict resolution v2 contract"
```

- [ ] **Step 11: Open Draft PR and obtain authoritative exact-head RED**

Open Draft PR `feat: add narrative conflict resolution v2` against `proof/narrative-dynamics-v0`, tracking #27. Accept RED only from the PR-triggered `proof` run on the exact test-only head. Record run id/number and verify all failures belong to Conflict V2 tests.

---

### Task 2: Conflict Records and Resolver Specification GREEN

**Files:**
- Create: `narrative_dynamics/narrative/conflict.py`

**Produces:** `ConflictParticipant`, `ConflictResolutionContext`, `ConflictResolverSpec`, `ConflictResolutionRecord`.

- [ ] **Step 1: Add validation and canonical payload helpers**

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

- [ ] **Step 2: Implement `ConflictParticipant`**

Validate ids/hashes, require `ActionOption` and `StateDelta`, canonicalize unique allowed cells, and require every original delta operation key `(subject_id,state_variable)` exists in the allowed-cell key set. `to_dict()` includes the full `action.to_dict()`, both hashes, canonical delta payload, and allowed cells; `content_hash` hashes `to_dict()`.

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
```

Context constructor must require at least two unique participants, sort by `(actor_id, decision_id, action.id)`, count actual participant writes, derive exact cells with count >= 2, and reject any supplied conflict-cell set that differs. `to_dict()` and `content_hash` are canonical.

- [ ] **Step 4: Implement `ConflictResolverSpec`**

Validate resolver id/version/domain identity, require non-empty unique sorted supported action types and callable hook, and bind:

```python
"implementation_identity": measure_implementation(self.resolver_hook).manifest_identity()
```

into `to_dict()` and `content_hash`.

- [ ] **Step 5: Implement `ConflictResolutionRecord`**

Validate resolver id/hash and prior hash, require `ConflictResolutionContext`, require `context.prior_state_hash == prior_state_hash`, require `StateDelta`, canonicalize `to_dict()`, and expose `content_hash`. Do not claim external model certification.

- [ ] **Step 6: Run record/identity tests**

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_records_bind_full_action_payload_and_exact_conflict_cells \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_action_argument_change_changes_participant_and_context_identity \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_resolver_identity_binds_hook_and_supported_action_types \
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

**Produces:** `WorldTransitionConflictResolutionError`, optional `WorldTransitionModelSpec.conflict_resolver`, resolver declaration validation, exact resolver-free payload compatibility.

- [ ] **Step 1: Add one-way imports and typed error**

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
conflict_resolver: ConflictResolverSpec | None = None
```

When configured, require exact type and exact `(domain_id, domain_version, domain_spec_hash)` match with the world model.

- [ ] **Step 3: Preserve resolver-free `to_dict()` exactly**

Build the current V1 payload first. Add `payload["conflict_resolver_hash"] = self.conflict_resolver.content_hash` only when resolver is configured. Never emit a null resolver key.

- [ ] **Step 4: Validate resolver declarations before action hooks**

Extend `_validate_transition_declarations(domain, model)` so every configured `supported_action_types` entry resolves through `domain._action_type()`. Wrap failure as `WorldTransitionError("conflict resolver names an undeclared action type")` with exact cause.

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
  tests.test_narrative_world_transition.NarrativeWorldTransitionTests.test_resolver_free_world_model_keeps_exact_v1_payload \
  tests.test_narrative_world_transition.NarrativeWorldTransitionTests.test_resolver_free_world_step_keeps_exact_v1_batch_payload \
  tests.test_narrative_world_transition.NarrativeWorldTransitionTests.test_different_value_overlap_is_typed_conflict \
  tests.test_narrative_world_transition.NarrativeWorldTransitionTests.test_same_value_overlap_is_also_typed_conflict \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_unsupported_action_type_rejects_before_action_and_resolver_hooks -v
```

Expected: PASS. Configured conflict-free-step test may still depend on `WorldStepResult.conflict_resolutions` from Task 5, so do not use it as the Task 3 gate.

- [ ] **Step 7: Commit resolver configuration**

```bash
git add narrative_dynamics/narrative/world.py
git commit -m "feat: bind conflict resolver to world model"
```

---

### Task 4: Connected Components, Canonical Context, Resolver Execution GREEN

**Files:**
- Modify: `narrative_dynamics/narrative/world.py`

**Produces private helpers:** `_conflict_components`, `_build_conflict_context`, `_validated_resolver_delta`, `_run_conflict_resolver`.

- [ ] **Step 1: Build connected components from actual record writes**

Canonicalize records by `_transition_record_key`; create an undirected edge only when `_record_write_cells(left) & _record_write_cells(right)` is non-empty; walk each graph component once; discard singleton records; sort members and components lexically by transition key. No-op deltas have empty write sets and create no edge.

- [ ] **Step 2: Recompute exact participant data from story/model records**

For every component record, find canonical `Decision`, require actor equality, find canonical `ActionOption` and require full equality, find `ActionTransitionSpec` by action type, require `_attested_transition_hash(transition) == record.transition_spec_hash`, recompute `_allowed_cells()`, then construct `ConflictParticipant` from exact runtime data.

Any mismatch raises `WorldTransitionConflictResolutionError("conflict participant does not bind canonical transition data")` before resolver execution. This is the production path exercised by the forged-action test.

- [ ] **Step 3: Build exact context**

Count actual write cells across component records, select cells written by at least two records, sort by `_cell_key`, construct `ConflictResolutionContext(prior.content_hash, participants, conflict_cells)`, reconstruct it once from public fields, and require identical payload.

- [ ] **Step 4: Extract pure canonical delta validation and add resolver-specific wrapper**

Move current action-delta structural/type/capability checks into `_canonical_validated_delta()` raising only `TypeError`/`ValueError`. Keep current `_validated_delta()` behavior/message for action hooks. Add `_validated_resolver_delta()` that wraps those raw errors as `WorldTransitionConflictResolutionError("conflict resolver produced an invalid state delta")`.

- [ ] **Step 5: Execute one resolver per component**

Before hook call, require every participant `action.type_name` is supported and call `_attested_resolver_hash()`. Pass the same immutable prior snapshot used by action hooks plus only `ConflictResolutionContext`. Catch `Exception`, not `BaseException`, and wrap hook errors as `WorldTransitionConflictResolutionError("conflict resolver hook failed")` with exact cause. Validate output against the union of participant allowed cells and return one `ConflictResolutionRecord`.

- [ ] **Step 6: Wire resolver invocation into `advance_world_step()` without partial world state**

After all action records exist, derive components. No components use exact V1 `_atomic_result()`. Components without resolver use existing `_reject_write_conflicts()`. Components with resolver produce one resolution record each, but no intermediate `WorldState` is constructed; Task 5 owns final application.

- [ ] **Step 7: Run component/preflight/safety tests**

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_three_way_transitive_attack_defend_taunt_is_one_component \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_two_independent_claim_components_are_canonical_and_order_invariant \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_noop_delta_does_not_create_component \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_forged_action_payload_rejects_before_resolver_hook \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_resolver_exception_is_typed_and_preserves_runtime_error_cause \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_resolver_output_cannot_exceed_union_capability \
  tests.test_narrative_conflict_resolution.NarrativeConflictResolutionTests.test_invalid_resolver_value_clear_set_and_duplicate_output_are_typed -v
```

Expected: component/context and typed failure tests PASS. Successful next-state/lineage tests remain RED.

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

**Produces:** successful resolved `WorldStepResult`, V2 batch lineage, final global collision validation, exact configured-but-unused behavior, atomic world commit.

- [ ] **Step 1: Append optional `conflict_resolutions` to `WorldStepResult`**

```python
conflict_resolutions: tuple[ConflictResolutionRecord, ...] = ()
```

Constructor must canonicalize resolutions, bind each prior hash to `prior_state.content_hash`, require every participant transition hash exists in `transitions`, compare participant actor/decision/action/spec/delta with its referenced transition, forbid one transition appearing in two resolutions, require one internal resolver id/hash across non-empty resolutions, and verify the next-state batch hash.

`to_dict()` adds `conflict_resolutions` only when non-empty.

- [ ] **Step 2: Extend `_transition_batch_hash()` with exact V1 empty-resolution path**

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

- [ ] **Step 3: Build effective mutations and global collision check**

Collect all participant transition hashes from resolutions. Keep raw deltas only from transitions not in that set; add one resolved delta per component. Sort sources canonically. Convert every effective operation to `StateCellRef`; if any cell has two effective writers, raise `WorldTransitionConflictResolutionError("resolved world step still contains overlapping effective writes")` before any next state exists.

- [ ] **Step 4: Runtime-certify resolution lineage against configured resolver**

Before successful next-state construction, require exact resolver id/hash, exact prior hash, exact component transition hashes, exact participant `ActionOption` payloads, and exact original deltas. `ConflictResolutionRecord` constructor alone is not sufficient for these external bindings.

- [ ] **Step 5: Extend `_atomic_result()` while preserving V1 branch exactly**

Empty resolution tuple keeps current raw conflict rejection, V1 mutation loop, V1 batch hash, and omitted-resolution payload. Non-empty resolution tuple skips raw overlap rejection, validates effective collisions, applies only effective mutations to a fresh prior-values copy, computes V2 batch hash, constructs one next `WorldState`, then one `WorldStepResult`.

- [ ] **Step 6: Complete `advance_world_step()`**

Final control flow:

```text
all action hooks -> records -> components
no components -> exact V1 atomic result
components + no resolver -> existing WorldTransitionConflictError
components + resolver -> one resolution per component -> runtime certification -> one atomic resolved result
```

Resolver hooks never see another component's output or a temporary world state.

- [ ] **Step 7: Run full world/reference/forgery suite**

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution \
  tests.test_narrative_world_transition -v
```

Expected: PASS, including reference scenarios, configured-but-unused resolver behavior, V1 hash locks, resolved batch hash, payload-forgery locks, final collision rejection, and atomic prior preservation.

- [ ] **Step 8: Run scheduler acceptance and prove scheduler production unchanged**

```bash
python3 -m unittest tests.test_narrative_simulation -v
git diff 1ad91cd3782e1da0154e3354737bbebf3f369197 -- narrative_dynamics/narrative/simulation.py
```

Expected: all simulation tests PASS and scheduler production diff is empty. Verify resolver context contains only world/action conflict data, not cognitive internals.

- [ ] **Step 9: Add one identity assertion for configured resolver transitivity**

In the existing simulation model identity test, construct two otherwise identical models whose world models differ only by resolver implementation (`PreferLexicalActorResolver` vs `PreferLexicalActorResolverV2`) and assert `SimulationModelSpec.content_hash` differs. This proves resolver attestation propagates through world-model identity into simulation identity without changing scheduler code.

- [ ] **Step 10: Run full Python suite**

```bash
python3 -m unittest discover -s tests -v
```

Expected: `OK`.

- [ ] **Step 11: Commit atomic world integration**

```bash
git add narrative_dynamics/narrative/world.py tests/test_narrative_simulation.py
git commit -m "feat: resolve simultaneous world conflicts atomically"
```

The simulation test change in this commit is only the resolver-identity assertion from Step 9; all behavior-driving simulation tests were already in the authoritative test-only RED.

- [ ] **Step 12: Require exact-head core GREEN**

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

Add `WorldTransitionConflictResolutionError` to existing `world` import block.

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
