# Narrative Projection V17 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deterministically project an accepted situated ABM trajectory into supported narrative beats, scenes, and an immutable cut for objective, named-agent-limited, or declared multi-POV narration.

**Architecture:** Add a contract-only module for policies and content-addressed projection artifacts, then a projector that derives entitlements from world events and optional V15.1/V14 cognitive-social trajectory records. The projector filters and orders beats with deterministic salience, omission-gap, and causal-coverage rules; it never calls an LLM and never promotes prose to world truth.

**Tech Stack:** Python 3 standard library (`dataclasses`, `enum`, `math`, immutable mappings), existing `stable_content_hash`, `unittest`/pytest.

**Spec:** `docs/superpowers/specs/2026-08-31-simulated-story-production-architecture-design.md`

## Global Constraints

- World history, story projection, and narrative realization remain distinct immutable products.
- The story projector may select supported facts but must not create actions, dialogue, knowledge, or state.
- Narrator authority is exactly one of objective, named-agent limited POV, or declared multi-POV.
- Temporal ordering is chronological or explicitly authored; a flashback changes cut order, never `round_index` or event time.
- Every beat cites exact content-addressed support and every cut carries the exact entitlements a later renderer may mention.
- Limited and multi-POV output may contain only percepts and private cognitive/social artifacts owned by an authorized POV agent; unavailable objective event payloads must not leak.
- Salience, scene breaks, omission-gap enforcement, and causal coverage are deterministic and provider-free.
- SQLite paths and other machine-local paths never enter content hashes.
- All pre-existing V10-V16 deterministic behavior remains compatible.

---

## File Structure

- `narrative_dynamics/abm/situated_projection_contracts.py`: enums, policy validation, support/fact/entitlement values, beat/scene/cut/projection integrity and hashing.
- `narrative_dynamics/abm/situated_projection.py`: authoritative extraction, POV-safe entitlement construction, deterministic selection/order/grouping, cognitive/social augmentation.
- `tests/test_network_abm_situated_projection_contracts.py`: contract mutation and hash tests.
- `tests/test_network_abm_situated_projection.py`: objective/limited/multi-POV, privacy, salience, causal coverage, flashback, cognition/social, and replay tests.
- `narrative_dynamics/abm/__init__.py`: public V17 imports and exports.
- `README.md`: concise V17 capability and boundary documentation.

### Task 1: Immutable narrative projection contracts

**Files:**
- Create: `narrative_dynamics/abm/situated_projection_contracts.py`
- Create: `tests/test_network_abm_situated_projection_contracts.py`

**Interfaces:**
- Consumes: `narrative_dynamics.contracts.stable_content_hash`.
- Produces: `NarrativeAuthority`, `NarrativeTemporalOrder`, `NarrativeBeatKind`, `NarrativeEntitlementScope`, `NarrativeSupportRef`, `NarrativeFact`, `NarrativeEntitlement`, `NarrativeProjectionPolicy`, `NarrativeBeat`, `NarrativeScene`, `NarrativeCut`, and `NarrativeProjection`.

- [ ] **Step 1: Write failing contract tests**

Create tests that import the new module and specify the public constructor behavior. Use literal hashes (`"sha256:" + "a" * 64`) rather than hashes computed by code under test. The minimum test cases are:

```python
class NarrativeProjectionContractTests(unittest.TestCase):
    def test_limited_policy_requires_exactly_one_named_agent(self):
        with self.assertRaisesRegex(ValueError, "exactly one POV agent"):
            NarrativeProjectionPolicy(
                "bob-cut", "1.0", NarrativeAuthority.AGENT_LIMITED,
                pov_agent_ids=(),
            )

    def test_authored_order_requires_unique_event_ids(self):
        with self.assertRaisesRegex(ValueError, "authored event order"):
            NarrativeProjectionPolicy(
                "flashback", "1.0", NarrativeAuthority.OBJECTIVE,
                temporal_order=NarrativeTemporalOrder.AUTHORED,
                authored_event_order=("event-2", "event-2"),
            )

    def test_projection_rejects_beat_support_outside_entitlement_bundle(self):
        support = NarrativeSupportRef("world_event", "event-1", HASH_A)
        beat = NarrativeBeat(
            "beat-1", NarrativeBeatKind.PHYSICAL,
            1, 1, "office", None, ("alice",),
            1.0, (support,), ("missing-entitlement",), (),
        )
        scene = NarrativeScene("scene-1", ("beat-1",), 1, 1, "office", None)
        cut = NarrativeCut("cut-1", ("scene-1",), ())
        with self.assertRaisesRegex(ValueError, "entitlement"):
            NarrativeProjection(HASH_A, None, POLICY, (beat,), (scene,), cut)

    def test_content_hash_is_stable_under_input_mapping_order(self):
        left = NarrativeProjectionPolicy(
            "stable", "1.0", NarrativeAuthority.OBJECTIVE,
            salience_weights={
                NarrativeBeatKind.PHYSICAL: 0.5,
                NarrativeBeatKind.INFORMATION: 0.8,
            },
        )
        right = NarrativeProjectionPolicy(
            "stable", "1.0", NarrativeAuthority.OBJECTIVE,
            salience_weights={
                NarrativeBeatKind.INFORMATION: 0.8,
                NarrativeBeatKind.PHYSICAL: 0.5,
            },
        )
        self.assertEqual(left.content_hash, right.content_hash)
```

Also cover: objective authority rejects POV IDs; multi-POV requires at least two unique IDs; chronological order rejects authored IDs; all numeric policy fields are finite and bounded; facts have unique keys inside an entitlement; content hashes have the `sha256:` format; beats/scenes/cut reject dangling or duplicate IDs; a private entitlement requires an owner while an objective entitlement rejects one; and a projection requires every beat in exactly one scene and every scene in exactly one cut.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest tests/test_network_abm_situated_projection_contracts.py -q
```

Expected: collection fails with `ModuleNotFoundError: narrative_dynamics.abm.situated_projection_contracts`.

- [ ] **Step 3: Implement minimal immutable contracts**

Implement string enums with these exact values:

```python
class NarrativeAuthority(str, Enum):
    OBJECTIVE = "objective"
    AGENT_LIMITED = "agent_limited"
    MULTI_POV = "multi_pov"

class NarrativeTemporalOrder(str, Enum):
    CHRONOLOGICAL = "chronological"
    AUTHORED = "authored"

class NarrativeBeatKind(str, Enum):
    PHYSICAL = "physical"
    INFORMATION = "information"
    BELIEF_SHIFT = "belief_shift"
    ACTION_REVERSAL = "action_reversal"
    MEMORY_RECALL = "memory_recall"
    CLAIM_REVISION = "claim_revision"
    RELATIONSHIP_CHANGE = "relationship_change"
    CAUSAL_PAYOFF = "causal_payoff"

class NarrativeEntitlementScope(str, Enum):
    OBJECTIVE = "objective"
    PRIVATE = "private"
```

Use frozen dataclasses. `NarrativeFact` is a non-empty `key`/`value` pair. `NarrativeSupportRef` is `(artifact_kind, artifact_id, artifact_hash)`. `NarrativeEntitlement` is `(entitlement_id, scope, owner_agent_id, round_index, facts, supporting_artifacts)` and sorts facts/support deterministically. `NarrativeBeat` has the constructor used by the test above and stores `cause_beat_ids`. `NarrativeScene` stores exact beat IDs plus time/place/active POV. `NarrativeCut` stores ordered scene IDs and the complete entitlement tuple.

`NarrativeProjectionPolicy` fields are:

```python
policy_id: str
version: str
authority: NarrativeAuthority
pov_agent_ids: tuple[str, ...] = ()
temporal_order: NarrativeTemporalOrder = NarrativeTemporalOrder.CHRONOLOGICAL
authored_event_order: tuple[str, ...] = ()
salience_weights: Mapping[NarrativeBeatKind, float] = field(default_factory=dict)
minimum_salience: float = 0.0
maximum_event_omission_gap: int = 0
required_causal_coverage: float = 1.0
scene_round_gap: int = 1
maximum_scene_beats: int = 8
include_beliefs: bool = True
include_memories: bool = True
include_claim_revisions: bool = True
include_relationship_changes: bool = True
```

An absent weight defaults to `1.0` in `weight_for(kind)`. Zero omission gap means no gap forcing. `NarrativeProjection` fields are `(story_hash, trajectory_hash, policy, beats, scenes, cut)`. Its validation enforces complete referential integrity without requiring beats to be chronological, because authored ordering is legal. Every value exposes `to_dict()` and `content_hash`.

- [ ] **Step 4: Run contract tests and verify GREEN**

Run the contract test file. Expected: all tests pass with no warning.

- [ ] **Step 5: Commit Task 1**

```powershell
git add narrative_dynamics/abm/situated_projection_contracts.py tests/test_network_abm_situated_projection_contracts.py
git commit -m "feat(abm): add narrative projection contracts"
```

### Task 2: POV-safe world-event projection, salience, causality, and scene cuts

**Files:**
- Create: `narrative_dynamics/abm/situated_projection.py`
- Create: `tests/test_network_abm_situated_projection.py`

**Interfaces:**
- Consumes: all Task 1 contracts; `SituatedStory`, `objective_timeline`, `perceptual_timeline`, `SituatedActionKind`, and `SituatedPerceptFidelity`.
- Produces: `project_situated_narrative(story, policy, *, trajectory=None) -> NarrativeProjection` for story-only projections in this task. Task 3 extends the non-`None` trajectory path.

- [ ] **Step 1: Write failing objective and limited-POV tests**

Build a two-round office story from existing `tests.situated_fixtures` and perception fixtures: Alice privately inspects the memo, then moves or tells. Tests must assert observable behavior, including:

```python
def test_objective_projection_cites_exact_world_events():
    story = objective_office_story()
    projection = project_situated_narrative(story, objective_policy())
    event_hashes = {event.content_hash for round_ in story.rounds for event in round_.events}
    cited = {
        ref.artifact_hash
        for beat in projection.beats
        for ref in beat.supporting_artifacts
        if ref.artifact_kind == "world_event"
    }
    assert cited == event_hashes
    assert all(item.scope is NarrativeEntitlementScope.OBJECTIVE for item in projection.cut.entitlements)

def test_bob_limited_projection_does_not_leak_alice_private_inspection():
    story = private_inspection_then_public_clue_story()
    projection = project_situated_narrative(story, limited_policy("bob"))
    serialized = json.dumps(projection.to_dict(), sort_keys=True)
    assert "restructuring" not in serialized
    assert "approved" not in serialized
    assert all(
        entitlement.owner_agent_id == "bob"
        for entitlement in projection.cut.entitlements
    )

def test_detected_percept_exposes_signal_without_actor_kind_or_outcome():
    projection = project_situated_narrative(detected_story(), limited_policy("bob"))
    facts = {
        fact.key: fact.value
        for entitlement in projection.cut.entitlements
        for fact in entitlement.facts
    }
    assert facts["fidelity"] == "detected"
    assert "actor_agent_id" not in facts
    assert "kind" not in facts
    assert "outcome" not in facts
```

Also test: an unknown POV agent is rejected; multi-POV emits separate private entitlements owned only by declared agents; limited support references private `SituatedPercept.content_hash`, not full event payload hashes as a world-event entitlement; repeated projection has byte-equivalent `to_dict()` and equal content hash.

- [ ] **Step 2: Run the new behavior tests and verify RED**

Run the three named tests. Expected: import failure for `situated_projection`.

- [ ] **Step 3: Implement authoritative extraction and entitlement construction**

For objective authority, iterate `objective_timeline(story)` and build one event-derived beat per event. Facts may include only exact event fields (`actor_agent_id`, `kind`, `place_id`, `success`, `outcome`, target when present, and each evidence detail). Use `INFORMATION` for `INSPECT` and `TELL`, otherwise `PHYSICAL`.

For limited or multi-POV authority, require `story.perception_model`; iterate `perceptual_timeline(model, story, agent_id)` independently for each authorized POV. Construct facts only from `SituatedPercept.to_dict()` fields that are present under its fidelity. Never resolve its source event to recover hidden fields. Support is the percept ID/hash and the entitlement scope is private with that POV owner.

Use stable IDs derived from semantic keys via `stable_content_hash`, for example `beat:<digest>` and `entitlement:<digest>`; never use Python `hash()` or process randomness.

- [ ] **Step 4: Run the POV tests and verify GREEN**

Run the whole projection test file. Expected: the world/POV tests pass.

- [ ] **Step 5: Write failing salience, causal, authored-order, and grouping tests**

Add literal assertions for these mutations:

- A high minimum salience can omit ordinary beats, but `maximum_event_omission_gap=2` forces a supported beat at least every two eligible event positions.
- `required_causal_coverage=1.0` forces all *authorized* direct cause beats of a selected event; a limited cut never names an unauthorized cause.
- A causally linked selected event whose visible cause is included is classified as `CAUSAL_PAYOFF` while retaining the underlying event facts.
- `AUTHORED` policy with reversed exact event IDs reverses scene order but leaves each beat's `round_index` unchanged.
- A policy-authored event ID absent from the accepted story is rejected.
- Same place, POV, continuous time, and capacity group beats in one scene; place/POV change, a round gap beyond `scene_round_gap`, or `maximum_scene_beats` starts a new scene.
- Every projected beat occurs once in scenes, and every scene occurs once in the cut.

- [ ] **Step 6: Run the new tests and verify RED**

Expected: at least one assertion fails because selection/order/grouping is not implemented.

- [ ] **Step 7: Implement deterministic selection, ordering, and grouping**

Compute score as `policy.weight_for(kind) * feature_magnitude`, using `1.0` for event/percept magnitude in this task. Select scores at or above `minimum_salience`; force first/last eligible event, omission-gap anchors when the configured gap is nonzero, and the earliest chronological fraction of authorized direct causes needed to meet `required_causal_coverage`. Cause enforcement never widens the POV entitlement set.

For authored order, require `authored_event_order` to be an exact permutation of selected distinct source event IDs and sort those event-derived beat groups by that position. Preserve `(round_index, sequence)` inside every beat. Chronological sorting uses `(round_index, sequence, active_pov, beat_id)`.

Group ordered beats greedily. Continue a scene only when place and active POV match, round distance is within `scene_round_gap`, and capacity remains. Keep an available direct cause/payoff pair adjacent before applying ordinary grouping. Scene and cut IDs are content-derived.

- [ ] **Step 8: Run projection tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_network_abm_situated_projection_contracts.py tests/test_network_abm_situated_projection.py -q
```

Expected: all V17 tests pass without warnings.

- [ ] **Step 9: Commit Task 2**

```powershell
git add narrative_dynamics/abm/situated_projection.py tests/test_network_abm_situated_projection.py
git commit -m "feat(abm): project POV-safe narrative scenes"
```

### Task 3: Cognitive/social beats, public API, and V17 documentation

**Files:**
- Modify: `narrative_dynamics/abm/situated_projection.py`
- Modify: `tests/test_network_abm_situated_projection.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `tests/test_network_abm_public_api.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `SituatedPerceptSocialCognitiveTrajectory`, decisions and recalls from each round, and prior/next `SituatedSocialMemoryState` values.
- Produces: the final trajectory-aware `project_situated_narrative(...)` and all V17 names through `narrative_dynamics.abm`.

- [ ] **Step 1: Write failing trajectory-aware tests**

Use an actual `SituatedPerceptSocialCognitiveTrajectory` fixture or construct a valid trajectory through the existing simulator; do not mock its records. Assert:

```python
def test_projection_emits_supported_private_cognitive_and_social_changes(trajectory):
    projection = project_situated_narrative(
        trajectory.final_story,
        limited_policy("bob"),
        trajectory=trajectory,
    )
    kinds = {beat.kind for beat in projection.beats}
    assert NarrativeBeatKind.BELIEF_SHIFT in kinds
    assert NarrativeBeatKind.MEMORY_RECALL in kinds
    assert NarrativeBeatKind.CLAIM_REVISION in kinds
    assert NarrativeBeatKind.RELATIONSHIP_CHANGE in kinds
    assert projection.trajectory_hash == trajectory.content_hash
    assert all(
        item.owner_agent_id == "bob"
        for item in projection.cut.entitlements
    )

def test_include_flags_remove_private_internal_categories(trajectory):
    policy = replace(
        limited_policy("bob"),
        include_beliefs=False,
        include_memories=False,
        include_claim_revisions=False,
        include_relationship_changes=False,
    )
    projection = project_situated_narrative(
        trajectory.final_story, policy, trajectory=trajectory
    )
    assert not ({
        NarrativeBeatKind.BELIEF_SHIFT,
        NarrativeBeatKind.MEMORY_RECALL,
        NarrativeBeatKind.CLAIM_REVISION,
        NarrativeBeatKind.RELATIONSHIP_CHANGE,
    } & {beat.kind for beat in projection.beats})
```

Also test: only the owner POV receives cognitive/social state; objective authority may carry private entitlements for all owners but each remains owner-tagged; an unrelated story/trajectory hash is rejected; belief-shift salience equals half L1 distance, recall salience uses `evidence_weight`, relationship salience uses `abs(trust_delta)+abs(affinity_delta)`, terminal claim transitions have magnitude `1.0`; action reversal is emitted only when the selected action changes from that agent's prior projected round; every such beat cites the exact decision, recall admission, social update, claim, or relationship artifact hash.

- [ ] **Step 2: Run the trajectory tests and verify RED**

Expected: the story-only projector rejects or ignores `trajectory`, causing the new behavior assertions to fail.

- [ ] **Step 3: Implement cognitive/social extraction without widening authority**

Validate `trajectory.final_story == story`. For each trajectory round:

- Emit a `BELIEF_SHIFT` beat when a decision's prior/posterior beliefs differ and the owner is authorized.
- Emit `ACTION_REVERSAL` only after a previous selected action exists for that same agent and changes.
- Emit one `MEMORY_RECALL` per non-consolidated recall admission with its exact admission hash and memory/event IDs; do not resolve memory text beyond the admitted artifact.
- Diff claims by `claim_id` and relationships by directed `(observer_agent_id, source_agent_id)` between each social update's prior and next states. Emit only changed records and owner them to the observer.
- Apply policy include flags before selection.
- Objective policy authorizes every owner but preserves `PRIVATE` scope and owner tags; limited/multi policy authorizes only declared POV owners.

Use fact keys that name the exact typed fields, numeric values serialized with `repr(float_value)`, and support refs that contain the exact source artifact `content_hash`. Do not include database paths or query strings.

- [ ] **Step 4: Run V17 tests and verify GREEN**

Run both V17 test files. Expected: all pass without warnings.

- [ ] **Step 5: Write failing public API test**

Extend `tests/test_network_abm_public_api.py` to import the V17 contracts and `project_situated_narrative` from `narrative_dynamics.abm`, then assert those objects are the same definitions as their source modules. This should fail before export wiring.

- [ ] **Step 6: Export V17 and document its boundary**

Import and add every Task 1 public type plus `project_situated_narrative` to `narrative_dynamics/abm/__init__.py` and `__all__`.

Add a README subsection titled `V17 deterministic narrative projection` explaining:

- accepted world/cognitive/social history becomes beats, scenes, and a cut;
- objective, one-agent limited, and declared multi-POV are supported;
- limited POV works from sanitized percepts and owner-private state only;
- authored order permits flashbacks without changing event time;
- the output is a content-addressed truth/entitlement packet for V18, not prose and not a world mutation.

- [ ] **Step 7: Run targeted and regression verification**

Run:

```powershell
python -m pytest tests/test_network_abm_situated_projection_contracts.py tests/test_network_abm_situated_projection.py tests/test_network_abm_public_api.py -q
python -m unittest discover -s tests -p "test_network_abm*.py"
python -m compileall -q narrative_dynamics
```

Expected: V17 target tests and the existing `unittest` network ABM suite pass, and compilation exits zero. Record the already-baselined pytest helper-collection issue separately rather than changing unrelated V14 tests in this feature.

- [ ] **Step 8: Commit Task 3**

```powershell
git add narrative_dynamics/abm/situated_projection.py tests/test_network_abm_situated_projection.py narrative_dynamics/abm/__init__.py tests/test_network_abm_public_api.py README.md
git commit -m "feat(abm): integrate trajectory narrative projection"
```

## Plan Self-Review

- Spec coverage: authority, temporal order, salience, omission gap, causal support, beats, scenes, cut entitlements, privacy, replay hashing, and style-independent V18 handoff each map to an explicit task/test.
- Deferred by design: V18 prose/screenplay rendering, dialogue generation, and V19 branch direction are consumers of this projection and are not implemented here.
- Placeholder scan: no TBD/TODO or unspecified implementation step remains.
- Type consistency: Task 2 creates the final function signature with optional `trajectory`; Task 3 fills that path without renaming the API. All later imports match Task 1 class names.
