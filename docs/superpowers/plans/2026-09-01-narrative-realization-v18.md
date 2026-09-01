# Narrative Realization V18 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Realize a V17 narrative projection into replayable prose or screenplay passages without widening scene-level entitlements or changing accepted story truth.

**Architecture:** Add immutable realization contracts, then a provider-neutral compiler that invokes a JSON language provider separately for each V17 scene. Runtime code validates canonical beat coverage, derives entitlement citations and IDs, records exact provenance, offers a deterministic exact-fact fallback, and exposes only accepted presentation artifacts.

**Tech Stack:** Python 3 standard library (`dataclasses`, `enum`, `json`, immutable tuples), existing `stable_content_hash`, V17 projection contracts, `unittest`/pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-entitlement-bound-narrative-realization-v18-design.md`

## Global Constraints

- V17 remains the sole source of world, cognitive, social, scene, beat, and entitlement truth.
- A provider is called once per scene and receives only that scene's exact entitlement closure.
- Provider responses can group adjacent beats and supply text, but cannot submit entitlement IDs or change beat order.
- Provider prose has `citation_bound`, not formally entailed, assurance; provider-free literal output has `exact_facts` assurance.
- No vendor SDK, network client, local path, secret, hidden chat history, or raw world/trajectory object enters a realization artifact.
- Every accepted artifact is content-addressed, replayable without a provider, and presentation-only.
- All pre-existing V10-V17 deterministic behavior remains compatible.

---

## File Structure

- `narrative_dynamics/abm/situated_realization_contracts.py`: V18 enums, policy/request, per-scene prompt packets, passages, realized scenes, and artifact integrity.
- `narrative_dynamics/abm/situated_realization.py`: scene-local prompt construction, strict provider compilation, exact-fact fallback, replay, and text view.
- `tests/test_network_abm_situated_realization_contracts.py`: constructor, immutability, hashing, and referential-integrity tests.
- `tests/test_network_abm_situated_realization.py`: privacy, provider schema, coverage, provenance, exact-fact, replay, and determinism tests.
- `narrative_dynamics/abm/__init__.py`: V18 public exports.
- `tests/test_network_abm_public_api.py`: V18 public identity checks.
- `README.md`: capability, provider example, assurance boundary, and V19/V20 deferral.

### Task 1: Immutable V18 realization contracts

**Files:**
- Create: `narrative_dynamics/abm/situated_realization_contracts.py`
- Create: `tests/test_network_abm_situated_realization_contracts.py`

**Interfaces:**
- Consumes: `stable_content_hash` and V17 `NarrativeProjection`, `NarrativeScene`, `NarrativeBeat`, `NarrativeEntitlement`.
- Produces: `NarrativeRealizationFormat`, `NarrativeRealizationAssurance`, `NarrativeRealizationProviderIdentity`, `NarrativeRealizationPolicy`, `NarrativeRealizationRequest`, `NarrativeSceneRealizationPrompt`, `NarrativeRealizationPrompt`, `NarrativePassage`, `NarrativeRealizedScene`, and `NarrativeRealizationArtifact`.

- [ ] **Step 1: Write failing contract tests**

Create literal, behavior-oriented tests for these breaks:

```python
def test_request_rejects_a_non_content_addressed_projection():
    with pytest.raises(ValueError, match="projection hash"):
        NarrativeRealizationRequest("render-1", "not-a-hash")


def test_scene_prompt_rejects_an_entitlement_outside_its_beats(projection):
    scene = projection.scenes[0]
    beats = tuple(
        beat for beat in projection.beats if beat.beat_id in scene.beat_ids
    )
    unrelated = next(
        item for item in projection.cut.entitlements
        if item.entitlement_id not in {
            entitlement_id for beat in beats for entitlement_id in beat.entitlement_ids
        }
    )
    with pytest.raises(ValueError, match="exact entitlement closure"):
        NarrativeSceneRealizationPrompt(scene, beats, (unrelated,))


def test_passage_rejects_empty_text():
    with pytest.raises(ValueError, match="text"):
        NarrativePassage(
            "passage-1", "scene-1", ("beat-1",), ("entitlement-1",), ""
        )
```

Also cover invalid format/assurance/provider identity, language longer than 64 characters, more than eight tone tags, tone tags longer than 64 characters, duplicate tone tags, passage-count limits outside 1..64, character limits outside 1..65536, scene prompt beat order, dangling entitlement IDs, duplicate passage IDs, invalid hashes, non-accepted validation results, and stable content hashes under equivalent construction.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest tests/test_network_abm_situated_realization_contracts.py -q
```

Expected: collection fails because `situated_realization_contracts` does not exist.

- [ ] **Step 3: Implement minimal immutable contracts**

Use frozen dataclasses and exact string enums:

```python
class NarrativeRealizationFormat(str, Enum):
    PROSE = "prose"
    SCREENPLAY = "screenplay"


class NarrativeRealizationAssurance(str, Enum):
    CITATION_BOUND = "citation_bound"
    EXACT_FACTS = "exact_facts"
```

Validate scene prompts against the exact ordered V17 beat tuple and exact set of referenced entitlements. Preserve passage and scene order. Sort only set-like fields such as tone tags and entitlement IDs. Every value exposes `to_dict()` and `content_hash`; fixed schema and template hashes are exposed by `NarrativeRealizationPrompt`.

- [ ] **Step 4: Run contract tests and verify GREEN**

Run the contract test file and require zero failures or warnings.

- [ ] **Step 5: Commit Task 1**

```powershell
git add narrative_dynamics/abm/situated_realization_contracts.py tests/test_network_abm_situated_realization_contracts.py
git commit -m "feat(abm): add narrative realization contracts"
```

### Task 2: Scene-local provider compilation and replay

**Files:**
- Create: `narrative_dynamics/abm/situated_realization.py`
- Create: `tests/test_network_abm_situated_realization.py`

**Interfaces:**
- Consumes: all Task 1 contracts and a provider exposing `identity` plus `complete_json(*, task, payload)`.
- Produces: `build_narrative_realization_prompt(...)`, `compile_narrative_realization(...)`, `replay_narrative_realization(...)`, and `render_narrative_realization_text(...)`.

- [ ] **Step 1: Write failing privacy and prompt-boundary tests**

Use real V17 projections from the situated office fixtures. The provider double records complete payloads but assertions target prompt behavior:

```python
def test_limited_realization_never_sends_hidden_private_facts():
    projection = project_situated_narrative(
        private_inspection_then_public_clue_story(), limited_policy("bob")
    )
    provider = RecordingProvider(valid_one_passage_per_scene_response)
    prompt = build_narrative_realization_prompt(
        projection, policy(), request(projection), provider
    )
    compile_narrative_realization(prompt, provider)
    serialized = json.dumps(provider.payloads, sort_keys=True)
    assert "restructuring" not in serialized
    assert "approved" not in serialized


def test_each_provider_call_contains_only_one_scene_context():
    artifact, provider, projection = compile_multi_scene_fixture()
    assert len(provider.payloads) == len(projection.scenes)
    for payload, scene in zip(provider.payloads, projection.scenes):
        assert payload["scene"]["scene_id"] == scene.scene_id
        assert {item["scene_id"] for item in [payload["scene"]]} == {scene.scene_id}
```

The production mutation caught by these tests is replacing scene-local payload construction with a global projection payload or resolving private percepts back to objective events.

- [ ] **Step 2: Run the two tests and verify RED**

Expected: import failure for `situated_realization`.

- [ ] **Step 3: Implement prompt construction and provider protocol checks**

Implement:

```python
def build_narrative_realization_prompt(
    projection: NarrativeProjection,
    policy: NarrativeRealizationPolicy,
    request: NarrativeRealizationRequest,
    provider: object,
) -> NarrativeRealizationPrompt:
    ...
```

Require request/projection hash equality and a valid immutable provider identity captured before invocation. Build scene prompts in `projection.cut.scene_ids` order, beats in each `scene.beat_ids` order, and entitlements as the exact union of those beats. Payload construction serializes only a single `NarrativeSceneRealizationPrompt`, policy, fixed schema, and limits.

- [ ] **Step 4: Write failing response-validation tests**

Add separate tests proving rejection of:

- non-object response and extra/missing keys;
- empty or excessive passage lists;
- unknown, duplicate, omitted, reordered, non-contiguous, or cross-scene beat IDs;
- empty/NUL/oversized text;
- provider identity mutation during any scene call.

Use literal malformed responses. Do not assert provider call counts except where one-call-per-scene is itself the privacy boundary.

- [ ] **Step 5: Run response tests and verify RED**

Expected: each named test fails because compilation or its validation branch is absent.

- [ ] **Step 6: Implement strict compilation**

Call the provider with task `situated_narrative_scene_realization_v1`. Require exact response keys and canonical flattened beat coverage. Derive each passage's entitlement IDs from its referenced beats, derive its stable ID with `stable_content_hash`, record the complete response hash, and create `NarrativeRealizedScene` and `NarrativeRealizationArtifact` values. Never retain the raw response outside accepted passage text and its hash.

- [ ] **Step 7: Write failing provenance and replay tests**

Assert that:

```python
assert artifact.projection_hash == projection.content_hash
assert artifact.assurance is NarrativeRealizationAssurance.CITATION_BOUND
assert artifact.validation_result == "accepted"
assert replay_narrative_realization(projection, artifact) is artifact
assert render_narrative_realization_text(artifact) == "literal expected text"
```

Also reconstruct artifacts with a wrong projection hash, prompt hash, schema hash, template hash, scene order, beat coverage, entitlement closure, response hash, or passage ID and require rejection by construction or replay. Replay must not receive or invoke a provider.

- [ ] **Step 8: Run all runtime tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_network_abm_situated_realization_contracts.py tests/test_network_abm_situated_realization.py -q
```

- [ ] **Step 9: Commit Task 2**

```powershell
git add narrative_dynamics/abm/situated_realization.py tests/test_network_abm_situated_realization.py
git commit -m "feat(abm): compile scene-local narrative prose"
```

### Task 3: Exact-fact fallback, public API, and documentation

**Files:**
- Modify: `narrative_dynamics/abm/situated_realization.py`
- Modify: `tests/test_network_abm_situated_realization.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `tests/test_network_abm_public_api.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Tasks 1-2 and V17 public projection values.
- Produces: `realize_narrative_exact_facts(...)` and all V18 names through `narrative_dynamics.abm`.

- [ ] **Step 1: Write failing exact-fact tests**

Assert that provider-free realization:

- calls no external object;
- emits one passage per beat in canonical scene order;
- serializes only literal `key=value` facts from the beat's exact entitlements;
- cites the exact derived entitlement IDs;
- reports `EXACT_FACTS` assurance and the fixed built-in provider identity;
- repeats byte-equivalently with an equal content hash;
- replays against the same projection and rejects another projection.

- [ ] **Step 2: Run exact-fact tests and verify RED**

Expected: `realize_narrative_exact_facts` is missing.

- [ ] **Step 3: Implement the deterministic fallback**

Build accepted passages directly from scene prompts. Use a stable, language-neutral literal form containing only sorted exact entitlement `key=value` facts; beat ID and kind remain passage metadata rather than output text. Reuse the same artifact validation and replay path with `EXACT_FACTS` assurance.

- [ ] **Step 4: Run exact-fact tests and verify GREEN**

Require deterministic equality and exact literal assertions.

- [ ] **Step 5: Write failing public API test**

Extend `tests/test_network_abm_public_api.py` to import every V18 contract/function from `narrative_dynamics.abm` and assert object identity with the definitions in their source modules.

- [ ] **Step 6: Export and document V18**

Add all public V18 names to `narrative_dynamics/abm/__init__.py` and `__all__`. Add README section `V18 entitlement-bound narrative realization` showing an injected JSON provider, exact-fact fallback, replay, prose/screenplay modes, scene-local context, and the honest `citation_bound` versus `exact_facts` distinction. State that V19 runtime unification and V20 authoring/visualization remain separate phases.

- [ ] **Step 7: Run targeted and regression verification**

Run:

```powershell
python -m pytest tests/test_network_abm_situated_realization_contracts.py tests/test_network_abm_situated_realization.py tests/test_network_abm_situated_projection_contracts.py tests/test_network_abm_situated_projection.py tests/test_network_abm_public_api.py -q
python -m unittest discover -s tests -p "test_network_abm*.py"
python -m compileall -q narrative_dynamics
git diff --check
```

Record the known repository-wide pytest collection issue separately; do not modify unrelated legacy tests in V18.

- [ ] **Step 8: Commit Task 3**

```powershell
git add narrative_dynamics/abm/situated_realization.py tests/test_network_abm_situated_realization.py narrative_dynamics/abm/__init__.py tests/test_network_abm_public_api.py README.md
git commit -m "feat(abm): expose entitlement-bound narrative realization"
```

## Plan Self-Review

- Spec coverage: scene-local context, strict beat coverage, derived entitlement closure, provider identity, response hashing, replay, exact-fact fallback, public API, and honest assurance semantics each map to a task and observable test.
- Scope: V18 is independently usable and testable; V19 network/situated unification and V20 product UI are intentionally separate plans because they are independent subsystems.
- Placeholder scan: no placeholder markers or unspecified behavior remain.
- Type consistency: Task 2 consumes the exact Task 1 contract names; Task 3 adds one function without renaming prior interfaces.
- Mutation check: tests fail for global-context leakage, unauthorized entitlement widening, provider identity drift, malformed schema, beat omission/reorder, invalid provenance hashes, and accidental provider use by replay/exact-fact fallback.
