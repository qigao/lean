# Scenario Package V21.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load a path-safe JSON scenario tree and compile its physical, social, agent, story, knowledge/asset, initial-state, and run documents into one exact content-addressed `CompiledSituatedScenario` over the existing V20 runtime.

**Architecture:** Separate source loading from semantic compilation. The loader returns immutable canonical JSON documents without filesystem paths; focused compilers resolve stable IDs and construct existing V10-V20 public contracts; the final compiled value binds every source hash, runtime model, spatial map, initial subsystem state, story plan, catalogs, and run policy. V21.1 performs no simulation, LLM call, asset fetch, output publication, or Blender launch.

**Tech Stack:** Python standard library (`dataclasses`, `enum`, `json`, `pathlib`), existing `stable_content_hash`, existing situated ABM contracts, Tiled JSON adapter, `unittest`/`pytest` tests.

**Spec:** `docs/superpowers/specs/2026-09-01-scenario-io-live-projection-v21-design.md`

## Global Constraints

- JSON is the only V21.1 interchange format; add no YAML or provider dependency.
- Never execute source content, deserialize Python class names, import a source-declared factory, or fetch a URI.
- Reject absolute paths, `..`, symlink escapes, duplicate JSON keys, hash mismatches, non-finite numbers, duplicate roles/IDs, and incomplete cross-document coverage.
- Machine-local package roots and filenames do not enter compiled scenario identity or public errors.
- Recompute every hash and construct existing runtime values through public constructors.
- Unsupported norm effects or story predicates fail closed; descriptive metadata cannot affect runtime state.
- Preserve V10-V20 authority, privacy, and fixed-roster invariants.
- Do not run the previously stopped repository-wide suite; use focused tests plus `test_network_abm*.py` regression at the final gate.

---

### Task 1: Source manifest contracts and path-safe JSON loading

**Files:**
- Create: `narrative_dynamics/abm/scenario_package_contracts.py`
- Create: `narrative_dynamics/abm/scenario_package.py`
- Create: `tests/test_network_abm_scenario_package.py`

**Interfaces:**
- Consumes: `stable_content_hash(value) -> str`.
- Produces: `ScenarioDocumentRole`, `ScenarioDocumentLocator`, `ScenarioPackageManifest`, `ScenarioSourceDocument`, `ScenarioPackageSource`, and `load_situated_scenario_package(root) -> ScenarioPackageSource`.

- [ ] **Step 1: Write manifest canonicalization tests**

```python
def test_manifest_and_source_identity_ignore_root_and_document_order(self):
    first = write_minimal_package(self.root / "first", reverse=False)
    second = write_minimal_package(self.root / "second", reverse=True)
    self.assertEqual(
        load_situated_scenario_package(first),
        load_situated_scenario_package(second),
    )

def test_loader_rejects_parent_escape_before_reading_document(self):
    root = write_minimal_package(self.root / "escape")
    replace_manifest_path(root, "physical.world", "../secret.json")
    with self.assertRaisesRegex(ValueError, "path"):
        load_situated_scenario_package(root)
```

- [ ] **Step 2: Run Task 1 tests and verify RED**

Run: `python -m pytest tests/test_network_abm_scenario_package.py -q`

Expected: import failure because `scenario_package` and its contracts do not exist.

- [ ] **Step 3: Implement immutable source contracts**

```python
class ScenarioDocumentRole(str, Enum):
    PHYSICAL_WORLD = "physical.world"
    PHYSICAL_PERCEPTION = "physical.perception"
    PHYSICAL_INITIAL_STATE = "physical.initial_state"
    PHYSICAL_MAP = "physical.map"
    SOCIAL_INSTITUTIONS = "social.institutions"
    SOCIAL_RELATIONSHIPS = "social.relationships"
    SOCIAL_NORMS = "social.norms"
    AGENT = "agent"
    STORY_OUTLINE = "story.outline"
    STORY_INTERVENTIONS = "story.interventions"
    KNOWLEDGE_CATALOG = "knowledge.catalog"
    KNOWLEDGE_ACCESS = "knowledge.access"
    ASSET_CATALOG = "asset.catalog"
    RUN = "run"

@dataclass(frozen=True)
class ScenarioDocumentLocator:
    role: ScenarioDocumentRole
    logical_id: str
    relative_path: str
    expected_hash: str

@dataclass(frozen=True)
class ScenarioPackageManifest:
    scenario_id: str
    version: str
    documents: tuple[ScenarioDocumentLocator, ...]
    schema: str = "narrative-dynamics.scenario-package/v1"

@dataclass(frozen=True)
class ScenarioSourceDocument:
    role: ScenarioDocumentRole
    logical_id: str
    schema: str
    value: Mapping[str, object]

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())

@dataclass(frozen=True)
class ScenarioPackageSource:
    scenario_id: str
    version: str
    documents: tuple[ScenarioSourceDocument, ...]
    manifest_hash: str
```

Validate non-empty text, exact `sha256:` hashes, unique `(role, logical_id)`, exact singleton-role cardinality, canonical ordering, recursively frozen JSON values, and stable `to_dict`/`content_hash` values. Require at least physical world, perception, initial state, one agent, social relationships, story outline, knowledge/access catalogs, asset catalog, and run documents. Map and institution/norm/intervention documents are optional only when the run document selects their declared fallback.

- [ ] **Step 4: Implement strict JSON and path loading**

```python
def _reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("scenario JSON object keys must be unique")
        result[key] = value
    return result

def _resolve_document(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("scenario document path must stay under the package root")
    resolved = (root / relative).resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError("scenario document path must stay under the package root")
    return resolved
```

Decode with `json.loads(..., parse_constant=reject_constant, object_pairs_hook=_reject_duplicate_pairs)`, require an object root and exact document schema, recompute `stable_content_hash` from canonical content, compare it with the manifest locator, and discard resolved paths before returning `ScenarioPackageSource`. Enforce a 1 MiB limit per JSON document and a 16 MiB limit for a Tiled map.

- [ ] **Step 5: Add malformed-input coverage**

Cover duplicate manifest roles, duplicate agent logical IDs, duplicate JSON keys,
absolute paths, symlink escape where supported, invalid UTF-8, NaN/Infinity,
oversized files, missing files, wrong schema, wrong expected hash, and an undeclared
file having no effect on identity.

- [ ] **Step 6: Run Task 1 tests and verify GREEN**

Run: `python -m pytest tests/test_network_abm_scenario_package.py -q`

Expected: all source-contract and loader tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add narrative_dynamics/abm/scenario_package_contracts.py narrative_dynamics/abm/scenario_package.py tests/test_network_abm_scenario_package.py
git commit -m "feat(abm): load canonical scenario packages"
```

### Task 2: Social, story, knowledge, asset, and run authoring contracts

**Files:**
- Create: `narrative_dynamics/abm/scenario_authoring_contracts.py`
- Create: `tests/test_network_abm_scenario_authoring_contracts.py`

**Interfaces:**
- Consumes: stable agent/place/action IDs resolved later by the compiler.
- Produces: `ScenarioExecutionMode`, `ScenarioInstitution`, `ScenarioMembership`, `ScenarioRelationship`, `ScenarioNormEffect`, `ScenarioNorm`, `ScenarioSocialWorld`, `ScenarioPredicateKind`, `ScenarioPredicate`, `ScenarioSceneDependency`, `ScenarioSceneContract`, `ScenarioStoryAct`, `ScenarioStoryPlan`, `ScenarioResourceKind`, `ScenarioKnowledgeResource`, `ScenarioKnowledgeCatalog`, `ScenarioAssetResource`, `ScenarioAssetCatalog`, `ScenarioResourceGrant`, `ScenarioRunPolicy`.

- [ ] **Step 1: Write failing graph and policy tests**

```python
def test_institution_containment_and_story_dependencies_are_acyclic(self):
    with self.assertRaisesRegex(ValueError, "institution.*cycle"):
        ScenarioSocialWorld(
            institutions=(
                ScenarioInstitution("firm", "company", "team"),
                ScenarioInstitution("team", "department", "firm"),
            ),
            memberships=(), relationships=(), norms=(),
        )
    with self.assertRaisesRegex(ValueError, "story.*cycle"):
        ScenarioStoryPlan(
            "plan", "1", ScenarioExecutionMode.HYBRID,
            acts=(ScenarioStoryAct("act", ("a", "b")),),
            scenes=(scene("a"), scene("b")),
            dependencies=(
                ScenarioSceneDependency("a", "b"),
                ScenarioSceneDependency("b", "a"),
            ),
        )

def test_relationship_direction_and_resource_grants_are_canonical(self):
    relationship = ScenarioRelationship("alice", "bob", "reports_to", 1.0)
    self.assertNotEqual(
        relationship,
        ScenarioRelationship("bob", "alice", "reports_to", 1.0),
    )
```

- [ ] **Step 2: Run Task 2 tests and verify RED**

Run: `python -m pytest tests/test_network_abm_scenario_authoring_contracts.py -q`

Expected: import failure because the authoring contracts do not exist.

- [ ] **Step 3: Implement social contracts and DAG validation**

Use frozen dataclasses and canonical tuples. `ScenarioSocialWorld` validates one
acyclic institution parent forest, unique `(agent_id, institution_id, role_id)`
memberships, unique directed `(source_agent_id, target_agent_id, relationship_type)`
edges, finite relationship strengths in `[-1, 1]`, and norm effects drawn only from:

```python
class ScenarioNormEffect(str, Enum):
    ALLOW_ACTION = "allow_action"
    DENY_ACTION = "deny_action"
    REQUIRE_KNOWLEDGE_GRANT = "require_knowledge_grant"
    REQUIRE_RELATIONSHIP = "require_relationship"
    DESCRIPTIVE = "descriptive"
```

Non-descriptive norms require registered action/resource/relation IDs; descriptive
norms retain bounded text but expose no enforcement hook.

- [ ] **Step 4: Implement story-plan contracts**

```python
class ScenarioExecutionMode(str, Enum):
    AUTHORED = "authored"
    HYBRID = "hybrid"
    SANDBOX = "sandbox"

@dataclass(frozen=True)
class ScenarioSceneContract:
    scene_id: str
    place_ids: tuple[str, ...]
    participant_agent_ids: tuple[str, ...]
    preconditions: tuple[ScenarioPredicate, ...]
    exit_predicates: tuple[ScenarioPredicate, ...]
    allowed_intervention_kinds: tuple[str, ...]
    desired_outcome_ids: tuple[str, ...]
    maximum_rounds: int
```

`ScenarioPredicate` accepts a finite enum of `agent_at`, `passage_open`,
`object_at`, `agent_holds`, `belief_at_least`, `claim_status`, and
`relationship_at_least`, with typed subject/object/value fields. Reject unknown
predicates, duplicate scene membership across acts, missing dependency endpoints,
dependency cycles, zero/negative limits, and authored plans with no scene.

- [ ] **Step 5: Implement resource catalogs and run policy**

Knowledge and asset values require stable IDs, content hashes, bounded URI/media
metadata, authority/license tags, and canonical access grants. A grant has exactly
one subject scope (`agent`, `role`, `institution`, or `public`) and a tuple of known
resource IDs. `ScenarioRunPolicy` validates positive round/checkpoint/output limits,
one execution mode matching the story plan, output kind allowlists, journal policy,
and Blender mode `none`, `final_blend`, or `live_mirror`.

- [ ] **Step 6: Run Task 2 tests and verify GREEN**

Run: `python -m pytest tests/test_network_abm_scenario_authoring_contracts.py -q`

Expected: all contract, cycle, identity, numeric, and canonical-order tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add narrative_dynamics/abm/scenario_authoring_contracts.py tests/test_network_abm_scenario_authoring_contracts.py
git commit -m "feat(abm): define scenario authoring graphs"
```

### Task 3: Compile physical, perception, cognition, memory, and social models

**Files:**
- Create: `narrative_dynamics/abm/scenario_compiler.py`
- Create: `tests/scenario_package_fixtures.py`
- Create: `tests/test_network_abm_scenario_compiler.py`

**Interfaces:**
- Consumes: `ScenarioPackageSource`, existing V10-V20 constructors.
- Produces: `compile_situated_scenario_package(source) -> CompiledSituatedScenario` and `ScenarioCompilationError(document_role, json_pointer, code)`.

- [ ] **Step 1: Create a complete data-authored office fixture**

Build `write_law_firm_package(root)` from JSON dictionaries, not Python runtime
objects. It must define lobby/meeting/archive places, two passages, one document
object, Alice/Bob/client agents, visual/auditory/interaction edges, per-agent
hypotheses/actions/goals, memory/recall policy, social topics/relationships, a hybrid
two-scene outline, knowledge/asset catalogs, initial state, and run policy. Compute
manifest expected hashes through a test-only canonical helper.

- [ ] **Step 2: Write compiler RED tests**

```python
def test_package_compiles_to_exact_v20_runtime_and_spatial_map(self):
    source = load_situated_scenario_package(write_law_firm_package(self.root))
    compiled = compile_situated_scenario_package(source)
    self.assertEqual(compiled.scenario_id, "law-firm-case")
    self.assertEqual(
        compiled.runtime_model.percept_memory_model.cognitive_model.world_model,
        compiled.spatial_map.world_model,
    )
    self.assertRegex(compiled.content_hash, r"^sha256:[0-9a-f]{64}$")

def test_unknown_cross_document_id_reports_role_and_pointer(self):
    root = write_law_firm_package(self.root)
    mutate_json(root / "agents/alice.json", "/body/initial_place", "missing")
    refresh_manifest_hash(root, "agent", "alice")
    with self.assertRaisesRegex(
        ScenarioCompilationError,
        r"agent:alice.*\/body\/initial_place",
    ):
        compile_situated_scenario_package(load_situated_scenario_package(root))
```

- [ ] **Step 3: Run Task 3 tests and verify RED**

Run: `python -m pytest tests/test_network_abm_scenario_compiler.py -q`

Expected: import failure for `compile_situated_scenario_package`.

- [ ] **Step 4: Implement leaf compilers in dependency order**

Implement private pure functions with exact return types:

```python
_compile_world(source) -> SituatedWorldModel
_compile_perception(source, world) -> SituatedPerceptionModel
_compile_cognition(source, world) -> SituatedCognitiveModel
_compile_percept_memory(source, perception, cognition)
    -> SituatedPerceptMemoryCognitiveModel
_compile_social_memory(source, cognition) -> SituatedSocialMemoryModel
_compile_runtime_model(source, percept_memory, social_memory)
    -> SituatedNetworkRuntimeModel
_compile_spatial_map(source, world) -> SituatedSpatialMap
```

Each helper requires exact object keys, rejects booleans where numbers are expected,
maps strings through existing enums, preserves authored action/scene order only where
semantically ordered, and lets existing public constructors enforce their own deeper
invariants. Wrap failures as `ScenarioCompilationError` with a stable code and nearest
logical JSON pointer; do not include source values or machine paths in messages.

- [ ] **Step 5: Compile social/story/catalog/run metadata and exact coverage**

Parse Task 2 values from their documents, then verify all agent/place/passage/object/
action/hypothesis/topic/role/institution/resource references against the global
catalog. Require exact fixed roster coverage across world, perception, cognition,
memory, social, relationships, knowledge grants, and initial state. Reject runtime
norm effects not supported by the compiled action/resource/relation catalogs.

- [ ] **Step 6: Run Task 3 focused tests and verify GREEN**

Run: `python -m pytest tests/test_network_abm_scenario_compiler.py tests/test_network_abm_scenario_authoring_contracts.py -q`

Expected: the complete package compiles; malformed cross-document references fail
with stable role/pointer diagnostics.

- [ ] **Step 7: Commit Task 3**

```bash
git add narrative_dynamics/abm/scenario_compiler.py tests/scenario_package_fixtures.py tests/test_network_abm_scenario_compiler.py
git commit -m "feat(abm): compile situated scenario packages"
```

### Task 4: Compile and bind the initial state

**Files:**
- Modify: `narrative_dynamics/abm/scenario_package_contracts.py`
- Modify: `narrative_dynamics/abm/scenario_compiler.py`
- Modify: `tests/test_network_abm_scenario_compiler.py`

**Interfaces:**
- Consumes: compiled model/spatial/authoring values from Task 3.
- Produces: `CompiledSituatedScenario` with exact initial story, `SituatedCognitiveState`, and `SituatedSocialMemoryState`; `initialize_compiled_scenario(database_path, scenario) -> SituatedNetworkRuntimeState`.

- [ ] **Step 1: Write failing initial-state authority tests**

```python
def test_compiled_initial_state_binds_every_subsystem_and_database(self):
    compiled = compile_fixture(self.root)
    with TemporaryDirectory() as temporary:
        state = initialize_compiled_scenario(
            Path(temporary) / "memory.sqlite3", compiled
        )
    self.assertEqual(state.story, compiled.initial_story)
    self.assertEqual(state.snapshot.story_hash, compiled.initial_story.content_hash)

def test_initial_state_rejects_unknown_holder_and_duplicate_object_location(self):
    root = write_law_firm_package(self.root)
    corrupt_initial_object_holder(root, "case-file", "unknown")
    with self.assertRaisesRegex(ScenarioCompilationError, "initial_state"):
        compile_situated_scenario_package(load_situated_scenario_package(root))
```

- [ ] **Step 2: Run Task 4 tests and verify RED**

Run: `python -m pytest tests/test_network_abm_scenario_compiler.py -q -k initial_state`

Expected: failure because compiled initial values and initializer are absent.

- [ ] **Step 3: Implement `CompiledSituatedScenario`**

```python
@dataclass(frozen=True)
class CompiledSituatedScenario:
    scenario_id: str
    version: str
    package_hash: str
    source_document_hashes: tuple[tuple[str, str, str], ...]
    runtime_model: SituatedNetworkRuntimeModel
    spatial_map: SituatedSpatialMap
    initial_story: SituatedStory
    initial_cognitive_state: SituatedCognitiveState
    initial_social_state: SituatedSocialMemoryState
    social_world: ScenarioSocialWorld
    story_plan: ScenarioStoryPlan
    knowledge_catalog: ScenarioKnowledgeCatalog
    asset_catalog: ScenarioAssetCatalog
    run_policy: ScenarioRunPolicy
```

Its `to_dict` contains only IDs, versions, source hashes, and exact child content
hashes. Validate exact world/model/story/cognitive/social/spatial bindings and matching
execution modes; canonicalize source-document identities by `(role, logical_id)`.
Expose `content_hash` as `stable_content_hash(to_dict())`; `package_hash` remains the
identity of the canonical source package and is one input to the compiled identity.

- [ ] **Step 4: Reconstruct initial subsystem state**

Compile `SituatedWorldState` from explicit agent locations, passage states, and
object holder/place values; call `validate_situated_state`; create a story with the
exact perception model; initialize percept-memory cognition and social memory through
their public initializers. No database is opened during compilation. Implement
`initialize_compiled_scenario` by initializing the SQLite percept-memory store and
calling `initialize_situated_network_runtime` with the exact compiled checkpoint.

- [ ] **Step 5: Verify deterministic path-independent compilation**

Compile semantically identical packages under two roots with reversed JSON object,
document, agent, relationship, and catalog input order. Assert equal compiled values
and hashes. Change one initial door state and assert package, initial story, compiled
scenario, and initialized V19 state hashes all change.

- [ ] **Step 6: Run Task 4 tests and verify GREEN**

Run: `python -m pytest tests/test_network_abm_scenario_package.py tests/test_network_abm_scenario_authoring_contracts.py tests/test_network_abm_scenario_compiler.py -q`

Expected: all V21.1 package, graph, compilation, and initialization tests pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add narrative_dynamics/abm/scenario_package_contracts.py narrative_dynamics/abm/scenario_compiler.py tests/test_network_abm_scenario_compiler.py
git commit -m "feat(abm): bind compiled scenario checkpoints"
```

### Task 5: Public API, authoring example, and authoritative gates

**Files:**
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `tests/test_network_abm_public_api.py`
- Modify: `README.md`
- Create: `examples/law_firm_scenario/scenario.json`
- Create: all JSON/Tiled documents declared by the example manifest under `examples/law_firm_scenario/`

**Interfaces:**
- Consumes: Tasks 1-4 public contracts/functions.
- Produces: documented, importable V21.1 input API and one complete data-only scenario.

- [ ] **Step 1: Extend the exact public API test and verify RED**

Add Task 1-4 public names to the expected `narrative_dynamics.abm.__all__` set and
assert every export is the source definition. Run:

`python -m pytest tests/test_network_abm_public_api.py -q`

Expected: failures for missing V21.1 exports.

- [ ] **Step 2: Export V21.1 APIs**

Export source/authoring/compiled contracts plus
`load_situated_scenario_package`, `compile_situated_scenario_package`, and
`initialize_compiled_scenario`. Do not import any provider, Blender, network, YAML,
or optional SDK from `narrative_dynamics.abm`.

- [ ] **Step 3: Add and verify the law-firm package**

Materialize the Task 3 fixture as human-readable example JSON/Tiled files. Its README
section shows:

```python
source = load_situated_scenario_package("examples/law_firm_scenario")
scenario = compile_situated_scenario_package(source)
state = initialize_compiled_scenario("law-firm-memory.sqlite3", scenario)
```

Document tree-versus-graph semantics, stable ID references, physical/social input,
story execution modes, knowledge/asset boundaries, path safety, compile diagnostics,
and that V21.2 output/live projection is the next separate phase.

- [ ] **Step 4: Run focused V21.1 and V20 compatibility tests**

Run:

```text
python -m pytest tests/test_network_abm_scenario_package.py tests/test_network_abm_scenario_authoring_contracts.py tests/test_network_abm_scenario_compiler.py tests/test_network_abm_situated_spatial_map.py tests/test_blender_replay.py tests/test_network_abm_public_api.py -q
```

Expected: all pass; the real Blender test may skip unless `BLENDER_EXECUTABLE` is
explicitly supplied.

- [ ] **Step 5: Run authorized regression and static gates**

Run:

```text
python -m unittest discover -s tests -p "test_network_abm*.py" -q
python -m compileall -q narrative_dynamics tests
git diff --check
```

Expected: all network ABM tests pass, compilation exits zero, and diff check is clean.
Do not run the stopped repository-wide suite.

- [ ] **Step 6: Self-review against the V21 design**

Confirm V21.1 covers safe loading, path-independent identity, all physical/social/
agent/story/catalog/run inputs, exact existing-runtime construction, initial-state
authority, public APIs, and a data-only example. Confirm output records, bus,
coordinator, live Blender transport, and LLM/retrieval adapters remain scoped to
V21.2-V21.5 and are not silently stubbed or accepted-but-ignored.

- [ ] **Step 7: Commit Task 5**

```bash
git add narrative_dynamics/abm/__init__.py tests/test_network_abm_public_api.py README.md examples/law_firm_scenario
git commit -m "feat(abm): expose scenario package authoring"
```
