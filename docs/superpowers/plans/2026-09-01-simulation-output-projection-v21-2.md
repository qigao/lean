# Simulation Output Projection V21.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn every accepted V19 situated-network round into deterministic, typed, audience-scoped output records and an atomically replayable public JSONL journal.

**Architecture:** Preserve V19 as the transition authority while retaining its exact private round evidence in the returned round result. Project that evidence into immutable payload-specific records, assemble one canonical atomic batch, derive audience views before serialization, and persist only public views. The output core is transport-neutral so later pure-Web JSON-RPC/H2/WSS services and remote Agent executors consume the same hashes and schemas.

**Tech Stack:** Python standard library (`dataclasses`, `enum`, `json`, `os`, `pathlib`, immutable tuples), existing V10–V21.1 ABM contracts, `unittest`/`pytest`.

**Spec:** `docs/superpowers/specs/2026-09-01-scenario-io-live-projection-v21-design.md`; `docs/superpowers/specs/2026-09-01-web-simulation-platform-v21-design.md`

## Global Constraints

- The editor is pure Web; V21.2 adds no VS Code, React Flow, AntV X6, PixiJS, Monaco, browser bundle, HTTP server, or WSS dependency.
- External transport is reserved as JSON-RPC 2.0 over HTTPS/HTTP2 and WSS; add no Protobuf or gRPC type or dependency.
- V19 remains the sole world-transition authority; output projection performs no provider, Blender, network, asset, SQLite mutation, or world mutation.
- Raw private percept, decision, recall, claim, and relationship content is emitted only in an `agent` record naming exactly one owner.
- `public` conveniences never serialize `agent`, `analyst`, or `internal` records.
- Every record and batch binds `CompiledSituatedScenario.content_hash`, exact V19 prior/next state hashes, the accepted V19 round hash, and canonical content hashes.
- Output record order is command results, objective events, state delta, private percepts, decisions, memory, social changes, metrics, story progress, narrative references, Blender delta, diagnostics; identity breaks ties.
- JSONL writes stage, flush, fsync, and `os.replace` one complete new journal; replay performs no Agent/provider call.
- Preserve V10–V21.1 path safety, privacy, fixed-roster, semantic identity, Tiled, SQLite, and Blender invariants.
- Do not run the stopped repository-wide suite; use focused V21.2 tests and `python -m unittest discover -s tests -p "test_network_abm*.py" -q`.

---

### Task 1: Retain exact transition evidence in V19 round results

**Files:**
- Modify: `narrative_dynamics/abm/situated_network_contracts.py`
- Modify: `narrative_dynamics/abm/situated_network.py`
- Modify: `tests/test_network_abm_situated_network_contracts.py`
- Modify: `tests/test_network_abm_situated_network.py`

**Interfaces:**
- Consumes: `SituatedPerceptSocialCognitiveRoundResult` returned by `simulate_situated_percept_social_cognitive_round(...)`.
- Produces: optional `SituatedNetworkRoundResult.transition` whose exact story/cognitive/social chains bind `prior_state` and `next_state`; newly simulated rounds always populate it.

- [ ] **Step 1: Add failing evidence-retention tests**

In the existing real-round test, assert:

```python
result = simulate_situated_network_round(database_path, model, initial)

self.assertIsNotNone(result.transition)
self.assertEqual(result.transition.next_story, result.next_state.story)
self.assertEqual(result.transition.next_cognitive_state, result.next_state.cognitive_state)
self.assertEqual(result.transition.next_social_state, result.next_state.social_state)
self.assertEqual(
    result.to_dict()["transition_hash"],
    result.transition.content_hash,
)
```

Add one contract test using `dataclasses.replace` to attach a transition from a different prior branch and assert `ValueError` containing `transition`.

The mutation caught is dropping the private transition artifact after the atomic V19 commit or accepting evidence from another branch.

- [ ] **Step 2: Run Task 1 RED**

Run:

```text
python -m pytest tests/test_network_abm_situated_network_contracts.py tests/test_network_abm_situated_network.py -q -k "transition_evidence"
```

Expected: failure because `SituatedNetworkRoundResult` has no `transition` field.

- [ ] **Step 3: Add the optional evidence contract**

Import `SituatedPerceptSocialCognitiveRoundResult` and extend the dataclass:

```python
transition: SituatedPerceptSocialCognitiveRoundResult | None = None
```

When non-null, validate all of these exact relationships:

```python
if self.transition.cognitive_round.cognitive_round.prior_state != self.prior_state.cognitive_state:
    raise ValueError("network round transition must bind prior cognitive state")
if self.transition.social_update.prior_state != self.prior_state.social_state:
    raise ValueError("network round transition must bind prior social state")
if self.transition.next_story != self.next_state.story:
    raise ValueError("network round transition must bind next story")
if self.transition.next_cognitive_state != self.next_state.cognitive_state:
    raise ValueError("network round transition must bind next cognitive state")
if self.transition.next_social_state != self.next_state.social_state:
    raise ValueError("network round transition must bind next social state")
```

Add `"transition_hash": None if transition is None else transition.content_hash` to `to_dict()`. Keeping the field optional preserves direct V19 contract construction, but V21.2 projection will reject missing evidence.

Pass `advanced` as the fifth argument when `simulate_situated_network_round(...)` constructs the result.

- [ ] **Step 4: Run Task 1 GREEN and regression**

Run:

```text
python -m pytest tests/test_network_abm_situated_network_contracts.py tests/test_network_abm_situated_network.py -q
```

Expected: all pass without warnings.

- [ ] **Step 5: Commit Task 1**

```text
git add narrative_dynamics/abm/situated_network_contracts.py narrative_dynamics/abm/situated_network.py tests/test_network_abm_situated_network_contracts.py tests/test_network_abm_situated_network.py
git commit -m "feat(abm): retain situated round evidence"
```

### Task 2: Typed output records, payloads, batches, and audience views

**Files:**
- Create: `narrative_dynamics/abm/simulation_output_contracts.py`
- Create: `tests/test_network_abm_simulation_output_contracts.py`

**Interfaces:**
- Consumes: existing `SituatedPercept`, `SituatedCognitiveDecision`, `SituatedMemoryRecallResult`, `SituatedNetworkEmergenceMetrics`, and stable `content_hash` contracts.
- Produces: `SimulationOutputRecord`, `SimulationOutputBatch`, `SimulationOutputView`, `SimulationAudienceCapability`, payload dataclasses, `output_record_sort_key(record)`.

- [ ] **Step 1: Add failing payload and record contract tests**

Create literal/unit tests covering:

```python
def test_agent_record_requires_matching_owner(self):
    payload = SimulationPrivatePerceptPayload(self.percept)
    with self.assertRaisesRegex(ValueError, "owner"):
        SimulationOutputRecord(
            "stream-1", SCENARIO_HASH, 1, 1, STATE_HASH,
            SimulationOutputKind.PERCEPT_PRIVATE,
            SimulationOutputAudience.AGENT,
            "bob",
            (),
            payload,
        )

def test_public_record_rejects_private_payload(self):
    payload = SimulationPrivatePerceptPayload(self.percept)
    with self.assertRaisesRegex(ValueError, "audience"):
        SimulationOutputRecord(
            "stream-1", SCENARIO_HASH, 1, 1, STATE_HASH,
            SimulationOutputKind.PERCEPT_PRIVATE,
            SimulationOutputAudience.PUBLIC,
            None,
            (),
            payload,
        )
```

Add batch tests proving duplicate/gapped sequences, mixed scenario hashes, wrong state hash, non-canonical record order, and record/payload kind mismatch are rejected. Add one view test proving an Agent capability sees public records plus only its own Agent records, while a public capability sees public records only.

Each test must name the production mutation it catches in a one-line comment above the assertion.

- [ ] **Step 2: Run Task 2 RED**

Run:

```text
python -m pytest tests/test_network_abm_simulation_output_contracts.py -q
```

Expected: import failure because the output-contract module does not exist.

- [ ] **Step 3: Implement exact enums and payload classes**

Define string enums using these exact values:

```python
class SimulationOutputAudience(str, Enum):
    PUBLIC = "public"
    OBJECTIVE = "objective"
    AGENT = "agent"
    ANALYST = "analyst"
    INTERNAL = "internal"

class SimulationOutputKind(str, Enum):
    STATE_DELTA = "state.delta"
    EVENT_OBJECTIVE = "event.objective"
    PERCEPT_PRIVATE = "percept.private"
    AGENT_DECISION = "agent.decision"
    MEMORY_UPDATE = "memory.update"
    SOCIAL_UPDATE = "social.update"
    NETWORK_METRICS = "network.metrics"
    STORY_PROGRESS = "story.progress"
    NARRATIVE_SCENE = "narrative.scene"
    BLENDER_DELTA = "blender.delta"
    COMMAND_RESULT = "command.result"
    DIAGNOSTIC = "diagnostic"
```

Implement frozen, validated, hashable payload classes with `to_dict()` and `content_hash`:

```python
SimulationStateDeltaPayload(prior_snapshot_hash, next_snapshot_hash,
    changed_agent_ids, changed_passage_ids, changed_object_ids)
SimulationObjectiveEventPayload(event_id, event_hash, action_id, action_kind,
    actor_agent_id, place_id, target_id, success, cause_event_ids)
SimulationPrivatePerceptPayload(percept)
SimulationAgentDecisionPayload(decision)
SimulationMemoryUpdatePayload(agent_id, prior_mind_hash, next_mind_hash,
    recalled_memory_ids, admitted_memory_ids)
SimulationSocialUpdatePayload(observer_agent_id, prior_claim_hashes,
    next_claim_hashes, prior_relationship_hashes, next_relationship_hashes,
    admitted_evidence_ids)
SimulationNetworkMetricsPayload(metrics)
SimulationStoryProgressPayload(active_scene_id, completed_scene_ids, status)
SimulationNarrativeScenePayload(scene_id, projection_hash, realization_hash)
SimulationBlenderDeltaPayload(agent_places, passage_states, object_placements)
SimulationCommandResultPayload(command_id, accepted, reason_code)
SimulationDiagnosticPayload(code, message)
```

`agent_places` is a canonical tuple of `(agent_id, place_id)`, `passage_states` a tuple of `(passage_id, open)`, and `object_placements` a tuple of `(object_id, place_id_or_none, holder_agent_id_or_none)`. Reject duplicate identities and non-finite/arbitrary mapping payloads.

- [ ] **Step 4: Implement record, batch, and view contracts**

Use these schemas and signatures:

```python
SIMULATION_OUTPUT_RECORD_SCHEMA = "narrative-dynamics.simulation-output-record/v1"
SIMULATION_OUTPUT_BATCH_SCHEMA = "narrative-dynamics.simulation-output-batch/v1"
SIMULATION_OUTPUT_VIEW_SCHEMA = "narrative-dynamics.simulation-output-view/v1"

@dataclass(frozen=True)
class SimulationOutputRecord:
    stream_id: str
    scenario_hash: str
    sequence: int
    round_index: int
    state_hash: str
    kind: SimulationOutputKind
    audience: SimulationOutputAudience
    owner_agent_id: str | None
    source_artifact_hashes: tuple[str, ...]
    payload: SimulationOutputPayload
    schema: str = SIMULATION_OUTPUT_RECORD_SCHEMA

@dataclass(frozen=True)
class SimulationOutputBatch:
    stream_id: str
    scenario_hash: str
    prior_state_hash: str
    next_state_hash: str
    round_result_hash: str
    first_sequence: int
    last_sequence: int
    records: tuple[SimulationOutputRecord, ...]
    checkpoint: bool = False
    schema: str = SIMULATION_OUTPUT_BATCH_SCHEMA
```

The batch requires a non-empty contiguous sequence, exact record round/state/scenario/stream binding, and `tuple(sorted(records, key=output_record_sort_key)) == records`. The sort key uses the fixed kind rank followed by owner ID, payload identity, and record sequence.

Define `SimulationAudienceCapability` and `SimulationOutputView.from_batch(...)`. Visibility is exact:

```text
public    -> public
objective -> public + objective
agent A   -> public + agent records owned by A
analyst   -> public + objective + analyst
internal  -> all
```

The view retains source record sequences and `source_batch_hash`; it does not require sequences to be contiguous after filtering.

- [ ] **Step 5: Run Task 2 GREEN**

Run:

```text
python -m pytest tests/test_network_abm_simulation_output_contracts.py -q
```

Expected: all pass without warnings.

- [ ] **Step 6: Commit Task 2**

```text
git add narrative_dynamics/abm/simulation_output_contracts.py tests/test_network_abm_simulation_output_contracts.py
git commit -m "feat(abm): define scoped simulation outputs"
```

### Task 3: Deterministic V19-to-V21 output projector

**Files:**
- Create: `narrative_dynamics/abm/simulation_output.py`
- Create: `tests/test_network_abm_simulation_output.py`

**Interfaces:**
- Consumes: `CompiledSituatedScenario`, evidence-bearing `SituatedNetworkRoundResult`, Task 2 contracts.
- Produces: `project_simulation_output(scenario, round_result, *, stream_id, first_sequence=1) -> SimulationOutputBatch` and `filter_simulation_output(batch, capability) -> SimulationOutputView`.

- [ ] **Step 1: Add failing real-round projection tests**

Use the committed law-firm package fixture, compile it, initialize a temporary SQLite store, and advance one real V19 round. Mutate the fixture's `run.json` before compilation so `allowed_output_kinds` contains all kinds emitted by this task.

Assert literal behavior:

```python
batch = project_simulation_output(
    scenario,
    round_result,
    stream_id="law-firm-run-1",
    first_sequence=41,
)

self.assertEqual(batch.scenario_hash, scenario.content_hash)
self.assertEqual(batch.prior_state_hash, round_result.prior_state.content_hash)
self.assertEqual(batch.next_state_hash, round_result.next_state.content_hash)
self.assertEqual(batch.round_result_hash, round_result.content_hash)
self.assertEqual(
    tuple(record.sequence for record in batch.records),
    tuple(range(41, 41 + len(batch.records))),
)
self.assertTrue(any(record.kind is SimulationOutputKind.NETWORK_METRICS for record in batch.records))
self.assertTrue(any(record.kind is SimulationOutputKind.PERCEPT_PRIVATE for record in batch.records))
self.assertTrue(any(record.kind is SimulationOutputKind.AGENT_DECISION for record in batch.records))
```

Add privacy tests proving a detected percept does not regain actor/kind/outcome/details, no objective event payload contains a TELL message or arbitrary event details, and every private record owner matches its payload Agent.

Add determinism tests projecting the same accepted round twice and asserting complete batch equality/content hash equality; changing only `first_sequence` changes record and batch identity but not payload hashes.

Add a missing-evidence test constructing a legacy `SituatedNetworkRoundResult` with `transition=None` and expecting `ValueError("transition evidence")`.

- [ ] **Step 2: Run Task 3 RED**

Run:

```text
python -m pytest tests/test_network_abm_simulation_output.py -q
```

Expected: import failure because the projector does not exist.

- [ ] **Step 3: Implement projection helpers**

Implement pure helpers that derive:

```python
_objective_event_payloads(round_result)
_state_delta_payload(prior_state, next_state)
_private_percept_payloads(transition)
_decision_payloads(transition)
_memory_payloads(transition)
_social_payloads(transition)
_network_metrics_payload(next_state)
_blender_delta_payload(prior_state, next_state)
```

State and Blender deltas compare stable IDs in exact prior/next world states. Objective events include only the typed safe fields in `SimulationObjectiveEventPayload`; never copy `SituatedActionIntent.message`, `SituatedWorldEvent.outcome`, or `details`. Private percept payloads reuse the already sanitized `SituatedPercept`. Social payloads are split by `observer_agent_id` before construction.

No `story.progress`, `narrative.scene`, `command.result`, or `diagnostic` record is fabricated in V21.2 because no accepted coordinator/narrative/command artifact exists yet.

- [ ] **Step 4: Implement canonical record assembly**

Use this exact public signature:

```python
def project_simulation_output(
    scenario: CompiledSituatedScenario,
    round_result: SituatedNetworkRoundResult,
    *,
    stream_id: str,
    first_sequence: int = 1,
) -> SimulationOutputBatch:
```

Validate exact runtime-model binding and non-null transition evidence. Generate payload candidates, discard kinds not in `scenario.run_policy.allowed_output_kinds`, sort candidates by the Task 2 kind rank and identity, then assign contiguous sequences. Set each record's `state_hash` to `round_result.next_state.content_hash` and source hashes to the exact accepted artifact hashes used by the payload.

`filter_simulation_output(...)` delegates to `SimulationOutputView.from_batch` and performs no serialization.

- [ ] **Step 5: Run Task 3 GREEN and focused privacy regressions**

Run:

```text
python -m pytest tests/test_network_abm_simulation_output.py tests/test_network_abm_situated_network.py -q
```

Expected: all pass without warnings.

- [ ] **Step 6: Commit Task 3**

```text
git add narrative_dynamics/abm/simulation_output.py tests/test_network_abm_simulation_output.py
git commit -m "feat(abm): project atomic simulation outputs"
```

### Task 4: Atomic public JSONL journal, replay, public API, and final gates

**Files:**
- Create: `narrative_dynamics/abm/simulation_output_journal.py`
- Create: `tests/test_network_abm_simulation_output_journal.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `tests/test_network_abm_public_api.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 2 `SimulationOutputBatch`/`SimulationOutputView`, Task 3 projector.
- Produces: `SimulationPublicJournal`, `write_public_simulation_journal(path, batch, *, parent_journal_hash=None)`, `replay_public_simulation_journal(path)`.

- [ ] **Step 1: Add failing real journal tests**

Create two consecutive real batches and assert:

```python
first = write_public_simulation_journal(path, batch_one)
second = write_public_simulation_journal(path, batch_two)
replayed = replay_public_simulation_journal(path)

self.assertEqual(replayed, second)
self.assertEqual(len(replayed.batches), 2)
self.assertTrue(all(
    record.audience is SimulationOutputAudience.PUBLIC
    for view in replayed.batches
    for record in view.records
))
self.assertNotIn("percept.private", path.read_text(encoding="utf-8"))
self.assertNotIn("agent.decision", path.read_text(encoding="utf-8"))
```

Add tests for tampered payload hash, record hash, view hash, batch hash, sequence regression, state-chain mismatch, scenario/stream mismatch, invalid UTF-8, duplicate JSON keys, non-finite JSON constants, and truncated final line. Assert failures expose no private payload value or machine-local path.

Add an atomicity test that patches `os.replace` to fail and proves the previous journal bytes remain unchanged; this mock is justified at the external atomic publication boundary and assertions remain on the real file bytes.

- [ ] **Step 2: Run Task 4 RED**

Run:

```text
python -m pytest tests/test_network_abm_simulation_output_journal.py tests/test_network_abm_public_api.py -q
```

Expected: import failures for missing journal/public exports.

- [ ] **Step 3: Implement journal contracts and strict decoding**

Use exact schemas:

```python
SIMULATION_PUBLIC_JOURNAL_SCHEMA = "narrative-dynamics.simulation-public-journal/v1"
SIMULATION_PUBLIC_JOURNAL_BATCH_SCHEMA = "narrative-dynamics.simulation-public-journal-batch/v1"
```

The first JSONL line contains:

```json
{"schema":"narrative-dynamics.simulation-public-journal/v1","stream_id":"...","scenario_hash":"sha256:...","parent_journal_hash":null,"content_hash":"sha256:..."}
```

Each later line serializes one public `SimulationOutputView`, the source batch hash,
all record payloads, payload hashes, record hashes, and the view hash. Implement strict duplicate-key and non-finite-number rejection. Reconstruct the exact typed payload class from the record kind and validate every supplied hash against recomputation.

- [ ] **Step 4: Implement atomic append-by-replacement and replay**

`write_public_simulation_journal` replays an existing file before append, checks stream/scenario/state/sequence continuity, creates a public capability view, writes the complete header and views as canonical compact JSON with one trailing newline to a same-directory temporary file, flushes and fsyncs, then calls `os.replace`. Clean the stage on failure and preserve the old journal.

`replay_public_simulation_journal` reads with a bounded per-line and total size policy, verifies the header and every typed view, and returns one immutable `SimulationPublicJournal` value. It never imports providers, Blender, HTTP, WSS, X6, PixiJS, Monaco, Protobuf, or gRPC.

- [ ] **Step 5: Export APIs and document the end-to-end V21.2 flow**

Export all public enums/contracts/projector/filter/journal/replay APIs from `narrative_dynamics.abm`. Add a README V21.2 section showing:

```python
batch = project_simulation_output(
    scenario,
    round_result,
    stream_id="law-firm-run",
)
public_view = filter_simulation_output(
    batch,
    SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
)
journal = write_public_simulation_journal("law-firm.jsonl", batch)
assert replay_public_simulation_journal("law-firm.jsonl") == journal
```

State explicitly that JSON-RPC/H2/WSS, the Web editor, the synchronous bus, coordinator, commands, live Blender, and remote Agents remain later phases.

- [ ] **Step 6: Run Task 4 GREEN and authorized final gates**

Run exactly:

```text
python -m pytest tests/test_network_abm_simulation_output_contracts.py tests/test_network_abm_simulation_output.py tests/test_network_abm_simulation_output_journal.py tests/test_network_abm_situated_network_contracts.py tests/test_network_abm_situated_network.py tests/test_network_abm_public_api.py -q
python -m unittest discover -s tests -p "test_network_abm*.py" -q
python -m compileall -q narrative_dynamics tests
git diff --check
```

Expected: all executed tests pass; only existing environment skips remain; compileall and diff check exit zero. Do not run the stopped repository-wide suite.

- [ ] **Step 7: Commit Task 4**

```text
git add narrative_dynamics/abm/simulation_output_journal.py tests/test_network_abm_simulation_output_journal.py narrative_dynamics/abm/__init__.py tests/test_network_abm_public_api.py README.md
git commit -m "feat(abm): journal public simulation outputs"
```
