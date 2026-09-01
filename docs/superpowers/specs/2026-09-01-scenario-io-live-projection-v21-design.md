# Scenario I/O and Live Projection V21 Design

## Goal

Make a complete situated simulation authorable as data and observable while it
runs. V21 introduces one versioned scenario package for physical, social, agent,
knowledge, story-plan, and run inputs; compiles that package into one exact
content-addressed runtime boundary; and projects every accepted round into typed,
audience-scoped output batches. Text, analysis, journals, and Blender are consumers
of those batches, never competing sources of truth.

The central invariant is:

```text
authoring tree -> validated graphs + initial state -> exact state transitions
                                                       |
                                                       v
                                      scoped output batches + causal history
                                                       |
                                  +--------------------+------------------+
                                  v                    v                  v
                               analysis              text              Blender
```

Input and output are one design. They share stable IDs, schema versions, source
hashes, audience rules, and the same compiled scenario hash. A consumer must never
infer a second identity system from labels, filenames, Blender object names, or
natural-language descriptions.

## Product boundary

V21 owns:

- a path-safe, standard-library JSON scenario-package loader;
- a normalized, immutable `CompiledSituatedScenario`;
- physical and social graph references, agent-local configuration, story-plan
  constraints, knowledge/asset catalogs, initial state, and run policy;
- deterministic per-round output projection;
- audience filtering and append-only public journaling;
- a one-way live Blender mirror with explicit gap recovery;
- validated inbound intervention requests as a separate command boundary;
- LLM authoring and agent proposals only as replayable, untrusted artifacts.

V21 does not turn Blender, an LLM, a JSON file, a UI, or an output subscriber into
the world-transition authority. V10-V20 contracts remain the authority for physical
events, perception, cognition, memory, social revision, network state, narrative
projection/realization, and final `.blend` export.

## Data topology

No single topology fits all data. V21 uses each topology only where its semantics
require it:

| Data | Topology | Reason |
|---|---|---|
| Package/files | rooted tree | ownership, versioning, replacement, access control |
| Places/passages | directed graph | movement and physical reachability |
| Relationships/access | directed multiplex graph | asymmetric trust, authority, perception, propagation |
| Institutions | containment tree plus cross-links | organizational ownership with cross-organization relations |
| Agent profile | agent-owned tree | body, goals, policies, knowledge grants |
| Story outline | act/scene tree plus dependency DAG | authored grouping plus causal/precondition order |
| Knowledge/assets | content-addressed catalog graph | many-to-many concepts, sources, permissions, representations |
| Runtime state | exact parent-linked chain | replay and atomic progression |
| Events | causal DAG | explanation and information provenance |

Trees express ownership. Graphs express relationships. State chains express time.
Files may be nested for humans, but cross-file references always use stable IDs and
are resolved before execution.

## Scenario package layout

The canonical V21 authoring layout is:

```text
scenario/
  scenario.json
  physical/
    world.json
    perception.json
    initial-state.json
    map.tmj
  social/
    institutions.json
    relationships.json
    norms.json
  agents/
    alice.json
    bob.json
  story/
    outline.json
    interventions.json
  knowledge/
    catalog.json
    access.json
  assets/
    catalog.json
  run.json
```

JSON is the V21 interchange format because the core has no YAML dependency and the
existing canonical hashing boundary already accepts JSON-like values. A later YAML
adapter may compile to exactly the same source documents; YAML syntax and comments
never enter canonical identity. Tiled orthogonal JSON remains the physical-map
adapter. The core never parses `.blend`; a Blender authoring integration must export
the same spatial-map document or Tiled-compatible geometry.

`scenario.json` contains only package identity and logical document locators:

```json
{
  "schema": "narrative-dynamics.scenario-package/v1",
  "scenario_id": "law-firm-case",
  "version": "1",
  "documents": [
    {
      "role": "physical.world",
      "path": "physical/world.json",
      "sha256": "sha256:..."
    },
    {
      "role": "physical.map",
      "path": "physical/map.tmj",
      "sha256": "sha256:..."
    },
    {
      "role": "social.relationships",
      "path": "social/relationships.json",
      "sha256": "sha256:..."
    }
  ]
}
```

The loader rejects absolute paths, `..`, symlink escapes, duplicate logical roles,
duplicate paths, missing documents, unexpected hashes, invalid UTF-8, duplicate JSON
keys, non-finite numbers, and documents larger than their declared role limit. It
never follows a network URI. Filesystem roots and relative filenames are operational
locators; compiled identity contains logical roles, schemas, canonical contents, and
hashes, not machine-local paths.

## Input model

### Source and compiled boundaries

`ScenarioPackageSource` is the immutable result of safely loading source documents.
It preserves the package ID/version, canonical document values, document identities,
and package hash. It contains no open file handles and no source root path.

`CompiledSituatedScenario` is the only value accepted by the V21 coordinator. It
binds:

- exact `SituatedNetworkRuntimeModel`;
- exact `SituatedSpatialMap`;
- exact initial `SituatedStory`, cognitive state, and social state;
- `ScenarioSocialWorld` institution, typed-relation, and norm definitions;
- `ScenarioStoryPlan`;
- `ScenarioKnowledgeCatalog` and per-agent grants;
- `ScenarioAssetCatalog`;
- `ScenarioRunPolicy`;
- every logical source-document hash and the complete package hash.

The compiler reconstructs existing runtime types through their public constructors.
It never deserializes Python class names, executes imported factories, evaluates
expressions, or trusts a serialized content hash. Every content hash is recomputed.

### Physical input

Physical documents define:

- V10 place, passage, object, and embodied-agent specifications;
- V15 visual, auditory, and interaction edges plus signal/profile thresholds;
- initial agent locations, passage states, object locations/holders;
- authored Tiled spatial geometry or deterministic auto-layout policy.

The logical place/passage graph is authoritative for legal movement. Geometry is a
bound projection used for layout and presentation. The compiler requires exact ID
coverage between the world graph, initial state, perception graph, spatial map, and
Blender-facing IDs.

### Social input

`ScenarioSocialWorld` separates social definitions from mutable social state:

- institutions and containment parents;
- memberships and roles;
- directed typed relationships such as `reports_to`, `trusts`, `represents`,
  `depends_on`, and `conflicts_with`;
- numeric trust/affinity initial values where the existing V14 runtime supports
  them;
- norms containing typed subject-role, action, target-scope, effect, and priority;
- grants that map roles or memberships to existing action/knowledge permissions.

Institution containment must be acyclic. Relationship edges are directed and unique
by `(source, target, type)`. A declared norm is accepted for runtime enforcement only
when its effect compiles to a registered deterministic validator. Unknown free-form
norm text may be retained as authoring metadata, but it is explicitly marked
`descriptive` and cannot affect an action, belief, metric, or output claim.

### Agent input

Each agent document owns body configuration, cognitive hypotheses, observation
symbols/likelihoods, available actions, goals/rewards, memory and recall policy,
social topics, role/membership references, and knowledge grants. Relationships are
not duplicated inside agent documents. An agent sees another agent through physical,
social, and knowledge projections, never because the other profile was nested in its
file.

Agent IDs must exactly cover the world, perception, cognition, memory, social, and
relationship rosters required by the compiled V19 runtime.

### Story-plan input

`ScenarioStoryPlan` combines an author-friendly tree with a dependency DAG:

- `ScenarioExecutionMode`: `authored`, `hybrid`, or `sandbox`;
- ordered acts containing ordered scene IDs;
- scene contracts with place/participant scopes, hard preconditions, allowed
  exogenous interventions, exit predicates, maximum rounds, and desired outcomes;
- dependency edges between scenes;
- immutable fact/continuity constraints;
- optional terminal conditions.

Hard preconditions and exit predicates use a finite registered predicate vocabulary;
they are not Python or LLM expressions. Scene dependencies must form a DAG. In
`authored` mode every round must belong to one active scene contract. In `hybrid`
mode agents remain autonomous while the coordinator may submit only declared typed
interventions. In `sandbox` mode the plan supplies initial conditions and terminal
limits but does not steer actions.

Failure to reach a desired outcome does not authorize fabricated events. A scene may
complete, time out, or become `stalled`; all three are outputs.

### Knowledge and asset input

Knowledge and asset catalogs contain metadata, not arbitrary embedded bytes:

- stable resource ID, kind, content hash, URI, media type, language, version/date;
- authority/provenance classification;
- concept/tags and optional index identity;
- access grants by agent, role, institution, or `public`;
- for 2D: dimensions, OCR/index hashes, preview hash;
- for 3D: dimensions/units, format, collection/asset ID, rig/collision metadata,
  multiview-preview hash, license metadata.

URIs are not fetched during package compilation. Retrieval integrations resolve an
allowed resource later and return evidence IDs/hashes. Agent memory, external
knowledge, objective world facts, and presentation assets remain distinct catalogs.
A lawyer's professional response may retrieve a statute, case file, and firm policy;
none becomes objective truth merely because retrieval or an LLM cited it.

### Run policy

`ScenarioRunPolicy` declares positive maximum rounds, checkpoint interval, execution
mode binding, optional deterministic seed, output-channel allowlist, public-journal
policy, Blender mode (`none`, `final_blend`, `live_mirror`), and resource limits.
Machine-local database paths, Blender executable paths, API keys, socket addresses,
and capability tokens are launch configuration, not scenario identity.

## Compilation and validation

Compilation is fail-closed and staged:

1. load and hash every declared source document;
2. validate document schemas and size limits;
3. build global ID catalogs and reject duplicates;
4. resolve cross-document references;
5. validate physical, perception, social, story, knowledge, and asset graphs;
6. construct exact existing V10-V20 contracts;
7. reconstruct and validate the complete initial subsystem state;
8. derive the authoritative V19 snapshot/metrics and compare them;
9. construct `CompiledSituatedScenario` and its content hash;
10. emit a deterministic `ScenarioCompilationReport` containing warnings and
   errors by logical document role and JSON pointer, never by secret value.

No SQLite database, Blender process, provider call, network access, or output
publication occurs before compilation succeeds.

## Runtime commands

Static package input and live commands are separate authority boundaries. The only
post-start inbound value is `SimulationCommandRequest`:

- command ID and exact expected current-state hash;
- typed command kind;
- declared actor/authority;
- canonical parameters containing registered IDs only;
- optional source artifact hash;
- capability scope.

Commands include pause/resume/step as control-plane operations and typed world
interventions explicitly admitted by the active story plan. A Blender interaction,
UI edit, or LLM director proposal becomes a request; it never mutates state directly.
Accepted/rejected command results are output records.

## Output model

### Authority and audience

Every output record has both a kind and audience:

- `public`: safe for UI, Blender, and public journals;
- `objective`: analyst truth ledger; may contain objective metadata but not raw
  private memory;
- `agent`: visible only to one named agent;
- `analyst`: privileged diagnostics explicitly requested by the host;
- `internal`: never published outside the coordinator.

Audience is enforced before callback, serialization, journaling, or transport. It is
not a label added after a subscriber receives a larger object.

### Typed records

`SimulationOutputRecord` wraps one typed payload and binds stream ID, scenario hash,
sequence, round index, state hash, kind, audience, optional owner agent ID, source
artifact hashes, and payload hash. V21 record kinds are:

- `state.delta`: public physical changes and content hashes;
- `event.objective`: payload-safe objective event metadata;
- `percept.private`: one existing sanitized percept, owned by one agent;
- `agent.decision`: selected structured action and public/private explanation view;
- `memory.update`: IDs/hashes and counts, with content only in the owning audience;
- `social.update`: relationship and claim deltas scoped to their entitled viewers;
- `network.metrics`: V19 aggregate metrics;
- `story.progress`: active/completed/stalled scene state;
- `narrative.scene`: an accepted V17/V18 projection or realization reference;
- `blender.delta`: public map/object/agent transform and timeline changes only;
- `command.result`: accepted/rejected command metadata;
- `diagnostic`: bounded redacted operational information.

Records never carry arbitrary untyped extension mappings in V21. New payload shapes
require a schema/version and a new record kind or version.

### Atomic round batches

`SimulationOutputBatch` binds one accepted V19 round result:

- exact prior and next state hashes;
- monotonically increasing first/last sequence;
- canonical ordered records;
- round result hash and batch hash;
- checkpoint flag.

No subscriber observes half a round. The coordinator first validates and publishes
the new runtime state, then derives the complete immutable batch, then delivers it.
A delivery failure cannot roll back or alter the accepted simulation state. Delivery
status is operational state outside canonical world identity.

Canonical record order is command results, objective events, state delta, private
percepts, decisions, memory, social changes, metrics, story progress, narrative
references, Blender delta, diagnostics; identities break ties. Reprojection over the
same compiled scenario and trajectory is byte-equivalent.

## Output bus and journal

The dependency-free core begins with a synchronous `SimulationOutputBus`:

- subscribers declare allowed kinds and one audience capability;
- subscription tables are copied before delivery, so callbacks cannot mutate the
  current dispatch;
- reentrant publication is rejected;
- callback exceptions are captured as redacted delivery reports and do not stop
  delivery to other subscribers;
- callbacks cannot return commands; commands use the explicit command API;
- no background thread or unbounded queue exists in the core.

`write_public_simulation_journal` appends only `public` records as canonical JSONL.
It writes one batch to a same-filesystem stage, fsyncs, and replaces the journal so a
crash cannot expose half a batch. Private and analyst records are never persisted by
this convenience API. A journal header binds schema, stream ID, compiled scenario
hash, and parent journal hash. Replay verifies every record/batch hash and sequence
without rerunning agents or LLMs.

## LLM integration

LLMs connect at two explicit boundaries.

### Authoring

A natural-language request may produce `ScenarioDraftArtifact`, which contains
provider identity, prompt/schema hashes, raw-response hash, and candidate package
documents. It is untrusted input. Acceptance requires the exact same package compiler
as hand-authored JSON. Unknown IDs, missing physical/social coverage, invalid graph
cycles, unsupported norms, and invented resource references are rejected.

### Agent decisions and knowledge

For one round, one agent receives only:

- its sanitized current percepts;
- its private memory/claim views;
- its local directed relationship neighborhood and trust values;
- its role/institution/norm permissions;
- allowed knowledge retrieval hits with source hashes;
- its goals, action allowlist, and current physical affordances.

The provider returns a structured `AgentProposalArtifact`, not prose and not a next
state. A deterministic validator converts an accepted proposal into an existing
`SituatedActionIntent`; the central synchronous resolver commits all intents. Agent
provider calls may run in parallel from the same prior snapshot, but state resolution
remains one barrier and one atomic commit. Accepted artifacts make replay provider-
free. No agent receives a shared hidden chat transcript or another agent's private
context.

Narrative LLM realization remains downstream: it consumes entitled narrative scenes
and cannot feed facts back into simulation except through a separately validated
future action proposal.

## Blender live mirror

V20 final `.blend` export remains supported. Live mode adds a persistent integration:

1. build or open one scene whose objects carry stable `nd_*_id` properties;
2. start a localhost-only bridge with an ephemeral capability token outside hashes;
3. subscribe only to `blender.delta`, public event markers, and public diagnostics;
4. queue decoded deltas off Blender's main thread;
5. apply `bpy` changes from a Blender main-thread timer;
6. keyframe accepted round frames while also showing the latest live state;
7. detect sequence gaps and request a complete public snapshot before continuing.

`BlenderStateDelta` contains changed transforms, passage states, object attachment,
public scalar display values, markers, and source hashes. It contains no TELL text,
private percept, memory content, provider prompt, API key, database path, or analyst
truth unavailable to the public view.

The transport queue is bounded. Transform updates for the same entity may coalesce
before application; events, command results, and timeline markers may not be dropped.
If non-coalescible capacity is exhausted, the mirror disconnects and requires
snapshot recovery rather than silently diverging.

Blender-to-simulator interaction uses `SimulationCommandRequest` with the exact
expected state hash. The simulator may accept, reject as stale, reject as unauthorized,
or reject as physically impossible. Blender never edits the authoritative state.

## Replay and recovery

Three replay levels remain distinct:

- trajectory replay reconstructs authoritative world/social/mind history;
- output reprojection reconstructs scoped batches from an accepted trajectory;
- journal/Blender replay reproduces a presentation stream without running agents.

A live consumer reconnects with `(stream_id, last_batch_hash, last_sequence)`. If the
coordinator retains the required public batches it resumes; otherwise it sends a full
public snapshot followed by later batches. Private replay requires a fresh matching
agent capability and is never available through the Blender bridge.

## Privacy and provenance invariants

- Package compilation never executes input or follows remote references.
- Source roots, database paths, executables, sockets, tokens, and API keys do not
  enter content hashes or public errors.
- Every accepted runtime value binds the compiled scenario hash.
- Output projection never reconstructs hidden fields from objective events.
- An agent output record is filtered before serialization and names exactly one owner.
- Blender receives public deltas only.
- Public journals contain no private memory, percept content, TELL payload, prompt, or
  provider response.
- LLM outputs are proposals or presentation artifacts, never state transitions.
- Output delivery cannot feed back into state except through a separately validated
  command request.
- Repeated compilation and reprojection are deterministic and path-independent.

## Public API direction

Core ABM APIs:

```python
load_situated_scenario_package(root) -> ScenarioPackageSource
compile_situated_scenario_package(source) -> CompiledSituatedScenario
initialize_compiled_scenario(database_path, scenario) -> SituatedNetworkRuntimeState
advance_compiled_scenario(database_path, scenario, state, commands=())
    -> ScenarioRoundCommit
project_simulation_output(scenario, round_commit) -> SimulationOutputBatch
replay_public_simulation_journal(path) -> tuple[SimulationOutputBatch, ...]
```

Output APIs:

```python
bus = SimulationOutputBus(stream_id, scenario.content_hash)
subscription = bus.subscribe(kinds, capability, callback)
report = bus.publish(batch)
bus.unsubscribe(subscription.subscription_id)
```

Optional integrations:

```python
BlenderLiveMirror(...)
ScenarioAuthoringProvider(...)
SituatedAgentProposalProvider(...)
KnowledgeRetrievalProvider(...)
AssetRetrievalProvider(...)
```

Integrations are injected and lazily imported; importing `narrative_dynamics.abm`
remains provider-, Blender-, network-, and SDK-neutral.

## Acceptance scenario

A law-firm package defines an office map, public lobby, meeting room, restricted
archive, door states, three lawyers, one client, a firm hierarchy, directed trust,
representation and reporting relationships, confidentiality norms, private case-file
grants, public statutes, a contract scan, and an archive-cabinet Blender asset. Its
hybrid story plan requires a missing-document discovery before a confrontation but
does not prescribe which agent accuses whom.

Compilation proves exact ID/roster/map coverage and rejects a draft granting the
client access to a private firm file. During execution Alice's agent proposal cites a
retrieved statute and private case memory, then becomes a validated `TELL` or `WAIT`
intent. A closed door leaves another agent with a sanitized detected sound only. One
atomic output batch contains public movement/door/event/metric deltas and separate
agent-private percept/decision/memory records. A public journal and live Blender
mirror show the same map and movement without the private message. Blender requests
opening the archive door against an old state hash and receives a stale rejection.
Replay reproduces the trajectory, output hashes, text projection, and Blender public
timeline without new provider calls.

## Delivery phases

V21 is divided into independently useful, reviewable releases:

1. **V21.1 Scenario package and compiler:** safe JSON package, canonical source
   identity, physical/social/agent/story/catalog validation, compiled scenario.
2. **V21.2 Output contracts and projector:** typed scoped records, atomic batches,
   deterministic projection, public JSONL journal and replay.
3. **V21.3 Scenario coordinator:** scene progress, typed commands, synchronous bus,
   round commits, capability filtering.
4. **V21.4 Blender live mirror:** bounded localhost transport, snapshot/gap recovery,
   main-thread scene application, validated inbound intervention requests.
5. **V21.5 LLM and retrieval adapters:** scenario drafts, per-agent structured
   proposals, text/2D/3D catalog retrieval, accepted-artifact replay.

Each phase preserves the same IDs and compiled-scenario/output schemas. No phase may
introduce a parallel ad-hoc scene or event format.

## Non-goals

- No arbitrary Python/YAML execution, imported factory, or plugin declared by input.
- No automatic download of models, images, documents, maps, or provider SDKs.
- No claim that Wikipedia or an LLM is authoritative professional evidence.
- No unrestricted predicates, entities, norms, or actions invented at runtime.
- No distributed consensus, exactly-once network delivery, or multi-writer state.
- No continuous physics, collision authority, navmesh authority, or acoustic solver
  delegated to Blender in V21.
- No Blender mutation of canonical state and no bidirectional `.blend` round trip.
- No private stream persistence in the public journal.
- No hidden chain-of-thought as Agent state or output.
- No automatic rendering, publication, or external communication.
