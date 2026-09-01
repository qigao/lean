# Situated Network Runtime V19 Design

## Goal

Provide one deterministic, content-addressed runtime view over the existing V15.1 percept-memory-cognition loop and V14 directed social-memory graph. A V19 state also binds the canonical logical V15.1 SQLite memory checkpoint that the next transition must consume. A V19 snapshot must show where every embodied agent is, what common tracked hypothesis it privately assigns probability to, which directed relationships currently carry trust, which physical perception paths currently exist, which source-labelled TELL events the sanitized percept layer disclosed to which observers, and the resulting population-level metrics.

## Boundary

V19 is an orchestration and projection layer. V10 remains authoritative for physical events, V15 for sanitized percepts, V11/V13 for private beliefs and decisions, and V14 for claims and directed trust. V19 advances those existing runtimes through their public V15.1 entry point, then derives a multiplex network snapshot without modifying any earlier contract or hash.

This foundation does not pretend that V1-V9 and V10-V18 already share one mutable population state. Embodied roster entry/exit/death and V1-V9 candidate-edge rewiring require a later V19.1 migration because V10-V15 currently bind exact fixed rosters into world, cognition, memory, and social hashes. V19 must not represent a merely analytical active flag as physical lifecycle semantics.

## Runtime model

`SituatedNetworkRuntimeModel` binds:

- one exact `SituatedPerceptMemoryCognitiveModel`;
- one exact `SituatedSocialMemoryModel` over the same cognitive model;
- one `tracked_hypothesis_id` present in every cognitive agent hypothesis space;
- an `adoption_threshold` in `[0, 1]`;
- a `relationship_trust_threshold` in `[0, 1]` used only to classify directed relationship edges as active for metrics.

The model requires an exact perception/world/cognitive/social roster match. Thresholds never change private beliefs or V14 trust.

## Multiplex snapshot

Each immutable `SituatedNetworkSnapshot` binds the exact story, cognitive state, and social state hashes at one round and contains:

- `SituatedNetworkAgentNode`: agent ID, declared role, current physical place, tracked private belief probability, and active claim count;
- `SituatedNetworkRelationshipEdge`: information source to observer, current trust and affinity, confirmation/contradiction counts, and an active classification derived from the model threshold;
- `SituatedNetworkAccessEdge`: ordered source/observer pair with optional visual cost, optional auditory loss, and direct interaction reach derived from the current V15 graph and world state;
- `SituatedNetworkTransmission`: a latest-round successful TELL event's source, distinct non-source observer, sanitized fidelity, and disclosed channels, without message text or hidden event details. A transmission exists only when that observer's V15 percept itself explicitly contains `kind=TELL` and a non-null actor distinct from the observer. The actor's mandatory self percept is not a network transmission, and an anonymous `DETECTED` sound remains solely in V15.

Relationship direction is stored as `source_agent_id -> observer_agent_id`, matching information flow while retaining V14's meaning that trust belongs to the observer. Access edges are derived for every distinct ordered agent pair. Transmission records are derived only from sanitized V15 percepts and therefore cannot widen observer knowledge.

## Emergence metrics

`measure_situated_network_emergence` returns one immutable metric record with:

- population size and occupied-place count;
- adopted count/rate for the tracked hypothesis;
- population mean and variance of the tracked belief;
- active directed relationship edge count and mean trust;
- direct-interaction ordered pair count;
- latest-round TELL event count, transmission count, reached observer count, and exact/detected/identified counts;
- active, confirmed, contradicted, superseded, and forgotten claim counts.

All denominators and empty cases are deterministic. `latest_tell_event_count` may count successful objective latest-round TELL events as a payload-free analyst aggregate, but anonymous `DETECTED` sounds do not enter any V19 transmission, observer, or fidelity bucket. Metrics disclose aggregates only and never message text, memory text, or private evidence payloads.

## Atomic round and trajectory

`initialize_situated_network_runtime(database_path, model, story, cognitive_state, social_state)` validates the exact story/cognition/social checkpoint, initializes or validates the file-backed V15 percept-memory schema, binds its canonical logical store hash, and derives the initial snapshot and metrics. The canonical hash covers sorted metadata and validated logical percept-memory records, including activation state. It excludes the machine path, SQLite row IDs, derived FTS/index/trigger state, and raw database byte layout.

`simulate_situated_network_round(database_path, model, state)`:

1. validates the complete prior chain and requires the actual logical database hash to equal the state's bound memory-store hash;
2. clones the committed SQLite database to a temporary stage with the standard-library backup API;
3. calls `simulate_situated_percept_social_cognitive_round` exactly once against that stage;
4. derives and validates the next story, cognition, social state, snapshot, metrics, exact underlying branch extension, and staged memory-store hash;
5. verifies that the original database still has the prior bound hash;
6. publishes the validated staged database to the original through SQLite backup and returns the parent-linked result whose next state binds the post-round logical hash.

`simulate_situated_network_runtime(..., round_count=N)` repeats that atomic transition and returns a trajectory whose story, cognition, social state, snapshot, metrics, memory-store hashes, and parent hashes form one exact chain. Each next story must have the same initial state and perception model and contain the prior rounds as its exact prefix; each next cognitive and social state must name the corresponding prior subsystem content hash as parent. A trajectory rechecks this underlying branch rule rather than relying only on outer V19 hashes.

All V15.1 writes occur only on the stage until the complete V19 result validates. A transition or validation exception therefore leaves the original logical store unchanged, including a later-agent conflict after an earlier agent successfully ingested on the stage. SQLite's backup publication is transactionally observable: a failed backup does not expose a partial logical store. The remaining crash boundary is between the committed SQLite publication and any caller-managed durable persistence of the returned V19 state; a process loss in that interval can leave the old state stale against the advanced database, and recovery must bind a checkpoint from the matching subsystem state and store. V19 requires a file-backed database because its underlying V15 helpers open independent connections.

## Privacy and provenance invariants

- Snapshot constructors require exact model/story/cognitive/social hashes and round alignment.
- A node carries only one declared scalar belief projection, never another agent's full private state.
- A transmission carries no message or event detail and every disclosed field must come from one sanitized latest-round percept; objective events may not restore a hidden actor or kind.
- Every runtime state content hash includes its canonical logical percept-memory store hash, never its database path.
- Every round is the exact one-round extension of its prior story, cognitive state, social state, and memory checkpoint.
- Relationship and access edges cannot create cognitive evidence.
- V19 metrics cannot feed back into the simulation.
- Repeating projection over identical inputs is byte-equivalent and content-address stable.

## Public API and documentation

Contracts live in `narrative_dynamics/abm/situated_network_contracts.py`; projection, metrics, and orchestration live in `narrative_dynamics/abm/situated_network.py`; canonical logical V15 store hashing lives beside the existing SQLite schema helpers in `narrative_dynamics/abm/situated_percept_memory.py`. Public names are exported from `narrative_dynamics.abm`. README documentation uses a fully defined office construction to show that a closed-door anonymous detection is not a V19 transmission, an open-door exact TELL is, and one database-bound runtime round advances synchronously.

## Non-goals

- No physical population entry, exit, death, spawning, or body removal.
- No V1-V9 scalar-belief propagation injected into private cognition.
- No relationship rewiring or runtime intervention engine.
- No continuous coordinates, collision, path planning, or physical simulation.
- No LLM call, semantic grounding change, narrative projection change, or rendering change.
- No UI or distributed execution.
