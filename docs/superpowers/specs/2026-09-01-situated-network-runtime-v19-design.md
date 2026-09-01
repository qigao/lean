# Situated Network Runtime V19 Design

## Goal

Provide one deterministic, content-addressed runtime view over the existing V15.1 percept-memory-cognition loop and V14 directed social-memory graph. A V19 snapshot must show where every embodied agent is, what common tracked hypothesis it privately assigns probability to, which directed relationships currently carry trust, which physical perception paths currently exist, which TELL events actually reached which observers, and the resulting population-level metrics.

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
- `SituatedNetworkTransmission`: a latest-round successful TELL event's source, observer, sanitized fidelity, and disclosed channels, without message text or hidden event details.

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

All denominators and empty cases are deterministic. Metrics disclose aggregates only and never message text, memory text, or private evidence payloads.

## Atomic round and trajectory

`initialize_situated_network_runtime` validates the exact story/cognition/social checkpoint and derives its initial snapshot and metrics.

`simulate_situated_network_round(database_path, model, state)`:

1. validates the complete prior chain;
2. calls `simulate_situated_percept_social_cognitive_round` exactly once;
3. derives the next snapshot from the returned story, cognitive state, and social state;
4. measures next population metrics;
5. returns a parent-linked `SituatedNetworkRoundResult` and `SituatedNetworkRuntimeState`.

`simulate_situated_network_runtime(..., round_count=N)` repeats that atomic transition and returns a trajectory whose story, cognition, social state, snapshot, metrics, and parent hashes form one exact chain.

## Privacy and provenance invariants

- Snapshot constructors require exact model/story/cognitive/social hashes and round alignment.
- A node carries only one declared scalar belief projection, never another agent's full private state.
- A transmission carries no message or event detail and must correspond to one sanitized latest-round percept.
- Relationship and access edges cannot create cognitive evidence.
- V19 metrics cannot feed back into the simulation.
- Repeating projection over identical inputs is byte-equivalent and content-address stable.

## Public API and documentation

Contracts live in `narrative_dynamics/abm/situated_network_contracts.py`; projection, metrics, and orchestration live in `narrative_dynamics/abm/situated_network.py`. Public names are exported from `narrative_dynamics.abm`. README documentation uses the existing office fixture semantics to show a closed-door detected transmission versus an open-door exact transmission and a synchronized runtime round.

## Non-goals

- No physical population entry, exit, death, spawning, or body removal.
- No V1-V9 scalar-belief propagation injected into private cognition.
- No relationship rewiring or runtime intervention engine.
- No continuous coordinates, collision, path planning, or physical simulation.
- No LLM call, semantic grounding change, narrative projection change, or rendering change.
- No UI or distributed execution.
