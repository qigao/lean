# Unified Evolving Network V5 Design

## Objective

Combine population lifecycle, adaptive edge trust, and endogenous topology rewiring into one deterministic network ABM state and one atomic round transition. V1–V4 remain stable focused APIs; V5 is the integration surface for simulations in which membership, information, learned source reliability, and relationship structure co-evolve.

The fixed catalog still defines every possible agent identity, profile, and directed edge identity. The evolving state determines which agents participate, which candidate edges are active, how much each source is trusted, and what every agent currently believes.

## Unified round order

Round `t + 1` executes exactly:

1. Apply canonical `ENTER`, `EXIT`, and `DEATH` events to the prior member snapshot.
2. Select agents active after those events.
3. Select catalogued edges whose endpoints are both active and whose prior topology state is active.
4. Scale each selected edge's base influence by its prior trust.
5. Propagate once using the V1 synchronous kernel.
6. Apply same-round truth feedback to edges that actually transmitted; learned trust affects later rounds only.
7. Compute post-propagation belief similarity and rewire active-to-active candidate edges; topology changes affect later rounds only.
8. Recombine active propagated agents with inactive/dead members and persist all trust/topology state in one next-state hash.

This order prevents circular causality. Lifecycle changes determine current participation. Prior trust and topology determine current information flow. New trust and topology can influence only the next round.

## Inactive and dead endpoints

Edges incident to an inactive or dead agent cannot transmit and are not rewired. Their trust, activity, similarity, feedback count, and rewiring count remain unchanged. An inactive returning agent therefore resumes its retained relationships and learned source history. A dead agent cannot return, but retained edge state remains as an auditable historical record.

## Model and state contracts

`EvolvingNetworkModel` binds:

- one exact base `NetworkABMModel` catalog with at least one candidate edge;
- a canonical non-empty initial active-agent set;
- trust learning rate and initial trust in `[0, 1]`;
- dissolution and formation similarities satisfying `0 <= dissolution < formation <= 1`.

`EvolvingPopulationState` binds one exact model and parent chain and contains:

- one `LifecycleMemberState` for every catalogued agent;
- one `EdgeTrustState` for every candidate edge;
- one `EdgeTopologyState` for every candidate edge.

Round zero copies base edge activity, initializes common trust, computes endpoint similarity from initial beliefs, and records one entry for initially active agents. Inactive seeded agents retain their seeded belief but do not broadcast.

`EvolvingRoundResult` records lifecycle events, effective transmissions, truth feedback, trust updates, rewiring updates, and the exact next state. `EvolvingTrajectory` validates a positive chain driven by paired event and feedback schedules.

## Cross-layer metrics

`EvolvingSystemMetrics` reports one macro snapshot:

- catalog, active, inactive, and dead population counts;
- active and dead population shares;
- mean active belief and active adoption rate, both `None` when no agent is active;
- mean trust and learned-edge rate across all candidate edges;
- effective active-edge count and rate, where an edge is effective only when topology is active and both endpoints are active;
- cumulative entries, exits, and rewirings.

The effective-edge rate uses the full candidate-edge catalog as denominator, keeping snapshots comparable as membership changes.

## Acceptance scenarios

1. An inactive relay enters before propagation, receives information immediately through an active trusted edge, and contributes to post-round rewiring.
2. An exiting source does not transmit, cannot receive truth feedback, and its incident edge trust/topology state remains unchanged.
3. Prior trust scales current-round influence; same-round feedback changes trust only for the following round.
4. Prior topology controls current transmissions; post-round similarity formation/dissolution changes only the following round.
5. Feedback is accepted only for a candidate edge that actually transmitted after lifecycle and topology filtering.
6. A dead agent cannot re-enter, and zero active agents advance with no transmissions, feedback, trust updates, or rewiring updates.
7. Reversing lifecycle-event and feedback input order preserves exact state, result, trajectory, and hash identity.
8. Every state covers the exact catalog roster and exact trust/topology edge sets.
9. Multi-round schedules preserve one exact unified parent chain.
10. Cross-layer metrics match a hand-computed state containing membership change, learned trust, and rewired topology.

## Deferred work

- creation of identities or edge identities outside the catalog;
- automatic lifecycle policies and mortality hazards;
- trust decay while endpoints are inactive;
- capacity-constrained partner selection;
- multiple topics and topic-specific trust/topology;
- stochastic contacts and learned evolution parameters.
