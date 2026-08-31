# Endogenous Network Rewiring V4 Design

## Objective

Extend the fixed-roster network ABM with a persistent relationship state that changes endogenously as agents' beliefs change. Propagation changes beliefs in the current round; belief similarity then activates or deactivates catalogued relationship channels for the following round, creating a deterministic cognition–structure feedback loop.

V4 changes topology state, not the identity of possible relationships. Every potential directed edge must already exist in the base catalog, so formation and dissolution remain bounded, inspectable, and content-addressed. Runtime composition with adaptive trust and population lifecycle remains deferred to a later unified engine.

## Rewiring rule and timing

For each catalogued edge from source `i` to target `j`, post-propagation similarity is:

```text
similarity(i, j) = 1 - abs(next_belief(i) - next_belief(j))
```

The model has a dissolution threshold `D` and formation threshold `F`, with `0 <= D < F <= 1`:

```text
if similarity >= F: next_active = true
elif similarity <= D: next_active = false
else: next_active = prior_active
```

The gap between `D` and `F` is hysteresis: intermediate similarity preserves the relationship rather than causing threshold chatter.

Each round executes in this order:

1. Build an effective V1 model from the prior edge-active snapshot.
2. Propagate information synchronously over those prior active edges.
3. Compute similarity from the resulting agent beliefs.
4. Apply the rewiring rule to every catalogued edge.
5. Store the new topology for round `t + 1`.

Thus an edge formed or dissolved in round `t` never changes propagation in that same round. It affects only later rounds.

## Contracts

`EndogenousRewiringModel` binds an exact `NetworkABMModel`, a dissolution threshold, and a formation threshold. The base model must contain at least one catalogued edge; each edge's initial `active` flag is the round-zero topology.

`EdgeTopologyState` identifies one exact directed edge and stores current activity, last observed similarity, and cumulative rewiring count.

`RewiringPopulationState` stores the exact complete agent roster and one topology state per catalogued edge. Agent beliefs, exposures, and broadcasting are the existing V1 `NetworkAgentState` contracts.

`EdgeRewiringUpdate` records prior activity, post-propagation similarity, next activity, and whether the edge changed in that round.

`RewiringRoundResult` binds the prior state, transmissions, all canonical edge updates, and the next state. `RewiringTrajectory` validates a positive multi-round state chain.

All public artifacts are immutable, canonically ordered, serializable with `to_dict()`, and identified by the repository's stable SHA-256 hashing.

## Structural emergence metrics

`NetworkStructureMetrics` reports:

- catalog edge count;
- active edge count and active-edge rate;
- mean similarity across active edges, defined as `0` when no edge is active;
- rewired-edge rate, the fraction of catalogued edges changed at least once;
- cumulative rewiring count;
- weakly connected component count over the current active directed graph, treating direction as irrelevant only for component membership;
- largest weak component share of the fixed agent roster.

Isolated agents each form one weak component. These metrics expose density, assortativity, structural churn, and fragmentation without adding external graph dependencies.

## Acceptance scenarios

1. Two dissimilar broadcasters use an initially active edge in round one; their post-round similarity dissolves it only for round two.
2. An initially inactive candidate edge whose endpoints become sufficiently similar forms after propagation and transmits only in the following round.
3. Similarity strictly between dissolution and formation thresholds preserves prior activity.
4. Reversing catalog agent and edge input order preserves exact state, trajectory, update ordering, and hashes.
5. Every state covers the exact base roster and exact candidate-edge set; missing, duplicate, or unknown topology identities are rejected.
6. An edge rewiring count increments only when activity changes and remains persistent across later rounds.
7. A topology with no active edges still advances beliefs and topology deterministically with zero transmissions.
8. Structural metrics match hand-computed connected and fragmented graphs.

## Deferred work

- adding previously uncatalogued edge identities;
- capacity-constrained partner selection and competition;
- directed similarity rules using role or trust asymmetry;
- joint execution with adaptive trust and population lifecycle;
- stochastic contacts and learned rewiring thresholds;
- network-level intervention comparison for rewiring policies.
