# Small-World Finite Metrics V23.2 Design

## Objective

Give the deterministic V23.1 certificate exact finite-network measurements. The
new Lean layer defines local clustering coefficient, shortest hop count, graph
diameter, and average shortest-path length, then proves that the qualitative
certificate entails the expected numerical bounds.

## Representation

`NarrativeDynamics.Core.SmallWorldMetrics` reuses `MeshGraph`, `MeshWalk`,
`ReachWithin`, `GlobalHopBound`, `PerfectLocalClustering`, and
`SmallWorldCertificate`. It introduces no second path semantics.

Finite enumeration uses `Fintype Node` and `DecidableEq Node`. Local adjacency
counting additionally requires `DecidableRel g`. Metrics use ordered node pairs.
For a symmetric graph, counting both orientations multiplies numerator and
denominator equally and therefore leaves the clustering ratio unchanged.

## Local clustering

For a center node:

- `neighborSet` contains every outgoing neighbor;
- `orderedNeighborPairs` contains distinct ordered pairs of its neighbors;
- `closedNeighborPairs` retains pairs joined by an edge;
- `localClusteringCoefficient` is the rational closed-pair count divided by the
  rational candidate-pair count.

Division by zero follows Lean's rational-field convention and yields zero. Thus a
node with fewer than two distinct outgoing neighbors has coefficient zero.

Required proofs establish that closed pairs are a subset of candidate pairs, that
perfect local clustering makes both sets equal, and that a nonempty candidate set
therefore gives coefficient one.

## Shortest paths and diameter

`shortestHopCount` uses `Nat.find` over the existing `ReachWithin` predicate. It is
defined only with proof that some `GlobalHopBound` exists. The accompanying
theorems prove that the chosen hop count is reachable and is no greater than any
other witnessed hop budget.

`meshDiameter` is the least natural number satisfying `GlobalHopBound`. Its
specification and minimality theorems make the definition usable without exposing
the `Nat.find` implementation. A `SmallWorldCertificate g limit` proves that this
diameter is at most `limit`.

## Average shortest path

`orderedDistinctNodePairs` enumerates every ordered pair of distinct nodes.
`averageShortestPathLength` sums exact shortest hop counts over that set and divides
by its cardinality as a rational. Empty and singleton node types have no distinct
pairs and therefore average zero.

For a finite certified graph, every shortest hop count is at most the certificate
limit. Summing those pointwise inequalities and dividing by the positive pair count
proves `averageShortestPathLength ≤ limit`; the empty-pair case is discharged
separately.

## Files

- Create `NarrativeDynamics/Core/SmallWorldMetrics.lean`.
- Create `NarrativeDynamics/Tests/SmallWorldMetrics.lean`.
- Modify `NarrativeDynamics.lean`, `.github/workflows/proof.yml`, and `README.md`.

## Non-goals

- No stochastic Watts--Strogatz generator or expectation theorem.
- No alternate attachment-model process or power-law asymptotics.
- No empirical estimator, confidence interval, or calibration claim.
- No Python/NetworkX implementation.
- No mutation of runtime Agent or world state.

## Acceptance

Lean theorem-use tests must demonstrate coefficient one under perfect clustering,
shortest-path minimality, certificate-bounded diameter, zero average for a
single-node type, and certificate-bounded average shortest path. The root build and
both preceding V23 theorem suites must remain green, with no `sorry` or `admit`.
