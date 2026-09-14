# Watts--Strogatz Foundation V23.3 Design

## Objective

Construct the deterministic graph layer that precedes a stochastic
Watts--Strogatz model. V23.3 defines a finite regular ring lattice, connects it to
Mathlib's cycle graph, evaluates a concrete clustering coefficient, and proves
that undirected shortcut augmentation cannot increase shortest hop count,
diameter, or average shortest-path length.

## Ring lattice

For `nodeCount = n` and neighborhood radius `r`, nodes are `Fin n`. Two distinct
nodes are adjacent when either modular difference has value at most `r`:

```text
u ≠ v ∧ ((u - v).val ≤ r ∨ (v - u).val ≤ r)
```

This is a symmetric, loopless, decidable mesh. `WattsStrogatzParameters` requires
positive radius and `2 * radius < nodeCount`; this is the non-degenerate initial
lattice region used by later stochastic work.

The radius-one ring contains Mathlib's `SimpleGraph.cycleGraph`. A converter maps
Mathlib `SimpleGraph.Walk` values to the project's exact `MeshWalk`. Since every
reachable finite pair has a simple path shorter than the number of nodes, the
cycle graph and therefore every positive-radius ring lattice have global hop
bound `nodeCount - 1`.

## Concrete clustering witness

The existing finite metric is executable for a decidable ring. A theorem-use test
computes the six-node radius-two ring's local clustering coefficient at node zero
as `2 / 3`. This is a concrete checked value, not an asymptotic WS formula.

## Shortcut augmentation

`addUndirectedShortcut g left right` preserves every old edge and adds both
orientations of one pair. Given a loopless symmetric base and distinct endpoints,
the result remains loopless and symmetric.

General edge-monotonicity theorems prove:

- exact shortest hop count cannot increase;
- exact mesh diameter cannot increase;
- finite average shortest-path length cannot increase.

These results apply to shortcut augmentation immediately. No clustering
monotonicity is claimed: a shortcut can close an existing wedge, but it can also
create a new neighbor and therefore a new open wedge.

## Files

- Create `NarrativeDynamics/Core/WattsStrogatz.lean`.
- Create `NarrativeDynamics/Tests/WattsStrogatz.lean`.
- Modify `NarrativeDynamics.lean`, `.github/workflows/proof.yml`, and `README.md`.

## Non-goals

- No random-variable or probability-space definition.
- No independent per-edge rewiring sampler.
- No expectation or concentration theorem.
- No proof of the classic WS asymptotic regime.
- No alternate attachment-model preferential-attachment or power-law result.
- No Python or NetworkX runtime implementation.

## Acceptance

Lean tests must check ring symmetry, looplessness, the concrete `2 / 3`
coefficient, cycle containment, the ring global hop bound, and monotonicity of
shortest path, diameter, and finite average after adding a shortcut. All V23 Lean
tests and the root build must pass without `sorry` or `admit`.
