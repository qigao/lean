# Mesh Feasibility Lean V23 Design

## Status

Approved direction: Lean mathematical proof precedes Python, NetworkX, P2P,
WebSocket, or Raft implementation.

## Goal

Establish a minimal formal foundation showing that independently evolving small
societies can be embedded into a larger social mesh without losing their local
paths, that explicitly closed societies cannot leak paths to outsiders, and that
declared bridge edges compose local paths into bounded global paths, including a
six-degree corollary.

This is a feasibility result for the mesh representation. It does not claim that
an empirical society is necessarily a Watts--Strogatz or Barabasi--Albert graph.

## Mathematical boundary

For a node type `Node`, a mesh snapshot is a directed relation
`MeshGraph Node := Node -> Node -> Prop`. Time is handled by selecting one graph
snapshot at a time; temporal transitions and stochastic generators remain later
proof layers.

`MeshWalk g n a b` witnesses exactly `n` directed steps from `a` to `b`.
`ReachWithin g k a b` witnesses some walk of length at most `k`.

A local society is a predicate `inside : Node -> Prop`. Its graph projection
retains only edges whose source and target both satisfy `inside`. A society is
closed when every edge leaving an inside node also ends inside.

Agent placement is represented separately from social membership:

- `physicalWorld : Agent -> World` is a total function and therefore gives one
  authoritative physical world per Agent;
- `socialMember : Agent -> Society -> Prop` is a relation and therefore permits
  simultaneous membership in multiple societies.

## Required theorems

1. `MeshWalk.refl`, `MeshWalk.single`, and `MeshWalk.append` establish finite
   path construction and exact length addition.
2. `reachWithin_mono` establishes monotonicity in the hop budget.
3. `restricted_walk_lifts` establishes that every path in a local projection is
   also a path in the global mesh.
4. `closed_walk_restricts` establishes that a global path starting inside a
   closed society can be reconstructed entirely inside its local projection.
5. `closed_no_cross_reach` establishes noninterference: a node outside a closed
   society is unreachable from an inside node.
6. `reachWithin_bridge` establishes that local bounded paths plus one bridge
   yield a global path bounded by the sum of both local bounds plus one.
7. `six_degrees_of_two_three_bridge` establishes the concrete corollary
   `2 + 1 + 3 <= 6`.
8. `physical_world_unique` and `multiple_social_memberships_compatible`
   establish that overlapping social membership does not weaken unique physical
   residence.

## Compatibility

The new module is additive. Existing `InfoGraph`, `TypedGraph`, `WorldGraph`,
and all V1--V22 runtime semantics remain unchanged. A later implementation may
project a typed temporal mesh snapshot into the existing graph relations.

## Non-goals

- no claim that every pair of Agents is within six hops;
- no average clustering coefficient or average path-length formalization;
- no probabilistic Watts--Strogatz rewiring theorem;
- no preferential-attachment or power-law asymptotic theorem;
- no Python runtime, NetworkX generator, transport, persistence, P2P, WSS, or
  Raft implementation;
- no automatic community detection or institutional authority semantics.

## Acceptance

- `NarrativeDynamics/Core/SocialMesh.lean` contains no `sorry` or `admit`;
- `NarrativeDynamics/Tests/SocialMesh.lean` elaborates concrete uses of every
  required theorem;
- the root Lean library imports the module;
- the proof workflow runs the new test file;
- focused Lean builds and the new theorem test succeed on Lean 4.32.0 and
  mathlib v4.32.0.
