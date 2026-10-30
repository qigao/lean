# R1.1: source-authoritative V19 social trust checkpoint admission

**Parent:** R1 [Draft PR #108](https://github.com/qigao/lean/pull/108)
and [Issue #107](https://github.com/qigao/lean/issues/107).

## Source authority and contract

V19's true relationship source is `SituatedSocialMemoryState.relationships`:
one directed record for every ordered Agent pair. Network
`SituatedNetworkSnapshot.relationship_edges[*].active` is derived from
`trust >= model.relationship_trust_threshold`, so editing a snapshot-only
Boolean cannot persist to a real simulation round.

This stage supplies an explicit **administrative checkpoint intervention**,
not a new graph owner or a substitute for normal event-driven trust learning.
`SocialTrustCheckpointIntervention` must bind exact V19 runtime, source
snapshot and social-state SHA-256 values, a directed source/observer pair,
a typed `ACTIVATE/DEACTIVATE` request and an explicit target trust in [0,1].
The operation must cross the *actual model threshold* in the requested
direction; it cannot fabricate reverse edges or transmission observations.

`admit_social_trust_checkpoint(database_path, model, checkpoint, event)`
checks the existing runtime's exact projection and SQLite memory-store hash,
then constructs a **new immutable social-memory checkpoint** with one
updated trust record. It reuses
`validate_situated_social_memory_state`,
`initialize_situated_network_runtime` and the original source projector
to produce the synchronized new runtime checkpoint and a source-/state-hash
receipt. Story/cognition round, affinity, relationship counters, claims,
processed evidence and unrelated access/transmission are unchanged.

The new state is an explicit *rebased checkpoint*: it can be passed into
`simulate_situated_network_round`, whose normal parent links then bind the
new social-memory state. No mid-round intervention is allowed. The actor and
reason are audit metadata, **not** authorization or an authenticated
human-world causal claim. External access control is not provided by R1.1.

## Acceptance

- Real model/story/cognition/social state projected by V19; one relation is
  deactivated at trust 0.0 (given threshold 0.5) and its reverse survives.
- Authoritative social-state hash, V19 source snapshot, metrics, V24 graph
  and selected path validity agree after the checkpoint; old snapshot and
  SQLite memory are not mutated.
- The real V19 round can advance from the rebased checkpoint with correct
  social-state parent and runtime identity. An ordinary noncheckpoint state
  rejects explicit intervention.
- Typed source/runtime/social stale hash, unknown node, no-op/non-crossing,
  mismatched action, bool/nan/out-of-range trust and invalid input are RED.
- A retired source-backed Bridge is rejected; unrelated direct-interaction
  and realized-transmission relations cannot drift.
- Keep all upstream Python/Lean/BB/PathN/Studio, exact-head CI and existing
  source-derived vectors unchanged. No new six-degree theorem.

**Remaining:** source-derived versioned R1.1 evidence vector and exact-head
full four-job CI. No claim of persistence to an external database or
authenticated administrative actors.

## Versioned real V19 checkpoint evidence

`tools/render_social_topology_checkpoint_vectors.py` regenerates a reviewed
schema-v1 JSON record from a genuine two-Agent V19
model/story/cognitive/social/SQLite checkpoint. It records the exact source,
social-memory and runtime identity at baseline, after deactivation, and after
reactivation. Each event binds its expected parent hashes; the V19 projector
and existing source-backed path/metric APIs determine observed edge changes.

The focused Python and independent Lean verification lanes compare these
canonical UTF-8 bytes against
`conformance/social_topology_checkpoint_v1.json`. The canonical file is checked in from real focused CI output and is mandatory.
Missing, stale or edited bytes are RED in both the focused Python and
independent Lean input-acceptance lanes. No hidden source graph or
external actor authentication is implied.

## Independent V19 source replay of checkpoint interventions

`SocialTrustCheckpointReceipt` rejects malformed hash fields, non-boolean
effective-edge flags, nonfinite thresholds and no-op transition identities.
`SocialTrustCheckpointResult` validates its source/runtime/social SHA chain
at construction. These *structural* checks do not imply the caller's target
trust was actually applied.

`verify_social_trust_checkpoint(database, model, original, recorded)`
re-executes the typed intervention from the supplied exact authoritative V19
checkpoint through the **same** `admit_social_trust_checkpoint` implementation,
then compares the complete new runtime state and receipt. Regression
deliberately forges a coherent receipt that binds a valid 0.25 trust update
while claiming that the original intervention set trust to 0.0: only actual
re-execution detects the mismatch. Canonical source-derived vectors are also
checked through this independent replay method without changing their bytes.

This remains deterministic in-process correctness evidence, not external
actor authentication, cryptographic proof of history or a second trust graph.
