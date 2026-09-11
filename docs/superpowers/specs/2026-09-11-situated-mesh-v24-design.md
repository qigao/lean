# Situated Mesh V24 Design

## Status and baseline

The projection-first direction was reviewed in the conversation and the user asked to continue on 2026-09-11, explicitly permitting verification through GitHub CI.

Build additively on PR #58 head `c268e5355b14fb408f30814ed1c2bee2f41f1144`, using a separate branch and stacked PR. Do not merge or modify #58 as part of this change. Its historical proof workflow is red; a new attempt was requested to distinguish baseline failures from V24 failures. A successful local-build statement in the older PR is not a replacement for current CI evidence.

## Goal

Make the existing situated network available as an immutable, typed society mesh, with deterministic bounded-path witnesses. Preserve the authoritative scenario, story, cognition, social-memory, physical-placement, output, and checkpoint semantics.

This phase is a topology projection, not a distributed runtime or a new information-delivery mechanism.

## Ownership and compatibility

`MeshSnapshot` retains the exact immutable `SituatedNetworkSnapshot` supplied by the caller. It adds only society membership and directed bridge declarations. Nodes, places, relationship attributes, access attributes, and transmission records are not independently mutable mesh state.

The existing network model already binds the compiled world. V24 introduces no new world identifier or second placement authority. Every source node continues to have one `agent_id` and one `place_id`; multiple memberships never duplicate a node, mind, memory, or execution history.

No existing runtime consumer is switched automatically. No existing dataclass, serialization format, hash definition, package-root export, coordinator, or storage component is changed. Existing snapshots and outputs remain byte-for-byte unchanged when mesh queries are used or not used.

The first slice is confined to one existing compiled-world snapshot. Agents may have zero, one, or multiple memberships. Empty societies are allowed. Unassigned agents remain known nodes but have no non-reflexive edges in the effective society mesh.

## Selected relations

A query must explicitly choose one `MeshRelation`:

- `RELATIONSHIP`: directed V19 relationship edges whose `active` flag is true.
- `DIRECT_INTERACTION`: directed V19 access edges whose `direct_interaction` flag is true.
- `TRANSMISSION`: directed endpoint pairs in actual V19 transmission records from the selected snapshot; repeated records for a pair yield one topological edge.

Do not union these relations implicitly. In particular, relationship reachability does not imply physical access, perception, successful delivery, or belief admission. A path in a transmission snapshot is a static topological walk, not evidence of a causally ordered multi-event transmission history.

## Membership and source-backed bridges

`MeshSociety(society_id, member_agent_ids)` uses canonical immutable tuples. Society IDs and membership entries must be unique within their respective collections. All members must occur in the source node roster.

`MeshBridge(relation, source_society_id, target_society_id, source_agent_id, target_agent_id)` declares one directed cross-society query connection. Both societies must exist and be distinct; endpoints must be distinct and belong to the named source and target societies. The directed endpoint pair must already exist in the selected source relation. Unknown nodes, unknown societies, unsupported relation strings, duplicate declarations, and unbacked bridges are rejected.

Bridges do not invent source relationships, accesses, or transmissions. Hypothetical shortcuts and remote channels remain out of scope.

For source relation R, membership M, and selected declarations B:

```
E_r(a,b) = R_r(a,b) AND
           ((exists society s, M(a,s) AND M(b,s)) OR B_r(a,b))
```

Thus every effective mesh edge is source-backed. All source edges internal to a society are preserved. A single society containing every source node yields the exact source relation. Between disjoint societies, an existing source edge participates only when explicitly bridged. Overlapping societies may connect through shared members without a bridge; absence of a bridge is not a general isolation guarantee.

A local society projection retains only its member nodes and internal effective edges. Querying an outside node is invalid, including a zero-hop outside-to-itself query. This is stricter than merely restricting edges on the original Lean node type, whose reflexive walks exist for every node; the Lean local-reachability contract must include endpoint membership explicitly.

## Snapshot identity and witnesses

`MeshSnapshot.content_hash` uses the existing `stable_content_hash` convention over a schema tag, the complete source snapshot's content hash, sorted societies, and sorted bridge declarations. Source round and state identities are therefore transitively bound. There is no union across rounds and no mutable adjacency cache.

`MeshPath` binds the mesh content hash, selected relation, optional society scope, and a nonempty ordered tuple of agent IDs. Its hop count is the tuple length minus one. A witness can be checked against a supplied mesh snapshot; changed source state, round, membership, bridges, scope, invalid endpoints, or missing selected edges must invalidate it.

`find_mesh_path(snapshot, source, target, *, relation, max_hops, society_id=None)` returns a shortest witness within the inclusive budget or `None`. Known self-reachability has zero hops. Unknown endpoints/scopes are errors, not unreachable results. Hop budgets must be nonnegative integers and must reject booleans. Deterministic breadth-first search with lexically sorted neighbors resolves equal-length ties. Visited-node tracking bounds work even for a very large requested budget.

`mesh_path_is_valid` optionally checks an inclusive hop budget. This is an executable witness checker, not a claim of formal verification of Python.

`is_closed_society` means no effective outgoing edge of the selected relation leaves the society. Incoming edges are permitted. Neither this predicate nor a social-path theorem proves complete information isolation across other channels.

## Lean boundary

Add `NarrativeDynamics/Core/MeshProjection.lean` on the existing V23 graph foundation. Define the selected source-backed society projection and membership-qualified local reachability. Prove:

1. Every projected edge and bounded walk lifts to the selected source relation.
2. Covering every node with one society recovers the source relation exactly.
3. Adding declarations on a fixed source snapshot preserves existing projected bounded paths.
4. Only the selected source relation and selected bridge relation affect the projection.
5. Local membership-qualified reachability lifts to the global projection and source graph.
6. Without a selected source edge, neither membership nor a bridge can manufacture that edge.

Use the existing closed-society and bridge-composition theorems in concrete tests. Do not require `SmallWorldCertificate`: symmetry plus global reachability plus perfect local clustering is much stronger than ordinary high-clustering small-world structure. Do not assert unconditional six-hop reachability, monotone clustering, probabilistic Watts-Strogatz laws, or verified Python execution.

## Implementation surface

- `narrative_dynamics/abm/mesh_contracts.py`: immutable society, bridge, snapshot, relation, and path contracts.
- `narrative_dynamics/abm/situated_mesh.py`: source-backed projection and selected relation/scope views.
- `narrative_dynamics/abm/mesh_queries.py`: deterministic path search, witness checking, and outgoing-closure queries.
- `tests/test_mesh_projection.py`: executable contract, topology, identity, and compatibility tests.
- `NarrativeDynamics/Core/MeshProjection.lean` and `NarrativeDynamics/Tests/MeshProjection.lean`: abstract contract and concrete theorem tests.
- `NarrativeDynamics.lean` and `.github/workflows/proof.yml`: root import and CI integration only.

No new Python dependency, NetworkX, transport, persistence backend, P2P, WebSocket, Raft, random generator, or runtime scheduler is introduced.

## Acceptance and CI evidence

Use GitHub CI for RED and GREEN evidence. Add a separately named focused mesh Python job so unrelated repository-suite failures cannot obscure whether the mesh tests execute. Preserve the existing full Python suite, Lean build, conformance-vector checks, and all theorem tests; do not suppress or reclassify failures as success.

Record the exact tested commit and distinguish PR-head checks from merge-ref checks. Keep baseline #58 and V24 runs separate. Do not label V24 merge-ready while a required check fails or is unresolved.

Required executable cases:

- A seven-node directed chain split into three-node and four-node societies reaches its target in exactly `2 + 1 + 3 = 6` hops with an existing source-backed bridge; five hops fail, reverse traversal fails, and removing the declaration removes that route.
- A universal society preserves each selected source edge set exactly; queries never mutate source serialization or hash.
- Overlapping membership shares one source Agent and preserves its original place.
- The three relation selections cannot substitute for one another; inactive relationship and unavailable direct-interaction edges do not participate.
- Invalid and unbacked bridge declarations fail before a query runs.
- Closure is outgoing-only and relation-specific; local outside zero-hop queries are rejected.
- Unknown IDs, empty graphs, singleton/self paths, disconnected nodes, negative and boolean budgets, and deterministic equal-length tie-breaking have explicit behavior.
- A path is bound to source snapshot, round, relation, society scope, and mesh configuration; changed snapshots or tampered witnesses are rejected.
- Existing tests remain active. Any baseline failure is reported with its evidence rather than fixed opportunistically in this migration.

## Subsequent boundary

Mesh-driven action or message propagation requires a separate reviewed transition contract through the existing action/perception/admission pipeline. Distributed execution and physical-world migration are not consequences of the graph representation and remain separate work.
