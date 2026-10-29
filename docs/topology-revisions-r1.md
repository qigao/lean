# R1: immutable directed topology revision / replay

[Issue #107](https://github.com/qigao/lean/issues/107). This is an
**executable runtime-model feature**, not another formal six-hop theorem.

## Ownership and observation

`SituatedNetworkSnapshot` remains the single authoritative source value.
An explicit `TopologyChange` with a strict expected content hash edits only
one **directed relationship** record by producing a new immutable source
snapshot. Deactivation retains the old relationship record with `active=False`;
readding a previously declared inactive pair is a conflict until a separately
specified reactivation operation exists. No reverse edge, direct interaction
or transmission is inferred.

`round_index` retains the original cognitive/story logical round: a topology
edit does **not** claim the story, cognition or social memory advanced.
Instead the new source snapshot content hash identifies this distinct editing
revision. This layer does not write the live social-memory state from which
other V19 network projections can be regenerated. Such integration requires a
separate explicit actor/runtime operation and must not happen implicitly.

## API

`narrative_dynamics.abm.mesh_topology_events` exports:
- `TopologyOperation.ADD` and `TopologyOperation.DEACTIVATE`
- immutable `TopologyChange` binding one event, source content hash,
  selected relationship relation, actor, reason and directed endpoints
- `apply_topology_change(source, event)` returning a new source and receipt
- `begin_topology_replay(source)`, `append_topology_change(state, event)`
  and `replay_topology_changes(source, tuple(events))` with a bounded
  append-only in-memory receipt chain. The next event must name exactly the
  previous authoritative content hash; replay of out-of-order input fails.
- `observe_topology(mesh, relation=...)` which reuses existing
  `find_mesh_path` and `mesh_edges` to report directed edges, ordered
  reachability, path distances, strongly connected components and defined
  directed diameter on finite selected snapshots.

Errors are typed: stale source version, duplicate event identity, unknown
Agent, conflicting/missing directed edge, unsupported relation or malformed
input. All failures leave the original snapshot unchanged.

## Review and acceptance

- [ ] Valid directed ADD and DEACTIVATE; never synthesize reverse edges or
      non-relationship transmissions.
- [ ] Deletion invalidates an existing `MeshPath` and
      `mesh_evidence` snapshot binding, and makes a now-unbacked
      `MeshBridge` fail closed (not silently pruned).
- [ ] Identical event stream / initial source produces identical source hash,
      receipts and V24 projection; conflicting event order/identity fails.
- [ ] Canonical checked-in versioned input/output vector with SHA/lineage
      checked byte-for-byte in focused and full Python CI.
- [ ] Full exact-head Python/Lean/BB/PathN/World Studio/Playwright CI.

This does not prove graph evolution preserves six hops, update arbitrary
human social facts, authenticate an external event, or implement a general
persistent event store. Reuse established network mathematical methods for
topology and empirical comparisons, not as new theorem claims.

## Versioned replay vectors and CI

`tools/render_topology_revision_vectors.py` deterministically generates
`conformance/topology_revision_v1.json` from real source fixtures and the
production topology edit/replay API. The fixture includes a two-event stream
with explicit expected source hashes, chained receipts, before/after selected
edges and directed reachability/component observations, stale path/proof
rejection, unchanged interaction/transmission and fail-closed invalid Bridge.

The focused R1 Python CI, full Python tests and independent Lean acceptance
lane compare regenerated **UTF-8 bytes** against the checked-in file. The
initial unpinned-vector run must report RED and expose the reviewed generated
contents; no CI task may silently accept or overwrite a missing vector.
