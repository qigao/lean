# V24 situated mesh implementation and acceptance

Source design: [PR #59](https://github.com/qigao/lean/pull/59), `docs/superpowers/specs/2026-09-11-situated-mesh-v24-design.md`.

Implementation: [PR #101](https://github.com/qigao/lean/pull/101), based on the post-#75 `proof/narrative-dynamics-v0` branch. Independent source/axiom trust gate: [PR #100](https://github.com/qigao/lean/pull/100).

## Authority

`project_situated_network_snapshot(model, story, cognitive_state, social_state)` from V19 remains the only authorized source of network state. `MeshSnapshot(source, societies, bridges)` holds this exact immutable snapshot and adds no runtime state owner or serialization changes. Society membership never changes `agent_id`, physical `place_id`, V1/BB evolution, story, cognition, action routing, or transmission semantics.

`MeshRelation.RELATIONSHIP` selects active relationship edges; `DIRECT_INTERACTION` selects available direct access edges; `TRANSMISSION` selects unique directed pairs from observed transmission records. They are never unioned. Bridge declarations must be existing directed source edges with valid society membership. Overlapping members are shared identities, **not** synthetic transmissions.

## Explicit proof boundary

`MeshPath` records source-anchored mesh hash, relation, scope, endpoints, and hop budget. A witness is a static directed walk, not a causally ordered transmission history or a cryptographic authentication. Validation accepts any matching valid walk, not only the shortest.

`assess_six_hop_prerequisites` checks a concrete selection of gateways and agent-to-society assignments against the exact sufficient conditions of Lean `global_six_hop_bound`:

- Every society gateway belongs to that society.
- Every source agent belongs to its assigned society and can reach its gateway and be reached from it in at most one directed hop.
- The society graph uses **only actual directed agent edges between designated gateways**; every ordered society pair is reachable within four hops.

A positive assessment supports a conditional `1+4+1=6` bound for that **selected immutable graph and relation**. It is an executable Python contract, not a Lean-kernel proof of a Python execution. The single seven-node `2+1+3` fixture demonstrates one conditional path, not these global prerequisites. Its directed chain deliberately fails the global premise checker.

Lean `MeshProjection` proves the abstract source/edge/walk lift, selected-relation fidelity, bridge monotonicity, local membership-qualified zero-hop exclusion, and absent-source-edge rejection. The pre-existing `SmallWorldCertificate` additionally requires symmetry and perfect local clustering; neither is implied by a six-hop assessment. No unconditional universal six-hop, empirical small-world, random Watts–Strogatz rewiring, or Python correctness theorem is claimed.

## Required validation

1. Focused Python `mesh-projection` job plus unchanged full Python discovery on identical, exact PR head.
2. Lean default build and `NarrativeDynamics/Tests/MeshProjection.lean` on that head, with all prior V23 tests preserved.
3. Independent #100 trust job, auditing core and test source and requiring kernel-generated theorem axiom reports, with no unsupported trust assumptions.
4. Keep both PRs Draft until both exact-head checks and the combined integration tree have passed; do not merge on a skipped/missing job.

## Later boundary

A production UI/consumer may request society snapshots from the V19 projector but cannot use the mesh to bypass the existing action/perception/admission pipeline. Persistence, distributed execution, and changes in BB/ABM agent evolution require separate explicit contracts and tests.
