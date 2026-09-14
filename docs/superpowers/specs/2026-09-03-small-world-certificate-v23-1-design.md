# Small-World Certificate V23.1 Design

## Objective

Advance the V23 social-mesh feasibility proof from one explicit bridge to a
composable, machine-checkable small-world criterion. The result must explain when
an Agent-level mesh has both a bounded global hop distance and strong local
clustering, without claiming that an empirical network automatically satisfies
those premises.

## Mathematical boundary

V23.1 is deterministic. It proves implications between explicit graph properties:

1. preserving every old edge preserves every old bounded walk;
2. a walk between society gateways lifts from the society graph to the Agent graph;
3. local entry, a bounded society path, and local exit compose additively;
4. one-hop gateway access plus a four-hop society diameter gives a six-hop Agent
   diameter;
5. symmetry, six-hop global reachability, and perfect local wedge closure construct
   a strong small-world certificate.

The certificate deliberately uses a strong qualitative clustering premise: every
pair of distinct outgoing neighbors is connected. For a finite symmetric graph
this corresponds to local clustering coefficient one wherever the denominator is
nonzero. V23.1 does not introduce finite counting or rational coefficients.

## Definitions

`NarrativeDynamics.Core.SmallWorld` adds:

- `MeshSubgraph g h`: every edge in `g` is also present in `h`;
- `SymmetricMesh g`: every active edge has its reverse edge;
- `GlobalHopBound g limit`: every ordered node pair is reachable within `limit`;
- `PerfectLocalClustering g`: distinct neighbors of one center are adjacent;
- `SmallWorldCertificate g limit`: a symmetric graph with a global hop bound and
  perfect local clustering;
- `SocietyGatewayModel agentGraph societyGraph`: society membership, one gateway
  per society, gateway membership, and a realization of each society edge as an
  Agent-level gateway edge;
- `OneHopGatewayCover model`: every Agent belongs to an assigned society and can
  reach and be reached from its gateway within one hop.

The existing `MeshWalk` and `ReachWithin` remain the sole path semantics. No
parallel distance implementation is introduced.

## Required theorems

### Edge and path monotonicity

- `MeshWalk.mapNodes` maps an exact-length walk through an edge-preserving node
  function.
- `reachWithin_append` composes two bounded paths and adds their budgets.
- `reachWithin_of_subgraph` proves that adding edges cannot destroy bounded
  reachability.
- `globalHopBound_of_subgraph` lifts that result to all node pairs.

These theorems formalize the safe part of shortcut addition. Arbitrary new edges
may introduce new open wedges, so V23.1 does not claim that edge addition preserves
perfect local clustering.

### Society-to-Agent lifting

- `SocietyGatewayModel.walk_lifts` maps an exact society walk to an exact walk
  between its Agent gateways.
- `SocietyGatewayModel.reach_lifts` preserves a society-level hop budget at the
  gateway level.
- `reachWithin_via_societies` composes Agent entry, lifted society travel, and
  Agent exit with budget `entryLimit + societyLimit + exitLimit`.
- `global_six_hop_bound` derives Agent-level global reachability within six hops
  from a one-hop cover and a four-hop global society bound.

### Small-world certificate

- `SmallWorldCertificate.closes_neighbor_wedge` exposes the local triangle edge.
- `SmallWorldCertificate.short_paths_survive_edge_addition` preserves the global
  hop claim when shortcuts are added.
- `smallWorld_of_gateway_cover` constructs an Agent-level certificate at limit six
  from the gateway-cover theorem plus explicit symmetry and clustering premises.

## Files

- Create `NarrativeDynamics/Core/SmallWorld.lean` for the definitions and proofs.
- Create `NarrativeDynamics/Tests/SmallWorld.lean` for executable theorem-use
  examples.
- Modify `NarrativeDynamics.lean` to include the module in the root library.
- Modify `.github/workflows/proof.yml` to run the theorem-use file.
- Modify `README.md` to state the V23.1 result and its non-claims.

## Non-goals

- No random graph generator or NetworkX dependency.
- No Watts--Strogatz probability distribution or expected-value theorem.
- No numerical clustering coefficient or average shortest-path implementation.
- No alternate attachment-model preferential attachment or power-law asymptotics.
- No temporal rewiring, distributed placement, P2P, WSS, or Raft behavior.
- No change to Python runtime simulation semantics.

## Acceptance

The root Lean library and `NarrativeDynamics/Tests/SmallWorld.lean` must compile
without `sorry` or `admit`. Tests must exercise edge monotonicity, society-walk
lifting, the one-plus-four-plus-one six-hop theorem, certificate construction, and
the certificate's clustering projection. Existing V23.0 theorem tests must remain
green.
