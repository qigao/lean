# Deterministic Pseudofractal Mesh V23.4 Design

## Status and scope

The user approved the deterministic scale-free small-world direction on 2026-09-11 with `go`, after permitting verification through GitHub CI. This document is the written mathematical-design checkpoint. Review this specification before the implementation plan; it does not record implemented Lean theorems.

Create an independent branch from PR #58 head `c268e5355b14fb408f30814ed1c2bee2f41f1144`. Stack its draft PR on `feature/mesh-feasibility-v23`, not on PR #59. Do not modify or merge #58 or #59. V24 society projections, Python runtime authority, snapshots, serialization, hashes, execution, perception, transport, and persistence are unchanged.

The deliverable is a deterministic graph family, finite degree/clustering identities, and a verified finite six-hop example. It is not a preferential-attachment sampler, empirical social model, or unconditional theorem that a power-law degree distribution implies six-hop reachability.

## Mathematical model

Use the Dorogovtsev–Goltsev–Mendes pseudofractal scale-free web [1]. Index the triangle as G_0, not a single edge. All graph statements refer to one undirected, loopless, simple snapshot.

For every existing undirected edge {u,v}, one expansion adds exactly one fresh vertex w_{u,v}, retains {u,v}, and adds {u,w_{u,v}} and {v,w_{u,v}}. The entire old edge set is frozen before expansion. New edges do not participate until the following generation. The two orientations of one edge must not generate two vertices.

The mathematical carrier of one expansion is the disjoint union of old vertices and old unordered edges. Old–old adjacency is the old graph; old–new adjacency means that the old vertex is an endpoint of the new vertex's parent edge; new–new adjacency is false. Prefer Mathlib `SimpleGraph` and its unordered edges; expose its adjacency as the existing project `MeshGraph`.

Old-vertex inclusion is an embedding between different carriers. Lift walks using an edge-preserving node map; do not apply a same-carrier `MeshSubgraph` theorem across generations without a justified reindexing.

### Executable numbering

The finite executable view uses persistent IDs:

1. G_0 has vertices 0,1,2 and canonical sorted edges (0,1),(0,2),(1,2).
2. At each step sort the old edges lexicographically, always storing the smaller endpoint first.
3. For old vertex count N and edge index i, assign the new vertex ID N+i.
4. Return the sorted union of all old and newly added canonical edges.

Prove that this numbering is a bijection from the mathematical carrier and that adjacency is preserved and reflected. A numeric edge list with matching counts is not sufficient: two graphs can share counts or degrees without sharing distances. No independently assumed adjacency matrix may replace the recursively defined graph.

Use total finite indexing. Certificate dimension, endpoint bounds, and parent indices must be checked; no missing entry defaults to a valid zero distance.

## Required structural and degree results

The following are goals to prove from the graph construction, not fields assumed in a certificate.

- Each generation is finite, decidable, symmetric, loopless, and connected.
- The old graph embeds and every old bounded walk lifts.
- Actual undirected edge counts satisfy E_0=3 and E_{t+1}=3E_t, hence E_t=3^(t+1).
- Actual vertex counts satisfy N_0=3 and N_{t+1}=N_t+E_t, hence 2N_t=3^(t+1)+3. The division-free identity avoids unproved natural-division assumptions.
- Every surviving old vertex doubles its actual degree; each newborn has degree 2.
- For 1<=j<=t, exactly 3^(t-j+1) vertices have degree 2^j; exactly three vertices have degree 2^(t+1); all other degree counts vanish. At t=0 only the last clause applies.
- For 1<=j<=t+1, the exact tail count obeys 2*#{v:degree(v)>=2^j}=3^(t-j+2)+3.
- A constructive, deliberately non-tight bound `GlobalHopBound G_t (2*t+1)` supplies all-pair reachability for the existing distance metrics. A newborn is one hop from an old endpoint; applying this on both sides gives the induction step. This bound is not the claimed exact diameter.

The finite dyadic spectrum and tail counts are the formal scale-free foundation. The conventional exponent gamma=1+ln(3)/ln(2) is an interpretation of the discrete scaling [1], not an exact probability law at every integer degree. Real-analysis limits and a generic power-law implication theorem are outside this first change.

## Clustering contract

Reuse `localClusteringCoefficient` on the actual projected adjacency. Do not define a purported measured coefficient to be its anticipated closed form.

Prove that a node of degree k has k-1 undirected edges among its neighbors; equivalently it has 2(k-1) closed ordered neighbor pairs. Every node has degree at least two, so the denominator is positive. Consequently C(v)=2/degree(v).

Define mean local clustering as the rational arithmetic mean over all vertices, and derive:

    meanLocalClustering(G_t)
      = (12*6^t + 18) / (5*2^t*(3^(t+1)+3)).

The fractions here are rational, not truncated natural division. In particular meanLocalClustering(G_5)=51/64.

Do not confuse the mean of local coefficients with global transitivity, which weights vertices by their number of neighbor pairs. The exploratory G_5 mean is 51/64, whereas its global transitivity is 4/37. Only the mean-local claim is a formal deliverable here.

Do not require or weaken the existing `SmallWorldCertificate`. Its combination of symmetry, all-pair reachability and perfect local wedge closure forces distinct vertices to be adjacent, unlike this graph. Do not introduce another generic small-world label that hides the selected metric.

## Distance evidence and its soundness

The existing `shortestHopCount`, `meshDiameter`, and `averageShortestPathLength` are the semantic metrics. A computed table must be related to those definitions by a theorem; recomputing a number and asserting it as an assumption is not verification.

Use finite distance labels as a checkable certificate. For each source s, provide natural labels d_s(v) and a parent for each v != s. Check:

1. d_s(v)=0 if and only if v=s.
2. For v != s, parent p is a valid vertex, p is adjacent to v, and d_s(p)+1=d_s(v).
3. For every directed orientation of every graph edge u–v, d_s(v)<=d_s(u)+1.

Prove checker soundness. Condition 2 and induction on the label produce a walk of length d_s(v). Condition 3 and induction on any walk show d_s(v) is no larger than that walk's length. Therefore the label is the exact shortest hop count. This proves both upper and lower bounds without treating the search algorithm as trusted mathematical evidence.

For the global six-hop goal, additionally check d_s(v)<=6 for every source and target in G_5. One label equal to 6 then proves the diameter is exactly 6, not just at most 6.

For the G_6 negative control, one verified source row containing a label 7 suffices to refute a global six-hop bound. Full G_6 aggregate statistics are not required Lean acceptance goals.

The label generator may use deterministic BFS, but its output has no authority until the checker accepts it for the actual generated graph. Keep the checker internal to the model's metrics module initially rather than expanding the project's public proof API unnecessarily.

### Evaluation and trust

Structural proofs and checker soundness use ordinary Lean proof terms, without `sorry`, `admit`, or new user axioms. Prefer kernel-reduced finite checks when practical. The current Watts–Strogatz theorem tests already use `native_decide`; the same mechanism is permitted for the larger finite certificates, but its use must be explicitly reported.

Record `#print axioms` output for the concrete exported claims and distinguish native evaluation from kernel-only reduction. Do not call a native-evaluated certificate kernel-only computation. Do not replace a failed or expensive proof by a Python assertion, an assumed bound, or a skipped test.

## Concrete acceptance cases

### Small generations

Verify triangle indexing, frozen-edge simultaneous expansion, old-edge retention, unique newborns, symmetry, no loops, exact carriers, actual edges/degrees, and correspondence between numbered and mathematical views at t=0,1,2,3. Instantiate the generic structural and clustering theorems in test files.

### G_5: the six-hop witness

Required exact results:

    vertices: 366
    undirected edges: 729
    degree counts: {2:243, 4:81, 8:27, 16:9, 32:3, 64:3}
    mean local clustering: 51/64
    ordered distinct pairs: 133590
    ordered distance counts:
      {1:1458, 2:16944, 3:52692, 4:48432, 5:13296, 6:768}
    mean shortest path: 76373/22265
    global hop bound: 6
    diameter: exactly 6
    six-hop coverage: all 133590 ordered distinct pairs

Under the numbering above, nodes 109 and 362 have exact distance 6. A path is:

    109 -> 8 -> 0 -> 1 -> 5 -> 40 -> 362

The path alone proves only an upper bound. A verified label row must exclude every path of length at most five between those endpoints. Exactly 768 ordered pairs require six hops; none require seven.

### G_6: prevent an unconditional six-hop claim

Under the same numbering, nodes 354 and 1087 have exact distance 7:

    354 -> 36 -> 3 -> 0 -> 2 -> 13 -> 119 -> 1087

Verify a path and a distance lower-bound certificate. This proves that the same family can exceed six hops after growth. The G_5 bound must never be generalized to every generation by the term “scale-free.”

### Negative checks

Concrete certificate validation must reject a zero label for a non-source, a missing or invalid parent, a nonexistent parent edge, a nondecreasing parent label, and a label assignment that violates an edge inequality. Replacing the distance-6 label by 5 must not certify the G_5 target pair. Reject wrong table dimensions and a certificate checked against the wrong generation. Exercise these as finite checker tests, not as unprovable theorems patched with axioms.

## Proposed implementation surface

- `NarrativeDynamics/Core/Pseudofractal.lean`: one-edge-one-newborn expansion, graph family, numbering equivalence, structural counts, degree spectrum, and coarse reachability.
- `NarrativeDynamics/Core/PseudofractalMetrics.lean`: actual rational clustering metrics, internal finite distance-label checker and soundness, and links to existing distance metrics.
- `NarrativeDynamics/Tests/Pseudofractal.lean`: small instances, generic theorem applications, and invalid certificates.
- `NarrativeDynamics/Tests/PseudofractalSixHop.lean`: G_5 exact finite results and the G_6 seven-hop negative control.
- `NarrativeDynamics.lean`: additive imports of the new core modules.
- `.github/workflows/proof.yml`: separately named pseudofractal theorem-test steps, preserving every existing Lean and Python check.
- `README.md`: only verified theorem scope, generation convention, finite example and trust boundary.

Do not create Python runtime modules, package-root exports, network generators, transport, scheduler behavior, or automatic mappings to V24 societies. No changes to old metric definitions or existing fixtures are required. If the implementation needs a broader representation change, record that explicitly rather than silently replacing this model.

## GitHub CI and review gates

Before implementation, review this committed spec and then write a separate implementation plan. During implementation, capture an expected RED for missing declarations or rejected certificates, then GREEN for the new proof/test surface on an identified commit. A generic infrastructure failure is not the expected RED.

Use GitHub CI to run the root `lake build`, existing conformance checks, all old theorem tests, and both new test files. Give the new finite checks named steps so execution is visible. Record actual checked-out SHAs, event type, job IDs, and native-evaluation/axiom output; distinguish PR merge-ref verification from exact-head push verification. Preserve the full Python job, even if its baseline failure is unrelated.

The refreshed #58 proof run `33702197135`, attempt 2, reports Lean job `103113321382` successful and Python job `103113321545` failed. This validates the old Lean baseline only, not this new model; it does not make the overall workflow green. Do not repair unrelated Python research locks in this change.

The existing workflow ignores changes confined to specs/plans. This specification-only commit does not earn a new Lean or Python test pass. Keep the PR draft at this design checkpoint; no merge or automatic merge is authorized.

## Exploratory evidence, not formal verification

During this design session, an independent Python 3.13.5 probe generated generations 0 through 6 from the frozen-edge rule. Complete queue BFS was cross-checked for every source against synchronous bitset reachability layers. The probe also checked the parent/edge-label conditions and computed clustering from actual neighbor connections using exact rational arithmetic.

| Generation | Vertices | Edges | Mean local clustering | Mean shortest path | Diameter | Six-hop fraction |
| --- | ---: | ---: | --- | --- | ---: | --- |
| 4 | 123 | 243 | 519/656 | 7415/2501 | 5 | 1 |
| 5 | 366 | 729 | 51/64 | 76373/22265 | 6 | 1 |
| 6 | 1095 | 2187 | 18663/23360 | 775322/199655 | 7 | 199351/199655 |

These are freshly computed acceptance targets, not Lean proofs, GitHub CI results, or a formally verified Python implementation. The exploratory script is not a production deliverable. A later Lean implementation must construct the graph itself and prove/check these properties independently.

## Sources and interpretation

[1] S. N. Dorogovtsev, A. V. Goltsev, J. F. F. Mendes, “Pseudofractal scale-free web,” Physical Review E 65, 066122 (2002), DOI `10.1103/PhysRevE.65.066122`, arXiv `cond-mat/0112143`. The literature supplies the model and discrete-scaling context; it does not substitute for repository proofs. The finite numerical targets above were recomputed during this design.

Short topological paths do not prove decentralized route discovery, information admission, six-round delivery, or six-hop reachability after V24 removes edges. Any eventual runtime use must check the final selected snapshot and relation, not merely inherit the generator's mathematical label.
