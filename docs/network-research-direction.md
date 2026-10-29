# Network research: executable models, not re-proving six degrees

**Status:** canonical project direction, 2026-10-30. **Tracking:** [Issue #96](https://github.com/qigao/lean/issues/96).

## Mathematical boundary

Six degrees of separation, small-world structure, Watts–Strogatz constructions,
basic reachability, path bounds and other standard graph methods are established
literature and reusable mathematical tools. The project does **not** claim novelty
from proving them again or accumulating generated copies of a known upper-bound
theorem. Existing Lean theorems and transitive trust audits remain available as
maintained mathematical infrastructure, not new publication/research milestones.

**Mathematical theorem**, **runtime conformance result**, **simulation
observation**, **empirical predictive claim**, and **causal/mechanistic claim**
must always be reported separately, with assumptions and scope.

## Source-of-truth and implementation boundaries

- **V19 authority:** immutable `SituatedNetworkSnapshot`, with stable
  `model_id/model_hash`, round, story, cognition, social state and source
  content hash. No second canonical graph owner, no mutation of historical
  snapshots, no synthetic edges invented by projection.
- **V24 derived projection:** `MeshSnapshot` chooses relationship,
  direct interaction or realized transmission explicitly. Members may overlap
  across Society identities; physical placement is distinct. Directed
  bridges require an existing selected source relation and endpoint
  membership, never infer symmetry or backwards connectivity.
- **Runtime contracts:** deterministic finite paths, query-scope validation,
  same-source replay, typed rejects, bounded resource ownership and
  explicit provenance. Gateway and Society evidence is checked as a concrete
  topology **input**; testing it is not discovering a six-degree theorem.
- **Proof boundary:** Lean may kernel-check a genuinely new implementation
  invariant or finite conformance obligation. A finite `Fin n` test does not
  prove every Python execution, authenticate an outside process, establish
  asynchronous transmission or guarantee an arbitrary network's diameter.
  Retain trust audits, no `sorry`, `native_decide` or oracle/fallback proof.
- **Existing accepted baseline:** #104 source-to-projection correspondence,
  versioned `conformance/mesh_runtime_vectors_v1.json`, two pinned Lean
  corpora, four negative source/bridge/path/identity RED tests and exact-head
  full CI. Keep these guards while avoiding redundant proof-generation.

## Actual research and engineering priorities

| Phase | Work | Evidence / acceptance |
|---|---|---|
| R0 | Source-backed static projection and API correctness (#101, #103, #104) | Real V19 input, checked-in vectors, relation-isolation and stale-revision rejection; complete Python/Lean/Studio CI |
| R1 | **Executable topology changes**: additions, deletions, rewiring, bridge and Gateway failure, Society split/merge, and temporal relation updates | A versioned `(snapshot, event, seed) -> (next_snapshot, result, provenance)` contract; deterministic replay, transition validation, no lost Agent/edge ownership |
| R2 | **Model comparison**: growth and attachment (including BB), small-world/rewiring and candidate pseudofractal families | Run the actual generators on common initial conditions/seeds; compare directed reachability, components, shortest paths, clustering, centrality, degree, community structure and robustness; no unconditional "six hops" pass gate |
| R3 | **Network–agent coevolution**: belief/exposure/propagation, interventions and schedule sensitivity (#96, #99) | Reuse authoritative ABM and post-incoming exposure semantics; time-series trajectories, finite-model counterexamples, exact/approximate claim taxonomy; no duplicated execution semantics |
| R4 | **Empirical and product validation**: external data, calibration, held-out testing, World Studio (#39, #43, #56) | Frozen dataset/source revisions, splits, metrics and model versions; reproducible evaluation and confidence/uncertainty reporting, with predictive-vs-causal firewall |

R1–R4 are research work to design and implement; they are **not** represented
as already complete by existing Lean proofs or a finite gateway fixture.

## Code and PR acceptance rules

1. One implementation per model; prefer existing
   `narrative_dynamics/abm` runtime and reused source data. New graph
   algorithms need justified behavior and testing, not a second private
   state authority.
2. New math work is justified only by a **new statement tied to executable
   model behavior**, such as a transition invariant, probabilistic identity,
   convergence condition or counterexample. Cite established methods instead
   of duplicating their proofs.
3. Every new PR must state the concrete API/behavior, model identity,
   assumptions, real fixtures, numeric or formal evaluation and exact-head
   CI. Distinguish `PROVED`, `DISPROVED` and `UNKNOWN`, simulation-only
   evidence and external scientific claims.
4. Keep the existing cache, superseded-head cancellation and bounded
   concurrency. No skipping original Python, Lean, browser or provenance gates
   to make refactoring GREEN.
5. Keep a narrow stack: **#101** holds V19→V24 projection and folded-in
   theorem trust audit formerly tracked by #100; **#104** holds independent
   runtime-derived conformance. #59 is closed as obsolete docs-only design.
   Neither should be presented as a new proof of six degrees.
6. #60 pseudofractal is a **parked candidate** pending runnable model and
   model-comparison acceptance; #69 Flywire and #1 browser safety are separate
   workstreams, not duplicates to delete as part of this network cleanup.
   Completed #94 / #38 are closed; #39, #43, #56, #96, #99, #102 and #103
   retain distinct unresolved responsibilities.

## Immediate next implementable milestone

Specify and implement the smallest **V19 topology transition/replay API**:
one source-authoritative immutable revision, explicit directed relation
addition/removal, an append-only event record and cross-revision path
invalidation. Demonstrate in a single deterministic test that an old path
becomes stale after edge deletion, the new snapshot remains authoritative,
and derived V24 projections / statistics follow only the updated source.
Then add an actual BB/rewiring comparison rather than an additional
six-degree theorem.

Do not merge the stacked #101/#104 changes before their original full
acceptance conditions and valid target-branch integration checks are met.
