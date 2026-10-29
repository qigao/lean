# Proof CI — one current run per PR

The repository's proof and World Studio workflows have been consolidated
into `.github/workflows/proof.yml`. Feature branches are verified once from
`pull_request`, while `master` and `proof/narrative-dynamics-v0` retain
their push checks. `workflow_dispatch` permits an explicit acceptance rerun.

A PR source ref is a workflow concurrency group with
`cancel-in-progress: true`: later commits supersede older queued and running
runs for that PR. Other PR groups do not cancel one another.

Full Python and Lean run **in parallel**, each with a repository-wide
`queue: max`, `cancel-in-progress: false` concurrency lane. The existing
`lean-ci-runner-${{ github.repository }}` lane serializes full Python and
World Studio and is backward-compatible with other branches still on the old
single-runner configuration. The additional proof lane is
`lean-ci-runner-lean-${{ github.repository }}`. This bounds managed CI to at
most two concurrent jobs across the repository while preserving queued jobs
from unrelated PRs.

World Studio has `needs: [python-tests, lean-proof]`: it runs only after
both independent test families finish. The `!cancelled()` job predicates
preserve Lean and Studio diagnostic evidence even when Python fails, with the
overall workflow still RED. The original test commands and strict proofs
remain unchanged.

The pipeline preserves all prior Python, Lean, BB/PathN, Studio and real
Playwright checks, plus exact-head checkouts. Previous fitness branch
exclusions remain in effect. Failed Python or Lean jobs remain RED while later
independent proof/UI jobs can continue; cancellation stops the chain.

Caches: setup-python pip, setup-node npm, `~/.elan`,
`.lake/packages`, the new `.lake/build` cache keyed by pinned Lean and
revision with a compatible restore prefix, and Chromium at
`~/.cache/ms-playwright` keyed by the frontend lockfile. Restoring caches
never substitutes for compiling, proof checking or running the real browser.

The cancellation policy applies only to runs created after this workflow is
present on their source ref. Unmerged historical PR branches using the old
workflow continue with their old policy; their old runs are not retroactively
cancelled. Once this PR is merged, subsequent branches from this base inherit
the policy.

Acceptance requires exact-head full CI success; a warm run must exhibit
cache restore HIT logs before measured speedup may be claimed.

## Warm-cache acceptance

The first run on a cache scope can MISS and save correctly. Subsequent
commits restore Lean products by the pinned-toolchain cache prefix, while
Chromium uses the frontend lockfile as its key. Build time and exact-key
cache-hit outputs are logged explicitly. Note: an exact-key result of false
can still accompany a successful Lean *prefix restore*. Read the cache
restore-key log before asserting cold vs warm behavior, and compare actual
elapsed times before asserting a speedup.

Original design-spec and plan path exclusions remain on push and PR events.

## V24 source-backed projection

A V24-focused source-backed Python regression runs first using the shared
runner queue. Full Python and Lean Proof both depend on that admission gate
and then run in parallel on their separate bounded lanes. World Studio waits
for both. The complete 35-declaration foundational audit and V24 projection
Lean tests/11-declaration trust audit remain strict and unchanged.

## Runtime V19 evidence and finite Lean conformance

The focused Python admission also validates genuine V19 source-derived
membership, bridge, selected relation, gateway cover and negative mutation
vectors. A separate finite Lean conformance stage runs immediately after the
full Lean library build and before costly BB/PathN theorem suites, using
`tools/check_mesh_evidence_lean.sh` and computable `decide_cbv` on generated
relations. Full Python and Lean retain their separate bounded concurrency
lanes, and World Studio waits for both families.

## Exact-byte corpus pin and real Lean mutation RED

The SHA-256 of the entire regenerated Lean input is checked against
`tests/fixtures/mesh_evidence_lean.sha256` before acceptance. Both the
focused Python regression and the proof job check the same canonical UTF-8
bytes. This is a pinned digest, **not** a separately checked-in Lean corpus;
any intentional generator or fixture change must be reviewed and re-pinned,
never silently auto-updated by CI.

After the unmodified corpus passes the Lean kernel, two independent negative
corpora remove (1) an actual selected-relation source edge and (2) the
associated source-backed directed bridge from the two-society gateway case.
Each deliberately retains the original *positive* projected-edge claim.
They must fail under real `lake env lean` (not merely a Python assertion),
while the authoritative evidence validator also rejects both. A compiler
timeout or missing Lean diagnostic is **not** counted as a negative success.

This adds counterfactual mutation gates; it does not yet kernel-check the
full concrete gateway-cover premises or every runtime path witness.

## Concrete runtime paths and gateway kernel witnesses

`tools/render_mesh_reach_lean.py` reuses the *same* authoritative evidence
and existing `projectedMesh` finite renderer. Every exported path now becomes
an exact-length Lean `MeshWalk` with each projected edge independently checked
by the kernel and a concrete `ReachWithin` hop bound.

For the fully covered directed two-society fixture, Lean checks the designated
gateway's society membership, every agent assignment, both directions of
the one-hop cover, and every directed society route under four hops. The
generator **does not** repeat the already-known conditional six-hop bound
for each fixture. The Python `premises_satisfied` flag selects a positive
fixture but is not admitted as a Lean premise. Other fixtures cover a directed
chain (local paths only), overlapping memberships, an outsider's impossible
local zero-hop claim, and nonempty selected interaction and transmission.

The original pinned 70-edge corpus and its digest stay unchanged. The
additional concrete generator runs in the same bounded Lean CI step and
must produce real kernel GREEN. Exact-byte pinning of this extended corpus,
and additional identity/path mutation-to-RED cases, remain future work.

## Extended-source identity and real negative path witnesses

Each concrete path corpus case now reifies the authoritative immutable V19
snapshot hash separately from the evidence's claimed source hash, then has
the Lean kernel check their finite literal equality. This is **not**
cryptographic verification within Lean and cannot authenticate arbitrary
external processes; canonical Python validation against the authoritative
snapshot still determines which evidence may enter production proof generation.

Two additional real Lean negative cases re-use those exact schemas:

- `stale-path`: the original V19 one-hop route `a0 → a1` was valid, but
  the next immutable source revision removed that relationship edge. The
  authoritative validator rejects the old evidence, and the Lean kernel must
  refuse the obsolete one-hop `ReachWithin` theorem under the fresh graph.
- `identity`: change the authoritative snapshot round and replay the
  previous source-hash claim, **recomputing the candidate evidence hash**.
  Python must reject the stale lineage, and the finite Lean equality claim
  must fail independently.

The extended corpus's SHA-256 is printed from the exact generated bytes,
and `tests/fixtures/mesh_reach_lean.sha256` must match. CI never silently
updates this reviewed pin; the rollout's initial digest-discovery run must
fail closed if no checked-in pin exists. No earlier source/bridge mutation
tests, fixed 70-edge corpus pin, 35+11 trust audits, or full workflow steps
are removed.

## Exact-head extended corpus pin — accepted digest candidate

The full extended runtime path/Gateway/identity Lean corpus is committed by
SHA-256 in `tests/fixtures/mesh_reach_lean.sha256`:
`69890c34922216660ce0ce79727ebfdcccc5bbbbba212a932a0d843e4ca09216`. The digest was read from actual CI proof job #40903497184,
before its intentionally fail-closed missing-pin result. Both focused Python
and the Lean proof script now independently recompute the canonical exact
UTF-8 bytes and compare them to the reviewed pin; the shell aborts on a
mismatch before executing negative mutation tests.

Negative mutation admission also requires the real Lean diagnostic from
the deliberately false `decide_cbv` path or source-hash `decide` equality;
a generic compiler diagnostic, timeout, or missing Lean tool is not enough.
The positive runtime source/graph ownership, existing 70-obligation pinned
corpus and all independent trust/World Studio gates remain unchanged.

## Reviewable versioned V19/V24 runtime proof vectors

`conformance/mesh_runtime_vectors_v1.json` is the **full, readable, ordered**
versioned canonical evidence corpus, not just a SHA-256 commitment. It is
regenerated from the same immutable authoritative `MeshSnapshot` fixtures
used by both `render_mesh_evidence_lean.py` and
`render_mesh_reach_lean.py`: three real office source-relation cases plus the
directed chain, positive gateway cover, overlapping memberships, nonempty
selected direct interaction and nonempty transmission examples.

Each vector includes the version/schema, selected relation, source model /
round / story / cognitive / social hashes, agent roster, society membership,
source/projected edges, directed bridges, path witnesses, gateway configuration
and check results, and evidence digest. No new runtime graph owner exists.

On every PR HEAD the focused Python job first regenerates the canonical UTF-8
JSON and performs strict **byte-for-byte `cmp`** with the checked-in vectors.
The independent Lean lane repeats that comparison before it generates the
kernel-checked source/bridge/path/Gateway proof obligations; focused and full
Python tests also validate the same bytes. Any missing or altered bytes are
RED, not silently regenerated into the repository. Digest pins of the two
generated Lean corpora and all real mutation-to-RED / source trust audits
remain in force. These vectors do not constitute cryptographic authentication
of an external process or a universal theorem about changing runtime graphs.

## Research direction and retired proof/bootstrap duplication (2026-10-30)

Canonical direction: [network-research-direction.md](network-research-direction.md),
Issue #96. Established six-degree methods are **reference tools**, not
research to re-prove. We continue to check the runtime-derived Agent/Society
relationship, source-backed edges, concrete paths, gateway assignment,
source lineage and observed-data versions. Re-running the existing abstract
six-hop result on every fixture added no new model correctness guarantee,
so that generated redundant example has been removed. The original
foundational mathematical library and its trust audit remain in place.

The initial `--transport` JSON-chunk bootstrap and corresponding CI
fallback are now deleted: the reviewed versioned JSON file exists and must
be checked against a fresh authoritative export with strict `cmp`.
The two existing Lean corpus SHA-256 pins remain mandatory; an intentional
change to the extended corpus requires recording and reviewing its new
actual digest on an exact-head CI run. No hashes or vectors are silently
updated.
