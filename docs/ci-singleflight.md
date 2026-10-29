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
the one-hop cover, and every directed society route under four hops. Only then
does the generated proof apply `projectedMesh_global_six_hop_bound`; the Python
`premises_satisfied` flag is used only for choosing a positive fixture, not
as a Lean theorem or admitted hypothesis. Other fixtures cover an uncovered
directed chain (local paths only), overlapping memberships, an outsider's
impossible local zero-hop claim, and nonempty selected interaction and
transmission relations.

The original pinned 70-edge corpus and its digest stay unchanged. The
additional concrete generator runs in the same bounded Lean CI step and
must produce real kernel GREEN. Exact-byte pinning of this extended corpus,
and additional identity/path mutation-to-RED cases, remain future work.
