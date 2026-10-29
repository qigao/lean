# Proof CI — one current run per PR

The repository's proof and World Studio workflows have been consolidated
into `.github/workflows/proof.yml`. Feature branches are verified once from
`pull_request`, while `master` and `proof/narrative-dynamics-v0` retain
their push checks. `workflow_dispatch` permits an explicit acceptance rerun.

A PR source ref is a workflow concurrency group with
`cancel-in-progress: true`: later commits supersede older queued and running
runs for that PR. Other PR groups do not cancel one another.

The single workflow has **two bounded runner lanes**, both using
`queue: max` and `cancel-in-progress: false`:
- **Shared lane** (`lean-ci-runner-${{ github.repository }}`): V24 focused
  Python, full Python, and World Studio; compatible with the older baseline
  repository-wide lock.
- **Proof lane** (`lean-ci-runner-lean-${{ github.repository }}`): the Lean
  kernel/proof suite, one concurrent Lean job per repository.

The V24 focused Python gate runs first. Full Python and Lean **both depend
on the focused gate** and may then execute concurrently on separate runners.
World Studio explicitly waits for **both** complete Python and Lean to
finish before running. Even if one family fails, `!cancelled()` allows
independent tests and UI evidence to finish, and the overall run stays RED.
This bounds concurrency to two managed runner jobs while retaining queued
work from unrelated PRs.

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

## V24 focused gate

The V24 source projection adds a focused Python gate before complete Python
and Lean start independently. It retains both the 35-theorem foundation trust
audit and the V24 projected mesh proof audit in the Lean job, with the
two bounded runner lanes and exact-head source checks. No test is dropped;
World Studio joins the two independent families before running.

## Runtime-derived finite Lean conformance

The next V24 increment adds the bounded source-derived graph witness generator.
It runs a kernel-checked finite Lean gate immediately after `lake build`,
before the existing BB proofs, and expands the focused Python suite with
source provenance and malicious evidence mutation tests. It preserves the
two-lane concurrency bound, World Studio join and single-PR cancellation policy.
