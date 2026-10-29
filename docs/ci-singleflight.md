# Proof CI — one current run per PR

The repository's proof and World Studio workflows have been consolidated
into `.github/workflows/proof.yml`. Feature branches are verified once from
`pull_request`, while `master` and `proof/narrative-dynamics-v0` retain
their push checks. `workflow_dispatch` permits an explicit acceptance rerun.

A PR source ref is a workflow concurrency group with
`cancel-in-progress: true`: later commits supersede older queued and running
runs for that PR. Other PR groups do not cancel one another. All four jobs
(V24 focused Python, full Python, Lean, World Studio) use one repository-wide job concurrency group,
`queue: max` and `cancel-in-progress: false`: only one CI runner job is
active, with unrelated PR jobs retained in the shared queue.

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

The V24 source projection adds a focused Python gate before complete Python.
It retains both the 35-theorem foundation trust audit and the V24 projected
mesh proof audit in the Lean job, with the same single runner concurrency
and exact-head source checks. The four stages do not bypass any full tests.
