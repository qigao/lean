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
