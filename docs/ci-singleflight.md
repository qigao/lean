# Single-flight Lean CI policy

This repository uses **one authoritative CI workflow**:
`.github/workflows/proof.yml`. The former standalone World Studio workflow
is incorporated as its final `verify` job.

## Which event runs CI?

- Pull requests: **one** workflow per current PR head. All tests are executed
  against the exact source head SHA rather than the synthetic merge ref.
- Push: only `master` and `proof/narrative-dynamics-v0`. A feature branch
  with an open PR does **not** also launch a second push workflow.
- A new commit on the same PR source ref supersedes earlier queued/running
  workflow instances with `cancel-in-progress: true`. Historical runs made
  under the previous workflow config cannot be retroactively canceled by
  this declaration.

## One runner, all required gates

Stages run in the following strict order using `needs`:

1. V24 focused Python regression (when the V24 tests exist on that ref).
2. Complete Python `unittest discover` (retaining the original fitness-branch
   exclusion).
3. Lean library build, BB/PathN contracts, mesh trust audits and runtime
   conformance, including the old complete theorem suite.
4. World Studio server, browser, typecheck/build and real Playwright flow
   (retaining the original fitness-branch exclusion).

Jobs share `lean-ci-runner-${{ github.repository }}` with
`queue: max` and `cancel-in-progress: false`. The shared group limits
this workflow to **one active runner job across PRs** without discarding jobs
from unrelated PRs. Each stage uses `if: !cancelled()` where it depends
on an earlier stage, so a failed earlier gate remains RED but later gates
continue collecting independent evidence; cancellation stops the chain.

## Warm caches

- Existing `~/.elan` and `.lake/packages` caches remain keyed by the pinned
  Lean toolchain and Lake project dependency configuration.
- `.lake/build` has a revision-keyed cache with a matching toolchain and
  project-key prefix for cross-commit warm reuse. Lean still checks source
  dependencies; cached compilation does not skip proof or axiom gates.
- Python `setup-python` pip and Node `setup-node` npm caches remain pinned
  to the corresponding requirements/lockfile.
- Playwright browser binaries in `~/.cache/ms-playwright` are cached by
  `world_studio_web/package-lock.json`. The locked `npm ci`, Chromium
  installation/dependency check and real browser flow still run.

## Acceptance

Confirm that (a) a PR head generates a **single** CI workflow, (b) a
subsequent commit cancels the prior CI run in the same PR group, (c) the
latest head passes the focused, full Python, Lean and World Studio jobs,
(d) the warm runs show actual cache restore evidence in job logs, and
(e) no test was disabled or hidden via `continue-on-error`.

The global job lock applies to this consolidated workflow. Other unrelated
GitHub Actions workflows that do not use the same runner group are not
covered by this repository-local policy.
