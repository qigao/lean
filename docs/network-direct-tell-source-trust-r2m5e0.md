# R2.M5e.0 — Opt-in source-owned trust for new EXACT testimony

Tracks [Issue #120](https://github.com/qigao/lean/issues/120) and stacks
on [Draft PR #124](https://github.com/qigao/lean/pull/124).
We do not infer a universal key-player power score or any new theorem.

## Existing gap discovered during M5d failure analysis

In original V19 private cognition, a newly received other-Agent EXACT
TELL is admitted through \`admit_situated_percepts\` using a full
likelihood regardless of the social-memory observer→source trust.
Historical/previously stored TELL recalled from SQLite was already
tempered by \`source_trust\` in \`recall_situated_percept_memories\`.

Therefore, even after an authentic Bob→Alice trust update, the
untempered direct path can bypass source trust for a *new* exact TELL.
This is not a valid implemented trust→new-information→action pathway.

## One owner; explicit protocol version

The original \`SituatedSocialMemoryPolicy\` gains
\`trust_weight_direct_testimony: bool = False\`.

- **False (all preexisting models):** no extra serialized policy key;
  identical V19 model hash, same original direct admission semantics
  and all existing 4,335/4,245/6,426/5,196-byte source vectors.
  All original Python, Lean, Studio tests must remain unchanged.
- **True (new explicitly registered model revision):** the one
  existing \`simulate_situated_percept_social_cognitive_round\`
  passes \`_trust_by_observer\` (validated from current
  \`SituatedSocialMemoryState\`) to the same
  \`admit_situated_percepts\` routine. No second planner, Bayesian
  engine, graph owner or invented trust state.
- Only a received **external EXACT TELL** is tempered, using the
  same weight-to-likelihood transformation as original SQLite recall:
  adjusted likelihood = (1 - trust) + trust * original likelihood.
  This retains valid Bayesian posterior normalization. An Agent's
  own INSPECT, SELF observations and non-TELL percepts remain
  fully weighted. A zero trust weight recognizes the TELL event
  but leaves its cognitive belief unchanged.
- \`SituatedBeliefAdmission.source_trust\` provides strict typed
  provenance; default 1 is omitted from serializations so historical
  evidence bytes do not change. Do not interpret trust zero as
  permission failure or suppress the occurrence of a testimony event.

## Same real-world Story, original native decision

Four source-owned social *policy* variants replay one identical,
real V19 M5c authored open-gate Story with Alice INSPECT → OPEN →
MOVE → source-linked false TELL and Bob MOVE → private INSPECT.
The world, Story, perception model and cognitive model are identical:

1. \`legacy_unweighted\`: source model declares initial trust 0.5 but
   legacy direct TELL uses effective weight 1.0.
2. \`weighted_zero\`: explicit protocol on, source model initial
   trust 0.0, direct TELL effective weight 0.0.
3. \`weighted_half\`: explicit protocol on, source model initial
   trust 0.5, direct TELL effective weight 0.5.
4. \`weighted_one\`: explicit protocol on, source model initial
   trust 1.0, direct TELL effective weight 1.0.

The four variants have **different declared social policy/model hashes**.
Do not falsely claim the full combined cognitive/social models are
identical; this is a deterministic source protocol sensitivity
qualification, not an isolated real-world trust intervention.

The original \`admit_situated_percepts\` is inspected using actual
source-derived Bob EXACT TELL/INSPECT percepts; the same original
SQLite-backed source cognitive/social executor then takes one real
decision and performs one native world action. The report separately
binds every policy, current social trust, direct effective weight,
source event, sanitized percept, native posterior, decision/Story,
social state and enacted result. Nothing is hand-written into the
Story or V24 graphs.

Expected structural/semantic invariants: legacy and trust=1 yield
the same full-likelihood posterior; trust=0 yields no TELL-driven
belief change; trust=0.5 yields a distinct intermediate update;
Bob's own INSPECT always keeps source trust weight 1.

## What M5e.0 still does not show

A policy sensitivity experiment is not evidence that a real person's
lower trust *caused* a particular behavior. It does not guarantee the
MAP-selected action differs among four variants. The same base world
does not mean the configured social policies or prior trust states are
identical. Results are \`sample_size_per_arm=1\`,
\`trust_only_causal_claim=NOT_ESTABLISHED\`,
\`confidence_interval=NOT_ESTIMABLE\`,
\`NO_EXTERNAL_VALIDATION\`; no universal influence ranking.

**M5e.1** still must add a new source TELL **after** an authentic
V19 trust revision, prove exact event/observer provenance and that
the original subsequent decision consumes its trust-weighted
admission. A future trust-only causal design would also require
matched prior world, perception and cognitive states, plus the
actual action comparison and explicit confounder handling.

All review corpus bytes originate from original native runner
execution; the one-time bootstrap is removed after human review.
Existing Python/Lean source-equivalence and Studio/Playwright
gates remain unchanged in scope. No extra workflow, consumer
project or fallback engine.
