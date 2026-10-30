# R2.M5d — Observe actual Agent actions after V19 social evidence (no trust-only claim)

Tracks [Issue #120](https://github.com/qigao/lean/issues/120). Stacked
on [Draft #123](https://github.com/qigao/lean/pull/123), parent
M5c epistemic evidence. Reuse the original V19/SQLite cognition, social
memory, world event resolution and source Story exclusively. No new
Agent policy engine, custom action executor, social graph owner, or
synthetic causal inference.

## Exact research distinction

M5c proved a *finite source mechanism* within the existing model:
opening an authorized passage changes actual TELL fidelity and
testimony admission; Bob's own INSPECT contradicts the source claim,
decreasing **Bob's judgment of Alice 0.5→0.3** and retiring only
**Alice→Bob** from the V19-derived RELATIONSHIP view. The observed
\`approved\` belief surpassing 0.7 is a *classification*, not evidence
that an Agent carried out a post-trust behavior.

M5d checks whether the *existing native decision owner actually
executes another round* after that social update. Even if the selected
action differs between the original WAIT/OPEN branches, those branch
histories also differ in passage state, location and perceived content.
Therefore **do not claim that the trust delta alone caused any action**.

## Native implementation / replay protocol

For each of exactly two source-qualified parent M5c branches:

1. Reexecute the same six native V19 Story rounds using
   \`replay_situated_story\`, preserving source event identity.
2. Initialize the original cognitive+social state and run one complete
   \`simulate_situated_percept_social_cognitive_round\` using its
   own SQLite database. This produces a *seventh* world round, the
   admitted claims/verification and the next source-owned social trust.
3. Pass the actual result's \`next_story\`, \`next_cognitive_state\` and
   \`next_social_state\` with the **same SQLite database** to the same
   original engine for an *eighth* round.
4. Read the emitted Bob \`SituatedCognitiveDecision\`: registered
   feasible actions, MAP-selected policy action, exact event
   success/failure/outcome and source decision/event hashes.
5. Read any new source-linked **Alice TELL** admissions from the
   native recall result and verify their weight uses the real
   current Bob→Alice directional trust. If there are no such
   admissions, report zero; the code must not fabricate exposure to
   establish a trust effect.
6. Compare the two actual actions **descriptively**, not as an
   identified trust-only treatment effect. Different enacted actions
   can have multiple explanation paths and cannot be attributed
   solely to the 0.5→0.3 trust difference in this design.

The output's \`actual_native_actions_observed=true\` means the native
eighth-round action owner ran and emitted a source event. It does NOT
mean that the actions necessarily differ or that trust caused
the action. A failed native action remains a real attempted event
and must not be counted as a successful outcome.

Immutable reports must bind the original M5c source report hash,
per-branch V19 prior/next Story and social hashes, native policy
decision and action event hashes, the observed per-branch trust
inputs, bounded eligible action roster and (if any) source-qualified
trust-weighted recalled testimony. No private TELL/INSPECT contents.

## Scientific admission rules and future isolation

- Model-local deterministic pair: sample_size_per_arm=1,
  causal_attribution=UNATTRIBUTED, confidence_interval=NOT_ESTIMABLE,
  external_validation=NO_EXTERNAL_VALIDATION.
- Trust isolation=NOT_ISOLATED_BETWEEN_SOURCE_BRANCHES,
  post_trust_action_effect=TRUST_ONLY_CAUSAL_EFFECT_NOT_ESTABLISHED.
- No global elite score; no inferred universal direction of social
  change; centrality is not permission or credibility.
- A future **trust-only** experiment must create a legitimate
  matched source-state intervention that changes only V19 source
  trust through the original evidence mechanism while controlling
  the observed world and cognitive state. It must also demonstrate
  the next native decision consumes that trust through actual
  source-qualified memory admission or another explicitly
  authorized effect. Simply editing a JSON trust value is forbidden.
- Empirical population inference requires independent samples,
  source-owned external observation/holdouts, declared confounders
  and a preregistered outcome, none of which exist here.

Regenerate one reviewed source JSON vector with the existing
\`proof.yml\` V24 Python and independent Lean source-conformance
lanes. Preserve all existing full Python, 35+11 theorem trust,
BB/PathN, Studio/Playwright and bounded CI singleflight. No
extra workflow, consumer, installer, fallback, second parser
or new six-hop theorem.

## Native source result — negative trust-to-action finding

The original [CI #41089014322](https://github.com/qigao/lean/actions/runs/41089014322)
successfully ran the native seventh and eighth source-world cognitive/social
rounds, and then deliberately failed closed because the reviewed JSON
source corpus had not yet been pinned. Its **5,196-byte source-generated
JSON**, with exact report digest
`sha256:67b1b49ae47be43ff3405acda0842f5bfae47786e09368232b0dde5cb6f2d2fa`,
is now checked in as
`conformance/network_post_evidence_actual_actions_v1.json`.
One-time stdout bootstrap has been removed; independent Python and Lean
gates require exact byte reproduction.

**Crucial model result:** Bob's eighth-round original native
`SituatedCognitiveDecision` selected **MOVE in BOTH branches**,
each from registered feasible actions `[move, wait]`. In the WAIT
branch, the real attempted MOVE failed with `passage_closed`; in
the authorized OPEN branch, the same selected MOVE succeeded with
`moved`. The decision probabilities were about 0.99963 and 0.99960,
respectively, but the selected action ID did **not** differ.

The original social owners reported directional trust in Alice of
0.5 (control) and 0.3 (open), but **neither eighth-round Bob decision
had any NEW trust-weighted Alice TELL recall**: count 0 in both
branches. Hence this experiment did not demonstrate a trust-to-recall-
to-choice effect and must not be cited as such. The different actual
MOVE outcomes reflect the original source passage state.

Both branch cognitive beliefs again changed by source-native
observations, but the 0.7 adoption classification still does not
constitute an action rule. The status remains
`TRUST_ONLY_CAUSAL_EFFECT_NOT_ESTABLISHED` /
`NOT_ISOLATED_BETWEEN_SOURCE_BRANCHES`, not a population causal result.

A future trust-isolated design must use a valid source-owned
intervention and match world location, previous observations and
candidate actions, then verify that the original decision path
actually consumes a *new* source-qualified trust-weighted admission.
Do not hand-edit trust or invent evidence. Keep all full Python,
Lean and Studio gates at the new exact HEAD.
