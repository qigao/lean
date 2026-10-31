# R2.M5f — Complete exploratory native V19 speaking-cost response grid

Tracker: Issue #120. Parent: Draft PR #127, which registered
a real two-choice original V19 Agent model and retained the
negative result that both branches choose and execute TELL.

## Study purpose — selected AFTER seeing the baseline

This is an **exploratory model-sensitivity grid**. The parameter
grid below is selected after inspecting M5e.2 source action
values, so it MUST NOT be represented as blind preregistration
or independent causal inference. The entire grid is nevertheless
frozen in both code and this design file **before the first M5f
grid execution**, and must be reported in full without filtering
unfavorable, unchanged, failed or uninteresting cells.

Research question: for which original V19 **model-local speaking
cost versions**, if any, would a V19 Agent select a different
MAP action after the confirmed vs contradicted source-history
evidence, using the already verified trust-weighted new TELL?

This is NOT a question of whether powerful people universally
cause events. Nor is a grid-search crossing proof that trust
alone caused a historical choice: the two prior source histories
still differ in private evidence, claims and cognitive beliefs.

## Frozen, bounded grid and exact version semantics

The complete numeric set is:

    speaking_cost = [0.0, 2.0, 3.0, 4.0, 5.0]

One new native V19 cognitive model is derived per cost, shared
between confirmed and contradicted history arms. In each model
only the original **Bob TELL action's goal reward** is changed,
**symmetrically across both hypotheses**, by:

    new_reward(goal, hypothesis, TELL) =
        original_reward(goal, hypothesis, TELL) - speaking_cost

All other Bob actions/rewards and all Alice/Carol/Dana
action/goal/reward models remain unchanged. In particular,
Bob's previously registered TELL repeatability remains True,
so at the target action decision both real original TELL and
WAIT actions must remain feasible. Preserve the original
TELL source-observation authorization constraint.

The original V19 social memory/SQLite owner, world resolver,
perception, likelihoods, action schedule, transition model,
goal weights, discount and beta must remain identical. The
registered cognitive-model hash (and consequently bound
memory/social-model hashes) must differ across cost cells.
The original social policy is unchanged, even though its
binding-containing model hash varies.

The **0-cost** grid cell must reproduce the source-native
M5e.2 model and already reviewed source report hash *exactly*.
The original 4,641-byte M5e.2 vector and the earlier 7,146-byte
M5e.1 vector must stay byte-identical under the shared owner.

## Complete native source experiment

For each of **all five** cost variants and **both** historical
arms, reexecute the same original V19 source world:

1. Prior authored Alice TELL confirmed or contradicted by Bob's
   own INSPECT; the original V19 social owner learns Bob→Alice
   source trust of 0.6 vs 0.3.
2. Original Alice emits a new source-qualified native TELL after
   that learning; original Bob's percept/cognition receives it as
   EXACT and applies the current directional source trust.
3. The source owner's own actual round-10 V19 Agent planner sees
   Bob's TELL and WAIT choices and computes real action values,
   finite-softmax probabilities and lexical MAP selected action.
4. Original V19 world resolver emits a real WorldEvent, recording
   its exact kind/outcome/success and source/decision hashes.

Report EVERY cost cell, BOTH history arms, the original full
TELL/WAIT action values and probabilities, native selected
action, actual outcome, and per-cell model/story hashes.
Retain zero-switch cells as first-class outcomes; never
delete them or tune an unregistered cost after seeing results.

The read-only reporter recomputes finite-softmax probabilities
from source-native action values, verifies every entry's two
feasible actions, and checks the old source's baseline digest.
It must not itself choose an action, write a social state,
generate private evidence or substitute another graph engine.

## Scientific limits and reporting

Use the exact labels:

- grid_design = EXPLORATORY_SELECTED_AFTER_M5E2_BASELINE
- source_history_comparison = CONFIRMED_VS_CONTRADICTED
- registered_model_change =
  SYMMETRIC_BOB_TELL_REWARD_MINUS_COST_EACH_HYPOTHESIS
- grid_completeness = ALL_REGISTERED_CELLS_INCLUDED
- scientific_scope = FINITE_MODEL_SENSITIVITY_NOT_CAUSAL_INFERENCE
- trust_only_causal_effect = NOT_IDENTIFIED
- sample_size_per_arm = 1
- confidence_interval = NOT_ESTIMABLE
- external_validation = NO_EXTERNAL_VALIDATION
- cross_mechanism_ranking = FORBIDDEN

A switch, if any, is a model-local response regime. A same-choice
result is equally important. Different source histories also carry
different prior private beliefs and claim status, so a switch cannot
be ascribed solely to trust. Comparing action values across
different speaking-cost models is NOT comparing the same unchanged
utility function: reward units are deliberately transformed.

The bounded grid is not a statistical sample. There is no
population effect estimate or confidence interval, and the
experiment does not establish an empirical general social law.
Independent historical network datasets, preregistered outcomes,
temporal confounder controls and genuine holdout regimes would
be needed for external/causal validity.

## CI acceptance and resource discipline

One original native source JSON vector is emitted for manual
review on the first focused V24 Python lane, then the full
output is pinned byte-for-byte and the one-time bootstrap
stdout removed. Re-run the original source in both focused V24
and independent Lean source-equivalence gates, while preserving
full Python, all Lean theorem/BB/PathN trust audits,
Studio/Playwright, cache and per-PR singleflight.

No new workflow, installer/consumer project, fallback graph
or algorithm, duplicate six-degree theorem, unreviewed source
revision, or reduced inherited CI gate.

Because a full grid costs more native model executions than
single-arm tests, the focused unittest suite shares one
setUpClass grid observation, and serializes that *same result*
for its pin check rather than re-running the grid inside the
test method. The original CI still independently regenerates
the vector through its entrypoint and compares exact bytes.

## First native source result — full grid, including every null cell

The first [CI #41152240777](https://github.com/qigao/lean/actions/runs/41152240777)
executed ALL five cost models and BOTH historical branches using
the original V19 source engine. It generated the complete
**19,047-byte** UTF-8 JSON vector, then stopped intentionally at
the mandatory missing reviewed corpus. There was NO source
runtime exception or reduced scientific/CI gate.

The entire actual output was reviewed, pinned in
conformance/network_native_speaking_cost_grid_v1.json, and
its report content hash is:
sha256:0bbd765d4bc6a75c0f1b5342fd235876ce46249cd4e39ccab7f2edc750c24363

Native original round-10 model-local results, without selection
or filtering:

| TELL cost | Confirmed (trust=0.6) | Contradicted (trust=0.3) | Different MAP? |
| --- | --- | --- | --- |
| 0 | TELL, success | TELL, success | No |
| 2 | TELL, success | TELL, success | No |
| 3 | TELL, success | WAIT, success | Yes |
| 4 | WAIT, success | WAIT, success | No |
| 5 | WAIT, success | WAIT, success | No |

Source original finite-softmax probability of TELL by cost:

| TELL cost | Confirmed | Contradicted |
| --- | ---: | ---: |
| 0 | 0.9999996967 | 0.9999974030 |
| 2 | 0.9992407941 | 0.9813246774 |
| 3 | 0.9529565979 | 0.4795749766 |
| 4 | 0.2733474091 | 0.0170706509 |
| 5 | 0.0068468193 | 0.0003183246 |

The cost-3 **native original V19 model** gives Bob two genuinely
feasible actions TELL/WAIT. In the confirmed-history case its
original action values are about 2.2943791 vs 1.5422544,
and the real world event is successful TELL (told). In the
contradicted-history case original values are about 1.3751066
vs 1.3955430, so MAP selects WAIT and the actual event is
successful WAIT (waited). All other cost cells are retained,
including cases with the same MAP outcome.

All five cognitive model hashes are distinct, source original
world and perceptual identities are common, and the original
authored historical Story hashes are identical across model
cost cells. Cost=0 exactly matches the reviewed M5e.2
source report hash and its earlier source corpus.
Private message and INSPECT contents are not exported.

**Scientific interpretation: exploratory and combined-history only.**
The cost grid was chosen AFTER seeing M5e.2's original source
baseline (not blind preregistration). The two history arms have
different private beliefs, claims and prior evidence, so even
the cost-3 action divergence cannot be attributed to trust
alone. Each cost is a separately versioned reward model;
the raw action utilities are not one invariant cross-model
unit. This is one deterministic source-generated finite
model-local response regime, not a population estimate
or a universal social/elite power law.

The reviewed vector has been pinned and the one-time
stdout bootstrap deleted. Original V24 Python and
independent Lean source lanes must reproduce exact bytes.
All full Python, Lean theorem trust/BB/PathN and
Studio/Playwright gates remain required on the NEW exact
HEAD, which is not yet fully CI accepted.
