# R2.M4b — Model-native BB, directed rewiring and evolving comparison

**Tracker:** [Issue #110](https://github.com/qigao/lean/issues/110).
**Parent:** [R2.M4a Draft PR #115](https://github.com/qigao/lean/pull/115).

The goal is to qualify **existing** ABM model families under frozen native
inputs and compare their supported dimensions *without* asserting that
their raw edge counts, trust, transmission counts, time or causal units
are equivalent. This is a deterministic execution **dossier**, not a
new ABM simulator or a second V19 social graph state owner.

## Model and unit registry

| Family | Real existing API | Topology/clock | Source evidence limitations |
| --- | --- | --- | --- |
| BB fitness-weighted birth | `replay_bb_population` | Undirected seed/birth edges; global tick, birth count, local V1 population round and exact rational cumulative trace mass | Fitness mass is not an Agent trust or a causal effect estimate; one undirected BB edge is not one directed social relation |
| Endogenous rewiring | `simulate_rewiring_population` | Directed candidate edges; transmit using **prior** active topology, then update using next beliefs and hysteresis thresholds | Belief similarity is not a verified testimonial claim; candidate topology is not a physical boundary or V19 relationship |
| Unified evolving | `simulate_evolving_population` | Catalog membership, directed edges, propagation, **external truth feedback**, trust adaptation and post-round rewiring | TruthFeedback is an explicit model input/oracle, **not** a private V19 INSPECT observation |
| V19 situated Agent/network | `build_paired_coupled_case` using original `replay_situated_story` / `simulate_situated_percept_social_cognitive_round` | Source-owned world/story/percept/social state, passage boundaries and directed relationship/interaction/actual TELL edges | The exact same event ID in divergent stories is not the same event hash/history |

The family observation APIs (`observe_bb_growth`,
`observe_rewiring`, `observe_evolving`) rerun each exact original
native transition implementation with a bounded program (≤8 steps)
and record its canonical model/state/edge/event hashes and **native**
metrics. The combined `compare_native_model_families` report has
`cross_family_numeric_ranking=FORBIDDEN`, `normalization=NONE` and
`causal_attribution=UNATTRIBUTED`. No pooled "average edges per
agent", numerical winner, standardized trust effect or inferred
six-hop theorem is allowed. This cannot identify general population
causal effects from four non-equivalent finite fixtures.

## Frozen native scenarios

- **BB**: exact two-Agent connected seed `0--1`; one externally authored
  birth `2` attached to `1`, then an idle tick. Confirm undirected
  topology, native transition hashes, inherited data, global-vs-birth
  clock separation and exact cumulative rational tick mass.
- **Rewiring**: three native Agents `a,b,c` with directed `a→b`
  initially active, `b→c` inactive, frozen beliefs `a=1,b=0,c=1`,
  hysteresis dissolution 0.2 / formation 0.8. Execute two native rounds.
  Verify old edge transmits before new `b→c` becomes eligible next
  round. Do not manufacture any social-memory evidence.
- **Evolving**: model-native `a,c` active with inactive `b`; explicitly
  ENTER `b` and provide external truth feedback `a→b=1.0`.
  First round propagates and can learn trust `a→b=0.75`, then
  activates `b→c`; second native round may transmit there.
  External oracle truth is not a private Agent verification.
- **V19 reference**: re-execute the already-reviewed [#115](https://github.com/qigao/lean/pull/115)
  common-initial-world control vs OPEN_PASSAGE then identical
  MOVE/TELL schedule. Record only the source/model/story revisions and
  `DIVERGENT_STORY_BRANCH` flag as a *reference*, never replace its
  social-memory owner.

## Mandatory CI and audit

- [ ] All four source programs use the existing implementation and are
  **reproducible** with exact hashes on repeated execution.
- [ ] Native BB undirected births, rewiring candidate changes, evolving
  lifecycle/trust changes and V19 private TELL differences remain
  distinct units and explicitly **non-rankable**.
- [ ] Incorrect source types, invalid budgets and false numeric
  comparability claims are fail-fast rejected.
- [ ] Immutable `NativeModelFamilyComparison` returns fresh decoded
  data, never leaking an editable canonical source object.
- [x] Review and commit the exact source-derived JSON
  `conformance/network_model_family_comparison_v1.json`; require full
  UTF-8 byte equivalence in focused Python and independent Lean CI.
- [ ] Preserve R1, R2.M0–M4a, Python, Lean/35+11 trust, BB/PathN,
  Studio/Playwright and cache/singleflight jobs at one exact HEAD.

**Open empirical work:** model-family comparison does not yet estimate
Watts-Strogatz style clustering, conduct statistically valid matched
population samples, or prove how boundary, environment or message
content produces general network growth. Such science needs explicit
common-denominator metrics, frozen sampling/seeds, null models,
confidence intervals and a separately admitted semantic mapping.
Do not lower original Lean gates, add installer/consumer CI, or
turn this qualification tool into a fallback graph engine.

## First native-run evidence — 2026-10-30

The first focused [CI #41023471575](https://github.com/qigao/lean/actions/runs/41023471575)
ran the independent source-native BB, rewiring and evolving adapters and
V19 control reference. Five behavioral/source/semantics regressions were
GREEN. One negative test expected the wrong **error message**: the actual
fail-fast rejection correctly said "model family dossier requires three
native sources and a V19 reference". That assertion was corrected, with
no production change. The remaining expected RED was the missing reviewed
vector.

The actual 12,436-byte canonical output was inspected and committed
exactly, preserving the model-native clocks and identities:

- BB birth and idle: 2→3 nodes, undirected `0--1--2`, exact
  cumulative mass `1/2`; directed V1 transmission records are
  separate from undirected birth topology
- Rewiring: prior active `a→b` transmits first; new `b→c`
  becomes eligible for transmission only on the second round
- Evolving: b ENTERs, external a→b truth-feedback updates
  native trust `0.5→0.75`; rewiring then permits b→c
  transmission in round two
- V19: the already-reviewed source-owned, branch-qualified
  WAIT vs OPEN_PASSAGE / identical MOVE/TELL control remains
  `DIVERGENT_STORY_BRANCH`

No content payload, private TELL message, six-hop claim or numerical
cross-family winner was added. The initial first-run textual vector
handoff is removed. Same-HEAD focused/full Python and independent Lean
must reproduce these exact reviewed bytes without changing the corpus.
