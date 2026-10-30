# R2.M4c.1 — three one-variable paired model-native controls

**Tracker:** [Issue #117](https://github.com/qigao/lean/issues/117).
**Parent:** Draft PR #118 / M4c.0, stacked on Draft #116 / M4b.
All results are original native ABM replays, not a new V19 graph owner.

## Preregistered finite interventions

| Source model | Common native origin | Single intervention | Frozen horizon | Valid within-family observable |
| --- | --- | --- | --- | --- |
| BB | Same seed 0--1, fitness, newborn and subsequent idle | Newborn 2 target 1 (control) vs target 0 | Birth then idle; global ticks 1,2 | Unique undirected edges and their actual endpoint sets; density may stay equal |
| Directed rewiring | Same *base* model, same initial Agent beliefs/candidate topology | Formation similarity 0.8 vs 0.9; only this one model-native parameter changes | Two full propagation-then-rewire rounds | Post-rewire active candidate fraction and delayed next-round transmissions |
| Evolving | Same exact model and initial state, same no-feedback schedule | Only b ENTER in first round vs no membership event | Two lifecycle/propagation/feedback/rewire rounds | Active-member fraction and real enabled transmissions |
| V19 | Already-reviewed WAIT vs OPEN_PASSAGE on same world | Source-native boundary; source-owned story/Agent memory | Actual V19 story rounds | Independent reference only; do not promote to cross-model scalar |

The rewiring **whole model and initial state hashes necessarily differ** across
arms because the declared parameter is part of the model identity. Rather than
lying about identical run hashes, the report checks equality of base-model,
initial Agent states and exact directed candidate topology, and records the
different model/state hashes. First-round transmissions use the same prior
active topology. Newly formed candidates cannot transmit until a later round.

BB branches share the exact original seed-frame hash and have an identical
birth/idleness schedule other than the targeted attach endpoint. Their edge
count/density remains the same even when the actual topology differs. This
explicitly defeats treating equal density as identical network behavior.

Evolving arms share an *identical* native model and full initial-state hash;
external TruthFeedback is **absent in both**. This isolates the only changed
lifecycle event and avoids claiming V19 INSPECT verification has been modeled.

## Qualification and limitations

Each family report binds its source/model/initial/final revision hashes,
pre-registered program, exact native timebase and two phase-aligned
NativeMetric records. It computes exact within-family descriptive differences
only with the same metric unit, denominator, roster scope and phase, never
mixing model families.

This is a **finite deterministic simulation contrast**, not a valid
population-causal estimator. The root requires:

- normalization=NONE and cross_family_numeric_ranking=FORBIDDEN
- sample_size_per_arm=1; confidence_interval=NOT_ESTIMABLE
- external_validation=NO_EXTERNAL_VALIDATION; causal_attribution=UNATTRIBUTED
- V19 reference=SEPARATE_SOURCE_REFERENCE_ONLY

The checked-in JSON conformance vector must come from original reexecution
and match exact bytes in both focused Python and independent Lean lanes.
Malformed native intervention programs, model mismatch, false normalizations
and fabricated inference labels are rejected.

No new six-hop theorem, no second graph store, fallback, installer/consumer
project or extra CI workflow. Keep Draft until exact-head full Python, Lean,
35+11 theorem trust, BB/PathN and Studio/Playwright pass. M4c.2 repeated
sampling, confidence and external validation are **not** established.
