# R2.M5e.0 — Direct V19 EXACT TELL trust is source-owned

Issue #120; stacked Draft PR #125 on #124. This is a model-local
software semantics qualification, not a universal social influence law.

## Genuine source semantic gap

The original V19 SQLite **recall** path tempers another Agent's
TELL memory by the current observer→source directional trust. The
original direct percept admission used full likelihood whenever an
external TELL was EXACT, regardless of current source trust.
Perceptual EXACT identifies message fidelity, not source credibility.

The first M5e.0 prototype used a six-round V19 Story and a cognitive
checkpoint created at the same sixth round. The existing
observation_floor_round properly rejected all prior percepts as new
evidence. Initial CI failed on a real source-condition error rather
than on the trust-weighted likelihood formula.

The revised admission experiment has a **new native TELL after an
actual cognitive checkpoint**, so no old percept is replayed as new.

## One source owner and versioned opt-in policy

SituatedSocialMemoryPolicy adds
trust_weight_direct_testimony: bool = False.

- **False:** the historical source-policy JSON and hashes remain byte
  identical; existing native V19 M5a/b/c/d golden vectors remain
  unchanged. Default direct TELL uses the original full likelihood.
- **True:** only original source-social state supplies an observer's
  current directional trust to original admit_situated_percepts,
  from the same native simulate_situated_percept_social_cognitive_round.
  No second Bayesian/social memory/graph/execution engine is created.
- Only an actual external EXACT TELL is tempered by effective
  likelihood (1 - trust) + trust × original_likelihood. Bob's own
  INSPECT, SELF and non-TELL evidence remain full weight.
- The typed SituatedBeliefAdmission.source_trust field uses default
  1, omitted from old serializations. Opted-in protocol or configured
  trust changes the social-model identity, not the authoritative world.

## Executable source check

All four arms start from one exact V19 world, perception model,
cognition model, source Story and cognitive checkpoint.

Alice is the declared controller of records-open. Three original
world actions occur before the cognitive checkpoint: her own
INSPECT of the memo, authorized OPEN_PASSAGE, and MOVE records→open.
Bob starts in the open office. The Story checkpoint is round 3.

Alice's INSPECT is available to her original private SQLite recall,
allowing the *original native planner*, not a test-authored action,
to generate a new source-linked TELL in native round 4. Bob's
actual source-qualified private percept must be EXACT. The **next
native round 5** must consume that newly disclosed TELL in the
original decision path, then execute a real native Bob world action.

Declared policy variants:

| Arm | Source Bob→Alice initial trust | Effective direct TELL trust |
| --- | ---: | ---: |
| legacy_unweighted | 0.5 | 1.0 |
| weighted_zero | 0.0 | 0.0 |
| weighted_half | 0.5 | 0.5 |
| weighted_one | 1.0 | 1.0 |

Every arm owns a different declared social-model policy except that
all share exactly the same world/Story/perception/cognitive model.
Each arm uses its own SQLite session and its exact native V19
social/cognitive owner.

The source report retains exact hashes of policy, initial and final
Story, Bob's new EXACT TELL percept, direct admitted source trust,
direct posterior, original round-5 decision, feasible actions and
real executed world event. Other direct evidence, if any, must have
effective trust 1.

Expected model-local semantics: legacy and source trust 1 give the
same TELL posterior. Zero trust leaves the TELL prior unchanged.
Source trust 0.5 yields an intermediate Bayesian update, not a
universal effect-size or population parameter.

This is NOT an isolated trust-only causal study because the four
sources intentionally declare different social policies. Even
different actual decisions would demonstrate only model-local
policy sensitivity, not that powerful people generally cause
specific events.

## Deferred gate: M5e.1 and empirical validation

To prove authentic post-revision trust→new evidence→choice, generate
a **new** native TELL after the original V19 social evidence engine
has actually revised directional trust, then show the original
next native decision consumes this new weighted admission.
A future causal design needs source-matched world/cognition/actor
opportunities and explicit treatment/confounder control. Do not
edit trust fields, create synthetic events, invent private data or
convert static topology to delivery.

This experiment has sample_size_per_arm=1;
trust_only_causal_claim=NOT_ESTABLISHED;
confidence_interval=NOT_ESTIMABLE;
external_validation=NO_EXTERNAL_VALIDATION;
cross_mechanism_ranking=FORBIDDEN.

The existing proof.yml V24 Python and independent Lean source-vector
gates must check reviewed canonical bytes. Keep original full
Python/Lean/Studio/Playwright, cache and singleflight; no extra
workflow, consumer/installer project or second engine.

## First native CI and reviewed source: exact negative action finding

Initial [CI #41094641073](https://github.com/qigao/lean/actions/runs/41094641073)
failed for a **real stale-checkpoint fixture error** (Story history
was not newly admissible). The revised initial
[CI #41095649116](https://github.com/qigao/lean/actions/runs/41095649116)
successfully ran the three original authored actions, generated a
**new source-owned Alice TELL in native round 4**, and reexecuted
Bob's original direct cognition/social admission + actual world
action in **round 5**. It intentionally stopped on the missing
review pin, without a source-model error.

The **9,518-byte** actual native runner JSON was reviewed and
committed to conformance/network_native_direct_tell_trust_v1.json.
Its exact report content hash is:
sha256:0a4e3147819a8555d37258e258c2d0beab5cf29f424e03680d249ad90141411d

The same exact original TELL event identity and source hash are
present in all four social-policy arms. Bob's direct TELL
source weights and real native posterior belief about approved were:
- legacy_unweighted: 1.0 → 0.9
- weighted_zero: 0.0 → 0.5
- weighted_half: 0.5 → 0.6333333333333333
- weighted_one: 1.0 → 0.9

**Actual negative action result:** Bob's native MAP-selected action
in ALL four arms was TELL, and each original V19 world event had
outcome told, success true. Trust-sensitive belief updates reached
the native decision owner, but **action identity did not change**.
Do not report this as a causal trust-to-behavior difference or a
general real-world social-law proof.

The source evidence includes no private message/INSPECT body.
The first-run vector stdout handoff has been removed; Python
and independent Lean now regenerate exactly these reviewed
UTF-8 bytes. Complete full Python, Lean theorem trust,
Studio/Playwright and stacked Draft integration remain required
at the new exact HEAD.
