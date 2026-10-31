# Network research: executable models, not re-proving six degrees

**Status:** canonical project direction, 2026-10-30. **Tracking:** [Issue #96](https://github.com/qigao/lean/issues/96).

## Mathematical boundary

Six degrees of separation, small-world structure, Watts–Strogatz constructions,
basic reachability, path bounds and other standard graph methods are established
literature and reusable mathematical tools. The project does **not** claim novelty
from proving them again or accumulating generated copies of a known upper-bound
theorem. Existing Lean theorems and transitive trust audits remain available as
maintained mathematical infrastructure, not new publication/research milestones.

**Mathematical theorem**, **runtime conformance result**, **simulation
observation**, **empirical predictive claim**, and **causal/mechanistic claim**
must always be reported separately, with assumptions and scope.

## Source-of-truth and implementation boundaries

- **V19 authority:** immutable `SituatedNetworkSnapshot`, with stable
  `model_id/model_hash`, round, story, cognition, social state and source
  content hash. No second canonical graph owner, no mutation of historical
  snapshots, no synthetic edges invented by projection.
- **V24 derived projection:** `MeshSnapshot` chooses relationship,
  direct interaction or realized transmission explicitly. Members may overlap
  across Society identities; physical placement is distinct. Directed
  bridges require an existing selected source relation and endpoint
  membership, never infer symmetry or backwards connectivity.
- **Runtime contracts:** deterministic finite paths, query-scope validation,
  same-source replay, typed rejects, bounded resource ownership and
  explicit provenance. Gateway and Society evidence is checked as a concrete
  topology **input**; testing it is not discovering a six-degree theorem.
- **Proof boundary:** Lean may kernel-check a genuinely new implementation
  invariant or finite conformance obligation. A finite `Fin n` test does not
  prove every Python execution, authenticate an outside process, establish
  asynchronous transmission or guarantee an arbitrary network's diameter.
  Retain trust audits, no `sorry`, `native_decide` or oracle/fallback proof.
- **Existing accepted baseline:** #104 source-to-projection correspondence,
  versioned `conformance/mesh_runtime_vectors_v1.json`, two pinned Lean
  corpora, four negative source/bridge/path/identity RED tests and exact-head
  full CI. Keep these guards while avoiding redundant proof-generation.

## Actual research and engineering priorities

| Phase | Work | Evidence / acceptance |
|---|---|---|
| R0 | Source-backed static projection and API correctness (#101, #103, #104) | Real V19 input, checked-in vectors, relation-isolation and stale-revision rejection; complete Python/Lean/Studio CI |
| R1 | **Executable topology changes**: additions, deletions, rewiring, bridge and Gateway failure, Society split/merge, and temporal relation updates | A versioned `(snapshot, event, seed) -> (next_snapshot, result, provenance)` contract; deterministic replay, transition validation, no lost Agent/edge ownership |
| R2 | **Model comparison**: growth and attachment (including BB), small-world/rewiring and candidate pseudofractal families | Run the actual generators on common initial conditions/seeds; compare directed reachability, components, shortest paths, clustering, centrality, degree, community structure and robustness; no unconditional "six hops" pass gate |
| R3 | **Network–agent coevolution**: belief/exposure/propagation, interventions and schedule sensitivity (#96, #99) | Reuse authoritative ABM and post-incoming exposure semantics; time-series trajectories, finite-model counterexamples, exact/approximate claim taxonomy; no duplicated execution semantics |
| R4 | **Empirical and product validation**: external data, calibration, held-out testing, World Studio (#39, #43, #56) | Frozen dataset/source revisions, splits, metrics and model versions; reproducible evaluation and confidence/uncertainty reporting, with predictive-vs-causal firewall |

R1–R4 are research work to design and implement; they are **not** represented
as already complete by existing Lean proofs or a finite gateway fixture.

## R2.M5: operationalize established mechanisms of key-person influence

Research question: by which **separate mechanisms** can a powerful or
strategically located person alter an event, and when can the resulting
network update feed back into future opportunities? The established
literature provides hypotheses and methods, NOT a universal numeric power
score (Emerson 1962 power/dependence; Borgatti 2006 task-specific key
players; Granovetter 1978 response thresholds). See Issue #120.

- **Resource-control / institutional authority (M5a):** add explicit
  controller Agent identities to selected source V19 passages. Role
  labels do not grant access; the native world executor checks the
  permission before OPEN/CLOSE. Denial is a real failed event and cannot
  mutate the world. A same-initial-model control contrasts controller
  WAIT vs successful OPEN vs co-located unauthorized manager attempt,
  then replays identical MOVE and TELL with the existing perception
  engine. This is a finite executable mechanism, not a population
  estimate.
- **Structural influence (M5b, NOT implemented):** select exactly one
  sourced V24 directed relation, register Agent roster, timestamp and
  diffusion/disruption objective; centrality or gateway location must
  not override passage permissions or imply a transmission occurred.
- **Epistemic influence (M5c, NOT implemented):** use original V19 exact
  perception, admitted testimony, private verification and directional
  trust as the only owners; message reach does not automatically imply
  agreement, truth, trust or authority.
- **Empirical qualification (later):** preregister distinct populations,
  matched controls, thresholds, dependencies, external benchmarks and
  uncertainty. Single deterministic simulations never prove societal
  causality.

Keep the original V19/V24 source identity, baseline conformance bytes,
mathematical theorem trust, cached CI singleflight and Draft gate. Do
not add a second graph store, new formal proof of six hops, fallback
engine, installer/consumer project or universal influence ranking.

## R2.M5b — explicitly separate structural influence tasks

M5b is a **read-only finite observation** of the same V19 source-backed
V24 relation selection, not an additional network engine. Register
one immutable world/story/model/cognitive/social revision, a selected
directed relation, exact Agent roster and Society scope, plus a
task-specific criterion. A broadcaster's static out-reach, an
intermediary's removal sensitivity, an Agent's actual passage resource
control, and a listener's trust are different concepts and must not be
collapsed into a social power scalar.

The first native case uses V19 perception-model INTERACTION edges
Carol→Bob→Dana and an independent controller Alice, then derives
V24 direct-interaction opportunities without inventing edges.
Out-reach recognizes Carol; removal sensitivity recognizes Bob;
Alice's passage permission does not create a social edge. Actual TELL
transmissions remain zero. These are finite descriptive properties,
not a power causal estimate, universal influence ranking or six-hop
theorem. Follow Issue #120 and the separate M5c epistemic gate.

## R2.M5c — epistemic influence is not structural reach or resource control

The previous M5a world-rule gate and M5b source-projected structural
metrics are prerequisites but **neither** establishes opinion change,
trust or enacted adoption. The next single-origin controlled native V19
case compares Alice WAIT versus authorized OPEN, followed by identical
INSPECT → MOVE → TELL → Bob MOVE → private INSPECT stages. The
original V19 source Story, private-memory/SQLite cognition and
social evidence/trust engine must remain the only owners.

Record Bob's DETECTED vs EXACT TELL, actual admitted testimony, his
independent inspection, claim contradiction, the **Bob→Alice trust
judgment** and the correctly oriented **Alice→Bob source-relationship
edge**, while checking the reverse direction separately. Both branches
may contain a legitimate private INSPECT; without a preceding admitted
claim, no source-directed trust change follows.

The 0.7 adoption threshold can classify the tracked belief snapshot,
but no subsequent **action** may be inferred unless an actual source
decision/behavior consumes it. The structural Carol/Bob M5b tasks
remain source-scoped, event count is not reachability, and the
passage controller Alice is not automatically a central or trusted
Agent. One deterministic paired run cannot identify societal causal
effects, and no universal leadership score is defined.

## R2.M5d — post-evidence actual action qualification

A recorded Agent belief above an adoption threshold cannot establish
that any source action occurred, nor prove one trusted person caused
another's choice. The first actual M5d extension reruns the same
original V19/SQLite private cognitive-social engine after the M5c
source evidence update, **then runs one additional native decision
round** and reads the emitted \`SituatedCognitiveDecision\` plus
V19 \`SituatedWorldEvent\`. This is a subsequent, executed source
action (possibly unsuccessful); if actions differ between original
WAIT/OPEN branches, the cause is *not identified as trust alone*
because world location, observed facts, claims and beliefs also differ.

The source report binds model/story/social/decision/event hashes,
native selected/feasible actions, actual world outcome, and any
trust-weighted new Alice TELL recall. Zero such recall means
there is no proven trust-to-memory-to-choice path in that round.
No extra policy executor, synthetic social owner, or automatic
trust from centrality. Keep finite deterministic
\`TRUST_ONLY_CAUSAL_EFFECT_NOT_ESTABLISHED\`,
\`UNATTRIBUTED\`, \`NOT_ESTIMABLE\`, and
\`NO_EXTERNAL_VALIDATION\` until qualified matched source
interventions and population data exist.

## R2.M5e.0 — native exact TELL must not bypass source trust

V19 source memory already tempers **recalled** other-Agent testimony
with the current directed observer→source trust, but direct
\`admit_situated_percepts\` historically applies full likelihood to
new EXACT TELL without consulting that trust. M5d's zero new trust-
weighted recalls consequently cannot demonstrate trust→new evidence
→choice and should not be relabeled as such.

Register an explicit source social-policy switch
\`trust_weight_direct_testimony\` (default **False**, omitted from
serialized legacy model). Under True, only original native source
social memory may supply Bob's trust in Alice to the *same direct
percept/cognitive admission routine*. The original Bayesian likelihood
is tempered by (1-trust)+trust×likelihood. Bob's private INSPECT,
SELF and non-TELL observations remain unchanged; default legacy
bytes and all previously reviewed evidence remain exact.

The first deterministic M5e.0 source protocol qualification keeps
one actual V19 world, Story, cognitive model, perception, same Bob
EXACT TELL and private INSPECT across four source-declared social
policy variants (legacy, trust=0, trust=0.5, trust=1). It records
native admission and actually executed Agent decision/World event;
no alternate updater or hand-edited trust. Changing policy IDs
is not an isolated trust-only real-world intervention. It is
a model-local consistency test. The actual trust-update-then-new-
TELL path and matched post-trust behavior remain a **separate**
M5e.1 gate. No universal score, unsupported causal estimate or
new six-hop theorem.

## R2.M5e.1 — authentic trust history before a *later* TELL

The M5e.0 native protocol checks that freshly received other-Agent
EXACT TELL uses the declared V19 observer→source trust, but its
different social-policy variants cannot isolate real-world trust
causality or prove a different action. M5e.1 requires **the same
underlying V19 model** and a source-history-only intervention:
prior Alice TELL content confirmed vs contradicted by Bob's own
INSPECT. Bob's original V19 social owner must learn the resulting
0.6 vs 0.3 directional trust in native round 8.

A TELL emitted during round 8 is NOT post-revision; only an
additional original-native Alice TELL in round 9 qualifies.
Original Bob's subsequent round-10 decision must consume that
specific new EXACT percept with the learned trust, then actually
execute a native world action. Record distinct Story hashes,
same current physical positions but potentially different prior
private beliefs/claims/observations. Never assert a trust-only
causal result if those epistemic confounders remain different.
Do not invent trust, events, world permissions, graph edges or
a universal elite influence score. Keep full original four-job
source/proof/Studio CI and independent native vector review.

## R2.M5e.2 — native action-choice eligibility rather than forced WAIT

The first M5e.1 source-qualified trust-learning experiment consumed
a new EXACT TELL and performed the actual round-10 decision, but
only WAIT remained feasible. This cannot test whether different
trust-weighted evidence yields different native action choices.

M5e.2 registers a DIFFERENT V19 cognitive model version where Bob's
existing, already-declared TELL has repeatable=True, retaining its
exact action kind/source constraints, all original rewards/likelihoods,
goals, beta and other actions. Both confirmed/contradicted histories
use this same new model; world, perception, social policy and history
intervention remain source-owned. The original V19 native planner
must expose TELL plus WAIT at round 10, report complete native
action values, finite-softmax probabilities, lexical MAP selection
and actual source-world event success/outcome.

This is a new preregistered model-version qualification, **NOT**
a proof that the isolated trust delta caused any behavioral
difference. Historical private belief/claims still differ between
arms. Accept a no-difference selected-action result even if
policies differ; never post-hoc change rewards to force divergence.
Preserve exact old M5e.1 conformance bytes and all upstream CI.

## R2.M5f — complete exploratory native V19 speaking-cost response grid

R2.M5e.2 showed two genuine feasible Agent actions under unchanged
native rewards, with different confirmed/contradicted history policy
probabilities but the SAME MAP TELL and world outcome. Only a
new explicitly **exploratory** model-version sensitivity analysis
may investigate other response regimes.

M5f freezes the complete numeric speaking-cost grid
\`[0.0, 2.0, 3.0, 4.0, 5.0]\` in source and design **before its
first M5f execution**, but acknowledges this range was chosen
after inspecting M5e.2's baseline: it is **not** blind
preregistration or an independent causal test. For EACH grid
cell, derive one new original V19 cognitive model in which
every Bob TELL hypothesis/goal reward is shifted by the same
declared cost, keeping all other rewards/likelihoods/goals/
beta/decisions and world/social/perception owners unchanged.
Both confirmed/contradicted histories run in the same model
version; different costs produce different model hashes.

Publish ALL cells including no-switch cases; record original
V19 action values, full finite-softmax policies, feasible
actions, original MAP selection and native world result, as
well as model/social/Story/trust hashes. The zero-cost cell
MUST reproduce the reviewed original M5e.2 source report.
No reward is changed post-hoc after source outcomes appear.
Keep unmatched historical private memory/belief confounders
explicit, forbid any universal elite score or trust-only
causal inference. Preserve full original CI and native
source-byte conformance without adding a second engine,
workflow or consumer/installer project.

## R2.M5g.0 — source-neutral audit evidence prerequisite

The complete exploratory M5f speaking-cost grid provides a model-local
TELL/WAIT MAP action split in one cost version, but Bob's two historical
branches differ in prior target belief as well as learned trust. This
cannot establish an isolated trust causal mechanism.

M5g.0 declares a **new single V19 cognitive/social model** with a
separate source-audit topic and two new audit symbols. Each audit
symbol has exactly the same observation likelihood under the
target approved/denied hypotheses (0.1 each), while original target
symbol probabilities are multiplied by the common factor 0.8 to keep
every original action/hypothesis likelihood row normalized. Thus
target testimony remains informative, while audit evidence has
likelihood ratio 1 and cannot change the target belief.

Both arms execute the same original native V19 source world
actions except prior Alice audit TELL content (confirmed vs
contradicted). Original private Bob INSPECT verifies the true
audit marker. Native source SQLite/social owner must report
Bob→Alice trust 0.5→0.6 vs 0.5→0.3, with one audit testimony
and one private audit verification each, yet BOTH Bob
original round-8 target prior and posterior stay exactly 0.5.
No target-topic testimony/claim is admitted from this training.
Exact physical locations/passages and single model hashes
remain common, but private history/claim identities are
explicitly still UNMATCHED.

This is a source mechanism prerequisite, NOT the next target
TELL/behavior test. M5g.1 later needs an actual new target TELL
after source trust learning, original native admission and a
decision with ≥2 feasible actions, plus explicit review of
other cognitive-state confounders. Never edit trust fields
or invent evidence to force causality.

## R2.M5g.1 — neutral V19 audit -> new source target TELL -> actual choice

A model-local, original V19/SQLite experiment may show source-social
trust learning while the target belief remains unchanged at the
training checkpoint (M5g.0). This does NOT establish equality of
all cognitive/private decision inputs. M5g.1 versions one common
model to make Alice's and Bob's existing native TELL repeatable
without altering source rewards, observation likelihoods,
goals, beta, physical permissions or social policy.

The native original social memory learns Bob→Alice trust 0.6 vs
0.3 during round 8 through a separate target-neutral audit
topic and Bob's own private INSPECT. A fresh original Agent
TELL about the target must be emitted in round 9, with an
observed native causal source event. Its Bob EXACT percept
must enter the native round-10 decision using the *then-current*
learned trust, with >=2 actual feasible original actions and
the real world success/outcome. Record the full source model,
Story, private observation/memory/claim-status hashes and
entire action policy.

**Crucial temporal confounder:** an intervening native target
TELL or other source event may change Bob's target belief
before the *new* round-9 event is admitted in round 10.
The report must state whether beliefs remain matched at that
point; if not, label NOT_MATCHED_AFTER_NATIVE_INTERMEDIATE_EVENTS.
Prior audit histories and private memory/claim IDs remain
NOT_MATCHED regardless. No claim of isolated trust causality
or universal leader power is permitted from this finite replay.

## Code and PR acceptance rules

1. One implementation per model; prefer existing
   `narrative_dynamics/abm` runtime and reused source data. New graph
   algorithms need justified behavior and testing, not a second private
   state authority.
2. New math work is justified only by a **new statement tied to executable
   model behavior**, such as a transition invariant, probabilistic identity,
   convergence condition or counterexample. Cite established methods instead
   of duplicating their proofs.
3. Every new PR must state the concrete API/behavior, model identity,
   assumptions, real fixtures, numeric or formal evaluation and exact-head
   CI. Distinguish `PROVED`, `DISPROVED` and `UNKNOWN`, simulation-only
   evidence and external scientific claims.
4. Keep the existing cache, superseded-head cancellation and bounded
   concurrency. No skipping original Python, Lean, browser or provenance gates
   to make refactoring GREEN.
5. Keep a narrow stack: **#101** holds V19→V24 projection and folded-in
   theorem trust audit formerly tracked by #100; **#104** holds independent
   runtime-derived conformance. #59 is closed as obsolete docs-only design.
   Neither should be presented as a new proof of six degrees.
6. #60 pseudofractal is a **parked candidate** pending runnable model and
   model-comparison acceptance; #69 Flywire and #1 browser safety are separate
   workstreams, not duplicates to delete as part of this network cleanup.
   Completed #94 / #38 are closed; #39, #43, #56, #96, #99, #102 and #103
   retain distinct unresolved responsibilities.

## Immediate next implementable milestone

Specify and implement the smallest **V19 topology transition/replay API**:
one source-authoritative immutable revision, explicit directed relation
addition/removal, an append-only event record and cross-revision path
invalidation. Demonstrate in a single deterministic test that an old path
becomes stale after edge deletion, the new snapshot remains authoritative,
and derived V24 projections / statistics follow only the updated source.
Then add an actual BB/rewiring comparison rather than an additional
six-degree theorem.

Do not merge the stacked #101/#104 changes before their original full
acceptance conditions and valid target-branch integration checks are met.
