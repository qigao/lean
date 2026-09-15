# Exact Path-Four Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the three actual rational BB histories have common-global-clock belief limits 127/192, 127/192 and 87/128, with positive difference limit 7/384.

**Architecture:** First connect a four-coordinate rational update to the existing propagation operator. Prove invariant-region closure and an exact modal iteration formula, certify real replay prefixes and idle tails, then cast to the reals for limits. Keep generic algebra, analytic limits and concrete replay certificates separately reviewable.

**Tech Stack:** Lean 4.32.0, mathlib v4.32.0, existing `Rat` dynamics, Python standard library for audit tools, existing GitHub proof workflow.

**Spec:** [Approved design](../specs/2026-09-15-bb-path4-convergence-v1-design.md), pinned at `e8d6a68c00969b3d6fc821ea9a10d3aaa6a0ca03`. User confirmed the written design. This plan has not been executed; Lean snippets below are intended contracts/tests, not claimed compilation results.

## Global Constraints

The following sentences are copied verbatim from the approved spec:

- Beliefs/profiles remain `Rat`.
- Use real casts only to state and prove limits of rational belief sequences.
- Exposure counters may keep increasing; only belief coordinates converge.
- Conditional attachment mass 1/8 is not a probability distribution over the three authored timing schedules.
- Uniform mean is not the invariant on this path.
- Keep n=0 valid: natural powers give 0^0=1.
- Existing runtime source, existing fixture corpora, toolchain and dependency pins stay unchanged.
- Do not assume an equality between a standalone toy matrix and the runtime.

Additional mandatory constraints: unchanged axiom allowlist `propext`, `Classical.choice`, `Quot.sound`; no admitted proofs or native-evaluation dependency shortcuts; default heartbeats; each new compile/test phase at most 240 seconds with a 10-second kill grace. Existing workflow event/checkout selection, default library build, earlier 35 foundation/three runtime reports and full Python discovery are preserved. No universal Python-float, arbitrary-graph, heterogeneous-fitness or real-world claim is added.

## File map and dependency order

| File | Responsibility | Tasks |
| --- | --- | --- |
| `NarrativeDynamics/Core/FitnessABMPath4.lean` | Canonical path, actual belief update bridge, rational invariants and closed form | 1–2 |
| `NarrativeDynamics/Core/FitnessABMPath4Convergence.lean` | Real-cast geometric limits for actual rational iteration | 4 |
| `NarrativeDynamics/Tests/FitnessABMPath4.lean` | Actual replay certificates, activation, common-clock limits, required reports | 1–5 |
| `tools/check_fitness_abm_path4.sh` | Source audit, separately bounded builds and mandatory report audit | 5 |
| `.github/workflows/proof.yml` | One additional bounded step, immediately after runtime replay conformance | 5 |
| This plan | Execution evidence and final review record | 6 |

No existing core module is rewritten. `SocialMesh.lean` already imports `Mathlib`; keeping analytic declarations in a separate new file separates responsibilities without claiming complete transitive import isolation.

## Preparation, included in Task 1

- [ ] Read the pinned spec, this plan, repository instructions if present, `NetworkPropagation.lean`, `FitnessABM.lean`, `FitnessABMReplay.lean`, and existing runtime fixture/gate patterns.
- [ ] Start an isolated implementation branch from the committed plan head. Preserve #77/#78 and their reviewed heads; do not merge them without authorization. Record base/head/tree before edits.
- [ ] Determine whether a working pinned `lake` exists. Current authoring environment has none. If unavailable, publish ordinary task commits and use the existing PR proof job. In Task 1, add the final new bounded CI step to build `NarrativeDynamics.Core.FitnessABMPath4` before executing the current test file; Task 4 adds the analytic-module build, and Task 5 replaces these provisional phase commands with the final gate script. Every provisional phase uses the same 240-second bound. Retain the same event and exact-head checkout semantics. No separate temporary workflow or proof-skip switch.
- [ ] Record the baseline: previous #77 exact head `96901a09bd2438a1829a427704b57e7d52b182a8` passed Python 1,723 tests (one existing skip), Lean and World Studio. Verify the implementation base differs only by the approved design/plan before relying on that baseline.

## Task 1: Actual propagation bridge and invariant region

**Files:** create `Core/FitnessABMPath4.lean`; create `Tests/FitnessABMPath4.lean`; add the one CI step if remote execution is required.

**Interfaces:** namespace `NarrativeDynamics.FitnessABMPath4` (abbreviated P4 below). Import `NetworkPropagation` and use its `Population` and `propagate`.

```lean
abbrev Beliefs := Fin 4 → Rat
def pathAdj (i j : Fin 4) : Prop :=
  i.val + 1 = j.val ∨ j.val + 1 = i.val
instance : DecidableRel pathAdj := fun _ _ => inferInstance
def population (x : Beliefs) (e : Fin 4 → Nat) :
    NetworkPropagation.Population 4 :=
  ⟨fun _ => ⟨1/2, 1/2⟩, fun i => ⟨x i, e i⟩⟩
def project (p : NetworkPropagation.Population 4) : Beliefs :=
  fun i => (p.agents i).belief
def allBroadcast (x : Beliefs) : Prop := ∀ i, 1/2 ≤ x i ∧ x i ≤ 1
def linearStep (x : Beliefs) : Beliefs :=
  ![(x 0+x 1)/2, x 0/4+x 1/2+x 2/4,
    x 1/4+x 2/2+x 3/4, (x 2+x 3)/2]
def beliefStep (x : Beliefs) : Beliefs :=
  project (NetworkPropagation.propagate pathAdj (population x (fun _ => 0)))
def trajectory (x : Beliefs) (n : Nat) : Beliefs := beliefStep^[n] x
```

Required theorems:

- `propagate_independent_exposures (x : Beliefs) (e : Fin 4 → Nat)`:
  `project (propagate pathAdj (population x e)) = beliefStep x`.
- `propagate_eq_linear (x : Beliefs) (e : Fin 4 → Nat) (hx : allBroadcast x)`:
  `project (propagate pathAdj (population x e)) = linearStep x`.
- `allBroadcast_step (x) (hx : allBroadcast x) : allBroadcast (beliefStep x)`.
- `allBroadcast_iterate (x) (hx : allBroadcast x) (n) : allBroadcast (trajectory x n)`.

- [ ] RED: add this consuming test before the module/theorems exist:

```lean
import NarrativeDynamics.Core.FitnessABMPath4
open NarrativeDynamics.NetworkPropagation NarrativeDynamics.FitnessABMPath4
example (x : Beliefs) (e : Fin 4 → Nat) (hx : allBroadcast x) :
    project (propagate pathAdj (population x e)) = linearStep x :=
  propagate_eq_linear x e hx
example : beliefStep ![3/4, 11/16, 5/8, 1/4] =
    ![23/32, 11/16, 21/32, 7/16] := by
  decide_cbv
```

- [ ] Run `timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPath4.lean`; record missing import/declaration RED. A missing toolchain is not RED.
- [ ] Implement the definitions. Prove exposure independence by unfolding `incoming`, `broadcasting` and `nextAgent`. Enumerate each `Fin 4` coordinate and its actual incoming set. Under `hx`, the sets are `{1}`, `{0,2}`, `{1,3}`, `{2}`. Use `fin_cases`, finite-set extensionality and rational normalization; do not infer adjacency from a hand-written matrix alone.
- [ ] Prove bounds on each convex combination using `linarith`, then induct on iteration. The literal test deliberately includes a nonbroadcasting fourth node and catches premature use of the all-broadcast formula.
- [ ] GREEN: build the new core module before testing the consumer. Verify the example and arbitrary-exposure bridge under 240-second bounds. Commit as `feat(lean): connect path-four updates to propagation`; review this unit before Task 2.

## Task 2: Weighted mean and exact iteration formula

**Files:** extend the algebra core and consuming test file.

**Consumes:** all Task 1 interfaces. **Produces:**

```lean
def mean (x : Beliefs) : Rat := (x 0+2*x 1+2*x 2+x 3)/6
def coeffA (x : Beliefs) : Rat := (x 0+x 1-x 2-x 3)/3
def coeffB (x : Beliefs) : Rat := (x 0-x 1-x 2+x 3)/3
def coeffC (x : Beliefs) : Rat := (x 0-2*x 1+2*x 2-x 3)/6
def modeV : Beliefs := ![1,1/2,-1/2,-1]
def modeW : Beliefs := ![1,-1/2,-1/2,1]
def modeZ : Beliefs := ![1,-1,1,-1]
def closedForm (x : Beliefs) (n : Nat) : Beliefs := fun i =>
  mean x + coeffA x*(3/4)^n*modeV i +
    coeffB x*(1/4)^n*modeW i + coeffC x*(0:Rat)^n*modeZ i
```

`mean_step (x) (hx : allBroadcast x) : mean (beliefStep x) = mean x`.
`iterate_closedForm (x) (hx : allBroadcast x) (n) : trajectory x n = closedForm x n`.
Internal algebra lemmas: `decompose (x) : closedForm x 0 = x` and
`linear_closedForm_succ (x) (n) : linearStep (closedForm x n) = closedForm x (n+1)`.

- [ ] RED: consuming tests reference the missing invariant/formula:

```lean
example (x : Beliefs) (hx : allBroadcast x) (n : Nat) :
    trajectory x n = closedForm x n := iterate_closedForm x hx n
example (x : Beliefs) (hx : allBroadcast x) :
    mean (beliefStep x) = mean x := mean_step x hx
example : closedForm ![45/64,44/64,43/64,35/64] 0 =
    ![45/64,44/64,43/64,35/64] := by decide_cbv
example : mean ![45/64,44/64,43/64,35/64] = 127/192 := by decide_cbv
```

- [ ] Run the bounded consumer, recording unknown-declaration RED.
- [ ] Prove mean preservation by coordinate expansion and `ring`. Prove decomposition and each mode identity with `funext`, `fin_cases`, `norm_num`/`ring`. Derive the successor formula with `pow_succ`. Induct for the actual trajectory, using the Task 1 region invariant to apply its bridge on each step. Keep `0^0` in the base case.
- [ ] GREEN: core build and consumer tests pass. Commit `feat(lean): prove path-four invariant and closed form`; review before Task 3.

## Task 3: Certify actual BB histories and idle-tail semantics

**Files:** extend `Tests/FitnessABMPath4.lean`. Import existing `FitnessABMReplay`; do not change it or reuse private fixture declarations by weakening visibility.

**Interfaces:** namespace `NarrativeDynamics.Tests.FitnessABMPath4` (abbreviated F4). Define `History` with constructors `bbii`, `bibi`, `iibb`, deriving `DecidableEq`. Exact input definitions:

```lean
inductive History where
  | bbii | bibi | iibb
  deriving DecidableEq
def rawSeed : NarrativeDynamics.FitnessAttachment.RawSeed :=
  ⟨2, #[1,1], #[(0,1)]⟩
def rawAgents : Array NarrativeDynamics.FitnessABM.RawAgent :=
  #[⟨1/2,1/2,1,0⟩, ⟨1/2,1/2,0,0⟩]
def rawBirth (target : Nat) : NarrativeDynamics.FitnessABM.RawBirthInput :=
  ⟨⟨1, #[target]⟩, 1/2, 1/2, 0⟩
def ticks : History → List NarrativeDynamics.FitnessABM.RawTick
  | .bbii => [some (rawBirth 1), some (rawBirth 2), none, none]
  | .bibi => [some (rawBirth 1), none, some (rawBirth 2), none]
  | .iibb => [none, none, some (rawBirth 1), some (rawBirth 2)]
def rawTail (h : History) (k : Nat) :=
  NarrativeDynamics.FitnessABM.replay rawSeed 1 rawAgents
    (ticks h ++ List.replicate k none)
```

Create `baseState : History → FitnessABM.JointState 4` using the canonical unit-fitness `SimpleGraph (Fin 4)` whose adjacency is P4.pathAdj; give its actual connectedness/positive-fitness proof. Establish adjacency equivalence for every pair by finite cases. Connectivity can use explicit paths along adjacent numeric vertices, with sixteen endpoint cases; no default fabricated graph is permitted.

Profiles are P4.population profiles. Baseline belief/exposure vectors are:

| History | Beliefs at tick 4 | Exposures at tick 4 |
| --- | --- | --- |
| bbii, bibi | [3/4,11/16,5/8,1/4] | [3,5,3,1] |
| iibb | [3/4,3/4,9/16,0] | [3,4,2,0] |

Define `tailState (h) (k) : JointState 4 := FitnessABM.advance^[k] (baseState h)` and `tailBelief (h) (k) : P4.Beliefs := P4.project (tailState h k).population`.
Produce `replay_baseline (h) : rawTail h 0 = .ok ⟨⟨4,4,baseState h⟩,1/8⟩` and
`raw_tail_bridge (h) (k) : rawTail h k = .ok ⟨⟨4,4+k,tailState h k⟩,1/8⟩`.

- [ ] RED: require actual replay equality, not just equality of two authored vectors:

```lean
example (h : History) (k : Nat) :
    rawTail h k = .ok ⟨⟨4,4+k,tailState h k⟩,1/8⟩ := raw_tail_bridge h k
example : tailBelief .bbii 2 = ![45/64,44/64,43/64,35/64] := activation_bbii.1
```

- [ ] Run bounded tests and capture the missing declarations. Build short checked-birth stages via `checkedBirth_spec`, `parseSeed_complete`, `parseAgents_complete` and `runInputs`; follow existing fixture proof structure. Prove equality of states by graph/profile/agent coordinates and proof irrelevance, not a giant opaque evaluation of the complete proof-bearing result. Derive `raw_tail_bridge` by induction through the actual idle `runInputs` branch and its clock increment.
- [ ] Add `activation_bbii`, `activation_bibi`, `activation_iibb`: each is a conjunction of the exact activation vector equality and `P4.allBroadcast` at tail lengths 2, 2, 3 respectively. Prove baseline and all preactivation tail states are not all-broadcast. In particular the last coordinate at k=1 for bbii/bibi is 7/16; at k=2 for iibb it is 15/32.
- [ ] Add `tail_from_activation (h) (n)`:
  `tailBelief h (activationOffset h+n) = P4.trajectory (tailBelief h (activationOffset h)) n`, with `activationOffset .bbii/.bibi = 2`, `.iibb = 3`. Use graph/profile preservation of `advance` and exposure independence; do not reset the actual state or exposure counters.
- [ ] Add `equal_control (k) : tailBelief .bbii k = tailBelief .bibi k`, by equality of their actual four-round base states. No claim of equality of full original replay histories.
- [ ] GREEN: verify every activation, predecessor, raw-tail and equality-control theorem. Commit `feat(lean): certify BB timing activation and idle tails`; review before analytic work.

## Task 4: Real-cast limits and positive common-clock separation

**Files:** create `Core/FitnessABMPath4Convergence.lean`; extend concrete test file to import it.

**Consumes:** `iterate_closedForm`, `mean`, activation certificates and `tail_from_activation`.
**Produces:** in P4, `trajectory_tendsto (x : Beliefs) (hx : allBroadcast x) (i : Fin 4)` with type

```lean
Filter.Tendsto (fun n : Nat => (trajectory x n i : Real))
  Filter.atTop (nhds (mean x : Real))
```

In F4 produce `bbii_tendsto`, `bibi_tendsto`, `iibb_tendsto`, each taking `(i : Fin 4)` and asserting the real cast of `tailBelief history k i` tends to 127/192, 127/192, 87/128 respectively as k tends to infinity. These are all global tick 4+k. Also produce `separation_limit (i : Fin 4)`:

```lean
Filter.Tendsto
  (fun k : Nat => (tailBelief .iibb k i : Real) - (tailBelief .bbii k i : Real))
  Filter.atTop (nhds (7/384 : Real)) ∧ (0 : Real) < 7/384
```

- [ ] RED: add consuming `example` declarations with these exact types and references to their missing theorem names. Run the bounded test file; record unknown-declaration failure.
- [ ] Verify the pinned mathlib lemma by `#check tendsto_pow_atTop_nhds_zero_of_lt_one`. Its declaration was inspected at [mathlib v4.32.0](https://github.com/leanprover-community/mathlib4/blob/v4.32.0/Mathlib/Analysis/SpecificLimits/Basic.lean): it takes `0 ≤ r` and `r < 1` and proves `Tendsto (fun n : Nat => r^n) atTop (nhds 0)`. The exact core geometric proof is:

```lean
open Filter Topology
example : Tendsto (fun n : Nat => (3/4 : Real)^n) atTop (nhds 0) :=
  tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
example : Tendsto (fun n : Nat => (1/4 : Real)^n) atTop (nhds 0) :=
  tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
example : Tendsto (fun n : Nat => (0 : Real)^n) atTop (nhds 0) :=
  tendsto_pow_atTop_nhds_zero_of_lt_one (by norm_num) (by norm_num)
```

- [ ] Cast the rational closed form with `push_cast`; combine constant multiples and sums using `Tendsto` operations and normalize the limit. This retains a valid n=0 formula without a special unproved deletion of the zero mode.
- [ ] Apply the general result to the three certified activation states. Compute their means with `norm_num`. Remove the finite offsets using the pinned `tendsto_add_atTop_iff_nat` shift equivalence (check orientation with `#check` before writing `.mp`/`.mpr`); it must establish the sequence k↦tailBelief h k, not only k↦tailBelief h (offset+k). Reassociate Nat addition explicitly.
- [ ] Subtract the iibb and bbii limit theorems at the same k; `norm_num` proves the difference and positivity. Verify bibi through the equality control too.
- [ ] GREEN: separately build the analytic module and run the concrete test suite, each bounded by 240 seconds. Commit `feat(lean): prove distinct BB history limits`; independent review must inspect actual replay dependence and time offsets.

## Task 5: Permanent fail-closed proof gate and negative checks

**Files:** create `tools/check_fitness_abm_path4.sh`; finalize the one new workflow step; append actual `#print axioms` reports to the test file.

**Required declaration suffixes:** P4.`propagate_independent_exposures`, `propagate_eq_linear`, `allBroadcast_iterate`, `mean_step`, `iterate_closedForm`, `trajectory_tendsto`; F4.`replay_baseline`, `raw_tail_bridge`, `activation_bbii`, `activation_bibi`, `activation_iibb`, `bbii_tendsto`, `bibi_tendsto`, `iibb_tendsto`, `equal_control`, `separation_limit`. P4/F4 expand to the namespaces defined above. Exactly these sixteen named reports are mandatory; additional reports may be retained but cannot substitute for them.

- [ ] Add the sixteen `#print axioms` commands using the exact expanded names listed above, and capture their actual compiled output. RED: require an additional nonexistent report with the existing auditor; verify failure specifically names that missing report. There must be real emitted reports, not fabricated log text. Use the following command sequence and expect its last command to exit nonzero:

```bash
set -euo pipefail
path4_probe_log="$(mktemp)"
trap 'rm -f "$path4_probe_log"' EXIT
timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPath4.lean \
  2>&1 | tee "$path4_probe_log"
python3 tools/audit_fitness_trust.py log "$path4_probe_log" \
  --require NarrativeDynamics.Tests.FitnessABMPath4.nonexistentProbe
```
- [ ] Implement the script structure:

```bash
#!/usr/bin/env bash
set -euo pipefail
path4_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$path4_root"
path4_log="$(mktemp)"
trap 'rm -f "$path4_log"' EXIT
python3 tools/audit_fitness_trust.py source \
  NarrativeDynamics/Core/FitnessABMPath4.lean \
  NarrativeDynamics/Core/FitnessABMPath4Convergence.lean \
  NarrativeDynamics/Tests/FitnessABMPath4.lean
for path4_module in NarrativeDynamics.Core.FitnessABMPath4 \
    NarrativeDynamics.Core.FitnessABMPath4Convergence; do
  /usr/bin/time -f "$path4_module elapsed=%e s peak_rss=%M KiB" \
    timeout --kill-after=10s 240s lake build "$path4_module"
done
/usr/bin/time -f 'FitnessABMPath4 tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPath4.lean \
  2>&1 | tee "$path4_log"
path4_required=()
for path4_name in propagate_independent_exposures propagate_eq_linear \
    allBroadcast_iterate mean_step iterate_closedForm trajectory_tendsto; do
  path4_required+=(--require "NarrativeDynamics.FitnessABMPath4.$path4_name")
done
for path4_name in replay_baseline raw_tail_bridge activation_bbii activation_bibi \
    activation_iibb bbii_tendsto bibi_tendsto iibb_tendsto equal_control separation_limit; do
  path4_required+=(--require "NarrativeDynamics.Tests.FitnessABMPath4.$path4_name")
done
python3 tools/audit_fitness_trust.py log "$path4_log" "${path4_required[@]}"
```

- [ ] Replace provisional phase commands with this script in the same workflow step:

```yaml
      - name: BB path-four convergence
        timeout-minutes: 15
        run: bash tools/check_fitness_abm_path4.sh
```

- [ ] Negative check A: change only the claimed separation target `7/384` to `1/384`, preserving the proof body; the Lean consumer must fail. Capture exact changed diff, exit code and log, restore the tracked file, and rerun the full gate. If the proposition occurs in several locations, change the theorem target only so the failure is meaningful.
- [ ] Negative check B: remove only the `#print axioms ...separation_limit` command. The theorem still compiles but the final gate must fail for that missing mandatory report. Restore and rerun all phases.
- [ ] A wall timeout, missing toolchain or unrelated syntax error is not a successful negative semantic check. On remote-only execution preserve separate mutation/restoration commits and CI runs; final head must restore the intended tree. Never merge mutation commits as the final tip.
- [ ] GREEN: real emitted reports contain only the unchanged three-axiom allowlist; all sixteen names are found. Commit `ci: gate exact path-four convergence proofs`; review gate behavior and restoration evidence.

## Task 6: Final verification, review and publication record

**Files:** update only this plan's execution record when evidence exists; PR description holds CI results for its final head to avoid endless documentation-only commits.

- [ ] Run `bash -n tools/check_fitness_abm_path4.sh` and `git diff --check`. Run the new gate, existing `lake build`, existing foundation/runtime/fitness gates and unchanged `python -m unittest discover -s tests -v` through applicable CI. World Studio remains required by its current workflow selection.
- [ ] Check actual Lean/Python checkout SHA, World Studio merge parents/tree, and all new theorem reports. Record individual phase time/RSS and final full-suite counts. Do not transfer a successful run from a different head without tree/provenance qualification.
- [ ] Independent review: assess the real replay bridge, inclusive thresholds, arbitrary-exposure handling, zero-mode n=0 case, activation offsets, positive limit difference, and absence of Python-float equivalence claims. Fix blockers in separate commits and repeat affected checks/full required CI on the new final head.
- [ ] Read back the published files and compare the remote tree to the reviewed/tested tree. Preserve branch history and the #77/#78 dependency chain; prepare the implementation PR for review without automatically merging it.
- [ ] Report only established results. If exact proofs succeed, distinguish the new rational-model theorem from the earlier finite Python experiments and their limited conformance coverage.

## Plan self-review and handoff

Coverage: Task 1 covers the actual operator/region and arbitrary exposures; Task 2 covers the invariant and all-n algebra; Task 3 covers checked real histories, all preactivation frames and tails; Task 4 covers real limits on a common clock; Task 5 covers bounded trust enforcement/mutations; Task 6 covers final CI/provenance/review. All new symbol producers precede consumers, except `History` which Task 3 defines before its input declarations. Existing external interfaces were inspected; only the exact mathlib shift invocation and proof tactics remain execution-level compiler work, not claimed checked code.

Execution status (2026-09-15): Tasks 1–5 are implemented at local head `e63a2d33d3b900fd28ee2e81b8dc0cd5b254765e` (tree `87ea079b3b475128caa4c432cecd1ec74206845a`) and each task's independent spec and quality review approved the unit with no findings. Those reviews covered the actual replay bridge, inclusive threshold behavior, arbitrary exposure counters, the n=0 zero mode, activation offsets 2/2/3, the positive limit difference, and the absence of a Python-float equivalence claim. The checklists in this document remain the historical implementation instructions rather than a claim that every final Task 6 check has run.

Bounded local record: after both required mutations were restored, `bash tools/check_fitness_abm_path4.sh` passed its source audit, two core builds, consuming Lean test, and final report audit. The final phases measured 4.83 s / 943104 KiB, 4.09 s / 942096 KiB, and 20.29 s / 7640476 KiB respectively. The test emitted 28 genuine axiom reports; the gate required each of the exact 16 mandated reports once, and every accepted report used only `propext`, `Classical.choice`, and `Quot.sound`. Changing the separation target from `7/384` to `1/384` failed in the unchanged proof with an unsolved goal (exit 1); after restoration the full gate passed. Removing only the required `separation_limit` report compiled the test and then failed because the auditor named that missing report (exit 1); the restored source hash matched and the final full gate passed. `bash -n tools/check_fitness_abm_path4.sh` and `git diff --check` also passed.

Scoped result: for the checked rational model and its three actual replay histories on the common global clock `4+k`, the belief coordinates tend to `127/192`, `127/192`, and `87/128`; the IIBB-minus-BBII limit is the positive rational value `7/384`. This is an exact theorem about that rational model, not a universal Python-float, arbitrary-graph, heterogeneous-fitness, or real-world result. Task 6 is still open: final exact-head required CI (including World Studio and the unchanged full-suite counts), checkout and merge-parent/tree provenance, whole-branch review, remote file read-back/tree comparison, and the final publication record must be completed on PR #79's final head and recorded in its description.
