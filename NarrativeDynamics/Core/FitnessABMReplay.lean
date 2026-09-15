import NarrativeDynamics.Core.FitnessABM
import NarrativeDynamics.Core.FitnessReplay
import NarrativeDynamics.Core.FitnessABMIdleTail

/-!
# Checked finite joint replay

Inputs are fixed parameters. Only the existing ordered BB births contribute
probability. Validation preserves the first error and exposes no partial run.
-/

namespace NarrativeDynamics.FitnessABM

open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessAttachment.Internal

structure RawAgent where
  receptivity : Rat
  threshold : Rat
  belief : Rat
  exposures : Nat
  deriving DecidableEq, Repr

structure RawBirthInput where
  birth : FitnessAttachment.RawBirth
  receptivity : Rat
  threshold : Rat
  belief : Rat

abbrev RawTick := Option RawBirthInput

inductive AgentField where
  | receptivity | threshold | belief
  deriving DecidableEq, Repr

inductive BirthError where
  | network (cause : FitnessAttachment.Internal.Error)
  | agent (field : AgentField)
  deriving DecidableEq, Repr

inductive JointError where
  | seedNetwork (cause : FitnessAttachment.Internal.Error)
  | initialM
  | seedAgentCount (expected actual : Nat)
  | seedAgent (index : Nat) (field : AgentField)
  | tickNetwork (tickIndex birthIndex : Nat) (cause : FitnessAttachment.Internal.Error)
  | tickAgent (tickIndex birthIndex : Nat) (field : AgentField)
  deriving DecidableEq, Repr

def RawAgent.Valid (raw : RawAgent) : Prop :=
  (0 ≤ raw.receptivity ∧ raw.receptivity ≤ 1) ∧
  (0 ≤ raw.threshold ∧ raw.threshold ≤ 1) ∧
  (0 ≤ raw.belief ∧ raw.belief ≤ 1)

def RawBirthInput.AgentValid (raw : RawBirthInput) : Prop :=
  (0 ≤ raw.receptivity ∧ raw.receptivity ≤ 1) ∧
  (0 ≤ raw.threshold ∧ raw.threshold ≤ 1) ∧
  (0 ≤ raw.belief ∧ raw.belief ≤ 1)

/-- Scalar checks are ordered and preserve every supplied value. -/
def parseAgent (raw : RawAgent) :
    Except AgentField {pair : AgentProfile × AgentState // pair.1.Valid ∧ pair.2.Valid} :=
  if hr : 0 ≤ raw.receptivity ∧ raw.receptivity ≤ 1 then
    if ht : 0 ≤ raw.threshold ∧ raw.threshold ≤ 1 then
      if hb : 0 ≤ raw.belief ∧ raw.belief ≤ 1 then
        .ok ⟨(⟨raw.receptivity, raw.threshold⟩, ⟨raw.belief, raw.exposures⟩),
          ⟨hr.1, hr.2, ht.1, ht.2⟩, hb⟩
      else .error .belief
    else .error .threshold
  else .error .receptivity

theorem parseAgent_sound (raw : RawAgent)
    (pair : {pair : AgentProfile × AgentState // pair.1.Valid ∧ pair.2.Valid})
    (h : parseAgent raw = .ok pair) :
    raw.Valid ∧ pair.val = (⟨raw.receptivity, raw.threshold⟩,
      ⟨raw.belief, raw.exposures⟩) := by
  unfold parseAgent at h
  split at h <;> try contradiction
  rename_i hr
  split at h <;> try contradiction
  rename_i ht
  split at h <;> try contradiction
  rename_i hb
  cases h
  exact ⟨⟨hr, ht, hb⟩, rfl⟩

theorem parseAgent_complete (raw : RawAgent) (h : raw.Valid) :
    ∃ pair, parseAgent raw = .ok pair := by
  simp only [parseAgent, dif_pos h.1, dif_pos h.2.1, dif_pos h.2.2]
  exact ⟨_, rfl⟩

/-- Structural left-to-right validation; later records cannot replace a failure. -/
private def checkRoster (index : Nat) (raw : List RawAgent) :
    Except JointError {_u : Unit // ∀ a ∈ raw, a.Valid} :=
  match raw with
  | [] => .ok ⟨(), by simp⟩
  | a :: rest =>
    match ha : parseAgent a with
    | .error e => .error (.seedAgent index e)
    | .ok pair =>
      match checkRoster (index + 1) rest with
      | .error e => .error e
      | .ok tail => .ok ⟨(), by
          intro b hb
          rcases List.mem_cons.mp hb with hab | hb
          · rw [hab]
            exact (parseAgent_sound a pair ha).1
          · exact tail.property b hb⟩
termination_by structural raw

private theorem checkRoster_complete (index : Nat) (raw : List RawAgent)
    (h : ∀ a ∈ raw, a.Valid) : ∃ checked, checkRoster index raw = .ok checked := by
  induction raw generalizing index with
  | nil => exact ⟨_, rfl⟩
  | cons a rest ih =>
    obtain ⟨pair, hp⟩ := parseAgent_complete a (h a (by simp))
    obtain ⟨tail, ht⟩ := ih (index + 1) (fun b hb => h b (by simp [hb]))
    unfold checkRoster
    split
    · rename_i e he
      have impossible : Except.ok pair = Except.error e := hp.symm.trans he
      cases impossible
    · simp only [ht]
      exact ⟨⟨(), h⟩, True.intro⟩

/-- Exact roster size is checked before examining any scalar field. -/
def parseAgents (n : Nat) (raw : Array RawAgent) :
    Except JointError {p : Population n // p.Valid} :=
  if hs : raw.size = n then
    match checkRoster 0 raw.toList with
    | .error e => .error e
    | .ok checked =>
      let record (i : Fin n) : RawAgent := raw[i.val]'(by have := i.isLt; omega)
      have valid (i : Fin n) : (record i).Valid :=
        checked.property _ (Array.getElem_mem_toList (by have := i.isLt; omega))
      .ok ⟨⟨fun i => ⟨(record i).receptivity, (record i).threshold⟩,
        fun i => ⟨(record i).belief, (record i).exposures⟩⟩,
        ⟨fun i => ⟨(valid i).1.1, (valid i).1.2,
          (valid i).2.1.1, (valid i).2.1.2⟩, fun i => (valid i).2.2⟩⟩
  else .error (.seedAgentCount n raw.size)

theorem parseAgents_sound (n : Nat) (raw : Array RawAgent)
    (p : {p : Population n // p.Valid}) (h : parseAgents n raw = .ok p) :
    raw.size = n ∧ (∀ a ∈ raw.toList, a.Valid) ∧
      ∀ (i : Fin n) (hi : i.val < raw.size),
        p.val.profiles i = ⟨raw[i.val].receptivity, raw[i.val].threshold⟩ ∧
        p.val.agents i = ⟨raw[i.val].belief, raw[i.val].exposures⟩ := by
  unfold parseAgents at h
  split at h <;> try contradiction
  rename_i hs
  cases hc : checkRoster 0 raw.toList with
  | error e => simp [hc] at h
  | ok checked =>
    simp only [hc] at h
    cases h
    exact ⟨hs, checked.property, fun _ _ => ⟨rfl, rfl⟩⟩

theorem parseAgents_complete (n : Nat) (raw : Array RawAgent)
    (hs : raw.size = n) (h : ∀ a ∈ raw.toList, a.Valid) :
    ∃ p, parseAgents n raw = .ok p := by
  obtain ⟨checked, hc⟩ := checkRoster_complete 0 raw.toList h
  simp only [parseAgents, dif_pos hs, hc]
  exact ⟨_, rfl⟩

private def newbornRaw (raw : RawBirthInput) : RawAgent :=
  ⟨raw.receptivity, raw.threshold, raw.belief, 0⟩

private def newborn
    (pair : {pair : AgentProfile × AgentState // pair.1.Valid ∧ pair.2.Valid}) : NewAgent :=
  ⟨pair.val.1, pair.val.2.belief, pair.property.1, pair.property.2⟩

/-- The BB step runs first and its exact network and mass are used once. -/
def checkedBirth {n : Nat} (s : JointState n) (m : Nat) (raw : RawBirthInput) :
    Except BirthError (JointState (n + 1) × Rat) :=
  match step s.network m raw.birth with
  | .error e => .error (.network e)
  | .ok next =>
    match parseAgent (newbornRaw raw) with
    | .error e => .error (.agent e)
    | .ok pair => .ok
      (⟨next.1, extendPopulation s.population (newborn pair),
        extendPopulation_valid s.population s.populationValid (newborn pair)⟩, next.2)

/-- Success is precisely the existing typed grow, using checked input data. -/
theorem checkedBirth_spec {n m : Nat} (s : JointState n) (raw : RawBirthInput)
    (out : JointState (n + 1) × Rat) :
    checkedBirth s m raw = .ok out ↔
      ∃ (v : ValidatedBirth n m)
        (pair : {pair : AgentProfile × AgentState // pair.1.Valid ∧ pair.2.Valid}),
        validateBirth s.network m raw.birth = .ok v ∧
        parseAgent ⟨raw.receptivity, raw.threshold, raw.belief, 0⟩ = .ok pair ∧
        out = (grow s v.targets v.positive
          ⟨v.fitness, ⟨pair.val.1, pair.val.2.belief,
            pair.property.1, pair.property.2⟩⟩, orderedMass s.network v.targets) := by
  constructor
  · intro h
    cases hs : step s.network m raw.birth with
    | error e => simp [checkedBirth, hs] at h
    | ok next =>
      cases hp : parseAgent (newbornRaw raw) with
      | error e => simp [checkedBirth, hs, hp] at h
      | ok pair =>
        obtain ⟨v, hv, rfl⟩ := (step_spec s.network raw.birth next).mp hs
        refine ⟨v, pair, hv, hp, ?_⟩
        simpa only [checkedBirth, hs, hp, Except.ok.injEq, grow, newborn] using h.symm
  · rintro ⟨v, pair, hv, hp, rfl⟩
    have hs := (step_spec s.network raw.birth _).mpr ⟨v, hv, rfl⟩
    simp only [checkedBirth, hs, newbornRaw, hp, grow, newborn]

private theorem checkedBirth_projection {n m : Nat} (s : JointState n)
    (raw : RawBirthInput) (out : JointState (n + 1) × Rat)
    (h : checkedBirth s m raw = .ok out) :
    step s.network m raw.birth = .ok (out.1.network, out.2) := by
  obtain ⟨v, pair, hv, _, rfl⟩ := (checkedBirth_spec s raw out).mp h
  exact (step_spec s.network raw.birth _).mpr ⟨v, hv, rfl⟩

private theorem checkedBirth_valid_iff {n m : Nat} (s : JointState n)
    (raw : RawBirthInput) :
    (∃ out, checkedBirth s m raw = .ok out) ↔ raw.birth.Valid n m ∧ raw.AgentValid := by
  constructor
  · rintro ⟨out, h⟩
    obtain ⟨v, pair, hv, hp, _⟩ := (checkedBirth_spec s raw out).mp h
    exact ⟨validateBirth_sound s.network raw.birth v hv, (parseAgent_sound _ pair hp).1⟩
  · rintro ⟨hb, ha⟩
    obtain ⟨v, hv⟩ := validateBirth_complete s.network raw.birth hb
    obtain ⟨pair, hp⟩ := parseAgent_complete (newbornRaw raw) ha
    exact ⟨_, (checkedBirth_spec s raw _).mpr ⟨v, pair, hv, hp, rfl⟩⟩

/-- Both positions are zero based; idle ticks consume no birth position or mass. -/
def runInputs (m tickIndex birthIndex : Nat) (s : RunState) (ticks : List RawTick) :
    Except JointError Result :=
  match ticks with
  | [] => .ok ⟨s, 1⟩
  | none :: rest =>
    runInputs m (tickIndex + 1) birthIndex
      ⟨s.nodeCount, s.roundIndex + 1, advance s.state⟩ rest
  | some raw :: rest =>
    match checkedBirth s.state m raw with
    | .error (.network e) => .error (.tickNetwork tickIndex birthIndex e)
    | .error (.agent e) => .error (.tickAgent tickIndex birthIndex e)
    | .ok next =>
      match runInputs m (tickIndex + 1) (birthIndex + 1)
          ⟨s.nodeCount + 1, s.roundIndex + 1, advance next.1⟩ rest with
      | .error e => .error e
      | .ok tail => .ok ⟨tail.final, next.2 * tail.probability⟩
termination_by structural ticks

theorem runInputs_replicate_idle
    (m tickIndex birthIndex k : Nat) (s : RunState) :
    runInputs m tickIndex birthIndex s (List.replicate k none) =
      .ok ⟨runIdleTrajectory s k, 1⟩ := by
  induction k generalizing tickIndex s with
  | zero =>
      simp [runInputs, runIdleTrajectory]
  | succ k ih =>
      simpa [List.replicate_succ, runInputs, idleRunStep, runIdleTrajectory_eq,
        Function.iterate_succ_apply', Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
        ih (tickIndex := tickIndex + 1) (idleRunStep s)

theorem runInputs_append_idle
    (m tickIndex birthIndex : Nat)
    (s : RunState) (ticks : List RawTick)
    (out : Result) (k : Nat)
    (h : runInputs m tickIndex birthIndex s ticks = .ok out) :
    runInputs m tickIndex birthIndex s
        (ticks ++ List.replicate k none) =
      .ok ⟨runIdleTrajectory out.final k, out.probability⟩ := by
  induction ticks generalizing tickIndex birthIndex s out with
  | nil =>
      simp only [runInputs, Except.ok.injEq] at h
      cases h
      simpa using runInputs_replicate_idle m tickIndex birthIndex k s
  | cons t rest ih =>
      cases t with
      | none =>
          exact ih (tickIndex := tickIndex + 1) (birthIndex := birthIndex)
            (s := idleRunStep s) (out := out) h
      | some raw =>
          cases hc : checkedBirth s.state m raw with
          | error e => cases e <;> simp [runInputs, hc] at h
          | ok next =>
              cases hr : runInputs m (tickIndex + 1) (birthIndex + 1)
                  ⟨s.nodeCount + 1, s.roundIndex + 1, advance next.1⟩ rest with
              | error e => simp [runInputs, hc, hr] at h
              | ok tail =>
                  have hout : (⟨tail.final, next.2 * tail.probability⟩ : Result) = out := by
                    simpa only [runInputs, hc, hr, Except.ok.injEq] using h
                  cases hout
                  have ht := ih (tickIndex := tickIndex + 1) (birthIndex := birthIndex + 1)
                    (s := ⟨s.nodeCount + 1, s.roundIndex + 1, advance next.1⟩)
                    (out := tail) hr
                  simpa only [List.cons_append, runInputs, hc, ht]

/-- Seed network, fixed attachment count, and roster are checked in that order. -/
def replay (seed : FitnessAttachment.RawSeed) (m : Nat) (agents : Array RawAgent)
    (ticks : List RawTick) : Except JointError Result :=
  match parseSeed seed with
  | .error e => .error (.seedNetwork e)
  | .ok network =>
    if 0 < m ∧ m ≤ seed.nodeCount then
      match parseAgents seed.nodeCount agents with
      | .error e => .error e
      | .ok p => runInputs m 0 0 ⟨seed.nodeCount, 0, ⟨network, p.val, p.property⟩⟩ ticks
    else .error .initialM

theorem replay_append_idle
    (seed : FitnessAttachment.RawSeed) (m : Nat)
    (agents : Array RawAgent) (ticks : List RawTick)
    (out : Result) (k : Nat)
    (h : replay seed m agents ticks = .ok out) :
    replay seed m agents (ticks ++ List.replicate k none) =
      .ok ⟨runIdleTrajectory out.final k, out.probability⟩ := by
  cases hs : parseSeed seed with
  | error e => simp [replay, hs] at h
  | ok network =>
      by_cases hm : 0 < m ∧ m ≤ seed.nodeCount
      · cases hp : parseAgents seed.nodeCount agents with
        | error e => simp [replay, hs, hm, hp] at h
        | ok p =>
            have hr : runInputs m 0 0
                ⟨seed.nodeCount, 0, ⟨network, p.val, p.property⟩⟩ ticks = .ok out := by
              simpa [replay, hs, hm, hp] using h
            have ha := runInputs_append_idle m 0 0
              ⟨seed.nodeCount, 0, ⟨network, p.val, p.property⟩⟩ ticks out k hr
            simpa [replay, hs, hm, hp] using ha
      · simp [replay, hs, hm] at h

def WellFormedTicks (n m : Nat) : List RawTick → Prop
  | [] => True
  | none :: rest => WellFormedTicks n m rest
  | some raw :: rest =>
    raw.birth.Valid n m ∧ raw.AgentValid ∧ WellFormedTicks (n + 1) m rest

def ReplayInputValid (seed : FitnessAttachment.RawSeed) (m : Nat)
    (agents : Array RawAgent) (ticks : List RawTick) : Prop :=
  seed.Valid ∧ (0 < m ∧ m ≤ seed.nodeCount) ∧ agents.size = seed.nodeCount ∧
    (∀ a ∈ agents.toList, a.Valid) ∧ WellFormedTicks seed.nodeCount m ticks

/-- Filter the supplied calendar without changing either target order or data. -/
def inputBirths (ticks : List RawTick) : List FitnessAttachment.RawBirth :=
  ticks.filterMap (fun t => t.map RawBirthInput.birth)

private theorem runInputs_valid_iff (m tickIndex birthIndex : Nat) (s : RunState)
    (ticks : List RawTick) :
    (∃ out, runInputs m tickIndex birthIndex s ticks = .ok out) ↔
      WellFormedTicks s.nodeCount m ticks := by
  induction ticks generalizing tickIndex birthIndex s with
  | nil => simp [runInputs, WellFormedTicks]
  | cons t rest ih =>
    cases t with
    | none => exact ih (tickIndex + 1) birthIndex _
    | some raw =>
      cases hc : checkedBirth s.state m raw with
      | error e =>
        have invalid : ¬ (raw.birth.Valid s.nodeCount m ∧ raw.AgentValid) := by
          intro hv
          obtain ⟨out, ho⟩ := (checkedBirth_valid_iff s.state raw).mpr hv
          simp [hc] at ho
        cases e <;> simp [runInputs, hc, WellFormedTicks] <;> tauto
      | ok next =>
        have hv := (checkedBirth_valid_iff s.state raw).mp ⟨next, hc⟩
        have ht := ih (tickIndex + 1) (birthIndex + 1)
          (⟨s.nodeCount + 1, s.roundIndex + 1, advance next.1⟩ : RunState)
        cases hr : runInputs m (tickIndex + 1) (birthIndex + 1)
            ⟨s.nodeCount + 1, s.roundIndex + 1, advance next.1⟩ rest with
        | error e =>
          simp [hr] at ht
          simp [runInputs, hc, hr, WellFormedTicks, hv.1, hv.2, ht]
        | ok tail =>
          have wf := ht.mp ⟨tail, hr⟩
          simp [runInputs, hc, hr, WellFormedTicks, hv.1, hv.2, wf]

private theorem replay_success (seed : FitnessAttachment.RawSeed) (m : Nat)
    (agents : Array RawAgent) (ticks : List RawTick) (out : Result)
    (h : replay seed m agents ticks = .ok out) :
    ∃ (network : FitnessAttachment.State seed.nodeCount)
      (p : {p : Population seed.nodeCount // p.Valid}),
      parseSeed seed = .ok network ∧ (0 < m ∧ m ≤ seed.nodeCount) ∧
      parseAgents seed.nodeCount agents = .ok p ∧
      runInputs m 0 0 ⟨seed.nodeCount, 0, ⟨network, p.val, p.property⟩⟩ ticks = .ok out := by
  cases hs : parseSeed seed with
  | error e => simp [replay, hs] at h
  | ok network =>
    by_cases hm : 0 < m ∧ m ≤ seed.nodeCount
    · cases hp : parseAgents seed.nodeCount agents with
      | error e => simp [replay, hs, hm, hp] at h
      | ok p => exact ⟨network, p, rfl, hm, rfl, by simpa [replay, hs, hm, hp] using h⟩
    · simp [replay, hs, hm] at h

theorem replay_success_iff_valid (seed : FitnessAttachment.RawSeed) (m : Nat)
    (agents : Array RawAgent) (ticks : List RawTick) :
    (∃ out, replay seed m agents ticks = .ok out) ↔ ReplayInputValid seed m agents ticks := by
  constructor
  · rintro ⟨out, h⟩
    obtain ⟨network, p, hs, hm, hp, hr⟩ := replay_success seed m agents ticks out h
    have hv := parseAgents_sound seed.nodeCount agents p hp
    exact ⟨parseSeed_sound seed network hs, hm, hv.1, hv.2.1,
      (runInputs_valid_iff m 0 0 _ ticks).mp ⟨out, hr⟩⟩
  · rintro ⟨hs, hm, hsize, ha, ht⟩
    obtain ⟨network, hn⟩ := parseSeed_complete seed hs
    obtain ⟨p, hp⟩ := parseAgents_complete seed.nodeCount agents hsize ha
    obtain ⟨out, hr⟩ := (runInputs_valid_iff m 0 0
      ⟨seed.nodeCount, 0, ⟨network, p.val, p.property⟩⟩ ticks).mpr ht
    exact ⟨out, by simpa [replay, hn, hm, hp] using hr⟩

private def projectState (s : RunState) : FitnessAttachment.RunState :=
  ⟨s.nodeCount, s.state.network⟩

private def projectResult (out : Result) : FitnessAttachment.ReplayResult :=
  ⟨projectState out.final, out.probability⟩

private theorem runInputs_projection (m tickIndex birthIndex : Nat) (s : RunState)
    (ticks : List RawTick) (out : Result)
    (h : runInputs m tickIndex birthIndex s ticks = .ok out) :
    runBirths m birthIndex (projectState s) (inputBirths ticks) = .ok (projectResult out) := by
  induction ticks generalizing tickIndex birthIndex s out with
  | nil =>
    simp only [runInputs, Except.ok.injEq] at h
    cases h
    rfl
  | cons t rest ih =>
    cases t with
    | none =>
      exact ih (tickIndex := tickIndex + 1) (birthIndex := birthIndex)
        (s := ⟨s.nodeCount, s.roundIndex + 1, advance s.state⟩) (out := out) h
    | some raw =>
      cases hc : checkedBirth s.state m raw with
      | error e => cases e <;> simp [runInputs, hc] at h
      | ok next =>
        cases hr : runInputs m (tickIndex + 1) (birthIndex + 1)
            ⟨s.nodeCount + 1, s.roundIndex + 1, advance next.1⟩ rest with
        | error e => simp [runInputs, hc, hr] at h
        | ok tail =>
          have hout : (⟨tail.final, next.2 * tail.probability⟩ : Result) = out := by
            simpa only [runInputs, hc, hr, Except.ok.injEq] using h
          cases hout
          have hp := checkedBirth_projection s.state raw next hc
          have ht := ih (tickIndex := tickIndex + 1) (birthIndex := birthIndex + 1)
            (s := ⟨s.nodeCount + 1, s.roundIndex + 1, advance next.1⟩) (out := tail) hr
          change runBirths m birthIndex ⟨s.nodeCount, s.state.network⟩
            (raw.birth :: inputBirths rest) =
              .ok (projectResult ⟨tail.final, next.2 * tail.probability⟩)
          rw [FitnessAttachment.replay_step, hp]
          dsimp only
          change runBirths m (birthIndex + 1)
            ⟨s.nodeCount + 1, next.1.network⟩ (inputBirths rest) =
              .ok (projectResult tail) at ht
          rw [ht]
          rfl

private theorem replay_project_exact (seed : FitnessAttachment.RawSeed) (m : Nat)
    (agents : Array RawAgent) (ticks : List RawTick) (out : Result)
    (h : replay seed m agents ticks = .ok out) :
    FitnessAttachment.replay seed m (inputBirths ticks) = .ok (projectResult out) := by
  obtain ⟨network, p, hs, hm, _, hr⟩ := replay_success seed m agents ticks out h
  simpa only [FitnessAttachment.replay, hs, if_pos hm, projectState] using
    runInputs_projection m 0 0 ⟨seed.nodeCount, 0, ⟨network, p.val, p.property⟩⟩ ticks out hr

/-- The projected full BB state agrees; the explicit carrier witness also
identifies every adjacency and stored fitness value across Fin.cast. -/
theorem replay_projection (seed : FitnessAttachment.RawSeed) (m : Nat)
    (agents : Array RawAgent) (ticks : List RawTick) (out : Result)
    (h : replay seed m agents ticks = .ok out) :
    ∃ bb : FitnessAttachment.ReplayResult,
      FitnessAttachment.replay seed m (inputBirths ticks) = .ok bb ∧
      bb.final = ⟨out.final.nodeCount, out.final.state.network⟩ ∧
      ∃ hn : out.final.nodeCount = bb.final.nodeCount,
        (∀ i j, out.final.state.network.snapshot.graph.Adj i j ↔
          bb.final.state.snapshot.graph.Adj (Fin.cast hn i) (Fin.cast hn j)) ∧
        (∀ i, out.final.state.network.snapshot.fitness i =
          bb.final.state.snapshot.fitness (Fin.cast hn i)) ∧
        out.probability = bb.probability := by
  exact ⟨projectResult out, replay_project_exact seed m agents ticks out h,
    rfl, rfl, fun _ _ => Iff.rfl, fun _ => rfl, rfl⟩

private theorem runInputs_rounds (m tickIndex birthIndex : Nat) (s : RunState)
    (ticks : List RawTick) (out : Result)
    (h : runInputs m tickIndex birthIndex s ticks = .ok out) :
    out.final.roundIndex = s.roundIndex + ticks.length := by
  induction ticks generalizing tickIndex birthIndex s out with
  | nil =>
    simp only [runInputs, Except.ok.injEq] at h
    cases h
    simp
  | cons t rest ih =>
    cases t with
    | none =>
      have ht := ih (tickIndex := tickIndex + 1) (birthIndex := birthIndex)
        (s := ⟨s.nodeCount, s.roundIndex + 1, advance s.state⟩) (out := out) h
      simpa only [List.length_cons, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using ht
    | some raw =>
      cases hc : checkedBirth s.state m raw with
      | error e => cases e <;> simp [runInputs, hc] at h
      | ok next =>
        cases hr : runInputs m (tickIndex + 1) (birthIndex + 1)
            ⟨s.nodeCount + 1, s.roundIndex + 1, advance next.1⟩ rest with
        | error e => simp [runInputs, hc, hr] at h
        | ok tail =>
          have hout : (⟨tail.final, next.2 * tail.probability⟩ : Result) = out := by
            simpa only [runInputs, hc, hr, Except.ok.injEq] using h
          cases hout
          have ht := ih (tickIndex := tickIndex + 1) (birthIndex := birthIndex + 1)
            (s := ⟨s.nodeCount + 1, s.roundIndex + 1, advance next.1⟩) (out := tail) hr
          simpa only [List.length_cons, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using ht

/-- Node and actual edge counts come from BB; all supplied ticks count as rounds. -/
theorem replay_counts (seed : FitnessAttachment.RawSeed) (m : Nat)
    (agents : Array RawAgent) (ticks : List RawTick) (out : Result)
    (network : FitnessAttachment.State seed.nodeCount)
    (hs : parseSeed seed = .ok network) (h : replay seed m agents ticks = .ok out) :
    out.final.nodeCount = seed.nodeCount + (inputBirths ticks).length ∧
    out.final.roundIndex = ticks.length ∧
    actualEdgeCount out.final.state.network.snapshot =
      actualEdgeCount network.snapshot + m * (inputBirths ticks).length := by
  have hp := replay_project_exact seed m agents ticks out h
  obtain ⟨initial, p, _, _, _, hr⟩ := replay_success seed m agents ticks out h
  refine ⟨?_, ?_, FitnessAttachment.replay_edges seed m _ (projectResult out) network hs hp⟩
  · exact (Fintype.card_fin out.final.nodeCount).symm.trans
      (FitnessAttachment.replay_nodes seed m _ (projectResult out) hp)
  · simpa only [Nat.zero_add] using
      runInputs_rounds m 0 0 ⟨seed.nodeCount, 0, ⟨initial, p.val, p.property⟩⟩ ticks out hr

theorem replay_probability_pos (seed : FitnessAttachment.RawSeed) (m : Nat)
    (agents : Array RawAgent) (ticks : List RawTick) (out : Result)
    (h : replay seed m agents ticks = .ok out) : 0 < out.probability :=
  FitnessAttachment.replay_probability_pos seed m (inputBirths ticks)
    (projectResult out) (replay_project_exact seed m agents ticks out h)

/-- An error is stable under every appended suffix, with its exact two positions. -/
theorem runInputs_append_error (m tickIndex birthIndex : Nat) (s : RunState)
    (preTicks suffix : List RawTick) (e : JointError)
    (h : runInputs m tickIndex birthIndex s preTicks = .error e) :
    runInputs m tickIndex birthIndex s (preTicks ++ suffix) = .error e := by
  induction preTicks generalizing tickIndex birthIndex s with
  | nil => simp [runInputs] at h
  | cons t rest ih =>
    cases t with
    | none =>
      exact ih (tickIndex := tickIndex + 1) (birthIndex := birthIndex)
        (s := ⟨s.nodeCount, s.roundIndex + 1, advance s.state⟩) h
    | some raw =>
      cases hc : checkedBirth s.state m raw with
      | error cause =>
        cases cause <;> simpa only [List.cons_append, runInputs, hc] using h
      | ok next =>
        cases hr : runInputs m (tickIndex + 1) (birthIndex + 1)
            ⟨s.nodeCount + 1, s.roundIndex + 1, advance next.1⟩ rest with
        | error err =>
          have he : err = e := by
            simpa only [runInputs, hc, hr, Except.error.injEq] using h
          cases he
          have ht := ih (tickIndex := tickIndex + 1) (birthIndex := birthIndex + 1)
            (s := ⟨s.nodeCount + 1, s.roundIndex + 1, advance next.1⟩) hr
          simp only [List.cons_append, runInputs, hc, ht]
        | ok tail => simp [runInputs, hc, hr] at h

end NarrativeDynamics.FitnessABM
