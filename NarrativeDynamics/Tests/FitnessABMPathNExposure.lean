import NarrativeDynamics.Core.FitnessABMPathNExposure

open NarrativeDynamics
open NarrativeDynamics.FitnessABMPathNExposure

private def halfSchedule : ExposureParameters :=
  ⟨fun _ => 1/2, 1/2⟩

example : ExposureParameters.Valid halfSchedule := by
  constructor
  · intro e
    norm_num [halfSchedule]
  · norm_num [halfSchedule]

private def thresholdState : State 2 :=
  fun i => if i = 0 then ⟨1/2, 0⟩ else ⟨0, 0⟩

example : broadcasting halfSchedule thresholdState 0 = true := by
  decide_cbv

example : beliefs thresholdState 0 = 1/2 := by
  norm_num [beliefs, thresholdState]

private def exposureSensitive : ExposureParameters :=
  ⟨fun e => if e = 0 then 1/4 else 3/4, 1/2⟩

private def sameStepState : State 2 :=
  fun i => if i = 0 then ⟨1, 0⟩ else ⟨0, 0⟩

example : incoming exposureSensitive sameStepState 1 = 1 := by
  decide_cbv

-- The current round contributes one incoming exposure first, so the receiver
-- uses α(1)=3/4 in this same step rather than α(0)=1/4.
example : (step exposureSensitive 2 sameStepState 1).exposure = 1 := by
  decide_cbv

example : (step exposureSensitive 2 sameStepState 1).belief = 3/4 := by
  decide_cbv

example (alpha tau : Rat) (n : Nat) (s : State n) :
    beliefs (step ⟨fun _ => alpha, tau⟩ n s) =
      FitnessABMPathNParameters.beliefStep ⟨alpha, tau⟩ n (beliefs s) := by
  exact constant_beliefStep alpha tau n s

example (n : Nat) (s : State n) :
    beliefs (step ⟨fun _ => 1/2, 1/2⟩ n s) =
      FitnessABMPathN.beliefStep n (beliefs s) := by
  exact half_beliefStep n s

-- Cumulative exposure never decreases, and valid schedules keep every exact
-- rational belief inside [0,1].
example (p : ExposureParameters) (n : Nat) (s : State n) (i : Fin n) :
    (s i).exposure ≤ (step p n s i).exposure := by
  exact exposure_mono p n s i

example (p : ExposureParameters) (n : Nat) (s : State n)
    (hvalid : p.Valid) (hbounded : BeliefsBounded s) :
    BeliefsBounded (step p n s) := by
  exact beliefs_bounded_step p n s hvalid hbounded

private def twoIncomingState : State 3 :=
  ![⟨1, 0⟩, ⟨0, 4⟩, ⟨1, 0⟩]

example : incoming halfSchedule twoIncomingState 1 = 2 := by
  decide_cbv

example : BeliefsBounded twoIncomingState := by
  intro i
  fin_cases i <;> norm_num [twoIncomingState]

example :
    (twoIncomingState 1).exposure ≤
      (step halfSchedule 3 twoIncomingState 1).exposure := by
  exact exposure_mono halfSchedule 3 twoIncomingState 1

example : BeliefsBounded (step halfSchedule 3 twoIncomingState) := by
  exact beliefs_bounded_step halfSchedule 3 twoIncomingState
    (by
      constructor
      · intro e
        norm_num [halfSchedule]
      · norm_num [halfSchedule])
    (by
      intro i
      fin_cases i <;> norm_num [twoIncomingState])

-- The model selects α only after adding this round's incoming broadcasts.
-- With one incoming neighbor, prior exposures 0 and 1 therefore query α(1)
-- and α(2), respectively.
private def historySchedule : ExposureParameters :=
  ⟨fun e => if e ≤ 1 then 1/4 else 3/4, 1/2⟩

private def lowExposureState : State 2 :=
  ![⟨1, 0⟩, ⟨0, 0⟩]

private def highExposureState : State 2 :=
  ![⟨1, 0⟩, ⟨0, 1⟩]

example : ExposureParameters.Valid historySchedule := by
  constructor
  · intro e
    simp only [historySchedule]
    split <;> norm_num
  · norm_num [historySchedule]

theorem exposure_history_same_beliefs :
    beliefs lowExposureState = beliefs highExposureState := by
  funext i
  fin_cases i <;> rfl

example : incoming historySchedule lowExposureState 1 = 1 := by
  decide_cbv

example : incoming historySchedule highExposureState 1 = 1 := by
  decide_cbv

example : (step historySchedule 2 lowExposureState 1).exposure = 1 := by
  decide_cbv

example : (step historySchedule 2 highExposureState 1).exposure = 2 := by
  decide_cbv

example : (step historySchedule 2 lowExposureState 1).belief = 1/4 := by
  decide_cbv

example : (step historySchedule 2 highExposureState 1).belief = 3/4 := by
  decide_cbv

theorem exposure_history_changes_next_belief :
    (step historySchedule 2 lowExposureState 1).belief ≠
      (step historySchedule 2 highExposureState 1).belief := by
  decide_cbv

/-- The exposure-independent parameterized baseline receives the same belief
vector in both cases, so it has one common next-belief result regardless of
which exposure history produced that vector. -/
theorem baseline_exposure_history_control :
    FitnessABMPathNParameters.beliefStep ⟨1/2, 1/2⟩ 2
        (beliefs lowExposureState) =
      FitnessABMPathNParameters.beliefStep ⟨1/2, 1/2⟩ 2
        (beliefs highExposureState) := by
  rw [exposure_history_same_beliefs]

#print axioms NarrativeDynamics.FitnessABMPathNExposure.exposure_mono
#print axioms NarrativeDynamics.FitnessABMPathNExposure.beliefs_bounded_step
#print axioms exposure_history_changes_next_belief
