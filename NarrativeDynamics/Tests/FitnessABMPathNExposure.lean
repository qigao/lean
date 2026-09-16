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
