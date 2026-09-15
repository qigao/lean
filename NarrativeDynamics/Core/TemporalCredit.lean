import Mathlib

namespace NarrativeDynamics.TemporalCredit

structure Trial (Cue Action Obs Label : Type) where
  cue : Cue
  causalAction : Action
  distractors : List Action
  observations : List Obs
  causalLabel : Label

structure LearnerView (Obs Reward : Type) where
  observations : List Obs
  terminalReward : Reward
  deriving DecidableEq

def trialReward
    (rewardFn : Cue → Action → Reward)
    (trial : Trial Cue Action Obs Label) : Reward :=
  rewardFn trial.cue trial.causalAction

def learnerView
    (rewardFn : Cue → Action → Reward)
    (trial : Trial Cue Action Obs Label) : LearnerView Obs Reward :=
  {
    observations := trial.observations
    terminalReward := trialReward rewardFn trial
  }

def withDistractors
    (trial : Trial Cue Action Obs Label)
    (distractors : List Action) : Trial Cue Action Obs Label :=
  { trial with distractors := distractors }

def withCausalLabel
    (trial : Trial Cue Action Obs Label)
    (label : Label) : Trial Cue Action Obs Label :=
  { trial with causalLabel := label }

theorem reward_invariant_under_distractor_substitution
    (rewardFn : Cue → Action → Reward)
    (trial : Trial Cue Action Obs Label)
    (xs ys : List Action) :
    trialReward rewardFn (withDistractors trial xs) =
      trialReward rewardFn (withDistractors trial ys) := by
  rfl

theorem learner_view_independent_of_causal_label
    (rewardFn : Cue → Action → Reward)
    (trial : Trial Cue Action Obs Label)
    (label₁ label₂ : Label) :
    learnerView rewardFn (withCausalLabel trial label₁) =
      learnerView rewardFn (withCausalLabel trial label₂) := by
  rfl

end NarrativeDynamics.TemporalCredit
