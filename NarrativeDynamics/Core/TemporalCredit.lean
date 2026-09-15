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
    {Cue Act Obs Label Reward : Type}
    (rewardFn : Cue → Act → Reward)
    (trial : Trial Cue Act Obs Label) : Reward :=
  rewardFn trial.cue trial.causalAction

def learnerView
    {Cue Act Obs Label Reward : Type}
    (rewardFn : Cue → Act → Reward)
    (trial : Trial Cue Act Obs Label) : LearnerView Obs Reward :=
  {
    observations := trial.observations
    terminalReward := trialReward rewardFn trial
  }

def withDistractors
    {Cue Act Obs Label : Type}
    (trial : Trial Cue Act Obs Label)
    (distractors : List Act) : Trial Cue Act Obs Label :=
  { trial with distractors := distractors }

def withCausalLabel
    {Cue Act Obs Label : Type}
    (trial : Trial Cue Act Obs Label)
    (label : Label) : Trial Cue Act Obs Label :=
  { trial with causalLabel := label }

theorem reward_invariant_under_distractor_substitution
    {Cue Act Obs Label Reward : Type}
    (rewardFn : Cue → Act → Reward)
    (trial : Trial Cue Act Obs Label)
    (xs ys : List Act) :
    trialReward rewardFn (withDistractors trial xs) =
      trialReward rewardFn (withDistractors trial ys) := by
  rfl

theorem learner_view_independent_of_causal_label
    {Cue Act Obs Label Reward : Type}
    (rewardFn : Cue → Act → Reward)
    (trial : Trial Cue Act Obs Label)
    (label₁ label₂ : Label) :
    learnerView rewardFn (withCausalLabel trial label₁) =
      learnerView rewardFn (withCausalLabel trial label₂) := by
  rfl

def td0DirectCredit (terminalPos pos : Nat) : ℝ :=
  if pos = terminalPos then 1 else 0

theorem terminal_td0_zero_direct_causal_credit
    (d : Nat)
    (hd : 0 < d) :
    td0DirectCredit d 0 = 0 := by
  have hne : (0 : Nat) ≠ d := Nat.ne_of_lt hd
  simp [td0DirectCredit, hne]

theorem terminal_td0_terminal_credit
    (d : Nat) :
    td0DirectCredit d d = 1 := by
  simp [td0DirectCredit]

end NarrativeDynamics.TemporalCredit
