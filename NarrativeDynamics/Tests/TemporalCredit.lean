import NarrativeDynamics.Core.TemporalCredit

open NarrativeDynamics.TemporalCredit

namespace NarrativeDynamics.TemporalCredit.Tests

inductive Cue where
  | left
  | right
  deriving DecidableEq

inductive Action where
  | stay
  | move
  deriving DecidableEq

inductive Label where
  | causal
  | hidden
  deriving DecidableEq

abbrev Obs := Nat
abbrev Reward := Int

private def rewardFn : Cue → Action → Reward
  | .left, .stay => 1
  | .right, .move => 1
  | _, _ => -1

private def baseTrial : Trial Cue Action Obs Label :=
  {
    cue := .left
    causalAction := .stay
    distractors := [.move]
    observations := [10, 20]
    causalLabel := .causal
  }

example :
    trialReward rewardFn (withDistractors baseTrial [.stay, .move]) =
      trialReward rewardFn (withDistractors baseTrial [.move, .move, .stay]) := by
  exact reward_invariant_under_distractor_substitution
    rewardFn baseTrial [.stay, .move] [.move, .move, .stay]

example :
    learnerView rewardFn (withCausalLabel baseTrial .causal) =
      learnerView rewardFn (withCausalLabel baseTrial .hidden) := by
  exact learner_view_independent_of_causal_label rewardFn baseTrial .causal .hidden

example : td0DirectCredit 0 0 = 1 := by
  exact terminal_td0_terminal_credit 0

example : td0DirectCredit 1 0 = 0 := by
  exact terminal_td0_zero_direct_causal_credit 1 (by decide)

example : td0DirectCredit 3 0 = 0 := by
  exact terminal_td0_zero_direct_causal_credit 3 (by decide)

example : td0DirectCredit 5 0 = 0 := by
  exact terminal_td0_zero_direct_causal_credit 5 (by decide)

end NarrativeDynamics.TemporalCredit.Tests
