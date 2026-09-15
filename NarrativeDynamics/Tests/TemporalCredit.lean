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

example (gamma lambda : ℝ) :
    causalTraceCoeff gamma lambda 0 = 1 := by
  simpa using causal_trace_coeff_closed_form gamma lambda 0

example (gamma lambda : ℝ) :
    causalTraceCoeff gamma lambda 1 = gamma * lambda := by
  simpa using causal_trace_coeff_closed_form gamma lambda 1

example (gamma lambda : ℝ) :
    causalTraceCoeff gamma lambda 3 = (gamma * lambda) ^ 3 := by
  exact causal_trace_coeff_closed_form gamma lambda 3

example (gamma lambda : ℝ) :
    causalTraceCoeff gamma lambda 5 = (gamma * lambda) ^ 5 := by
  exact causal_trace_coeff_closed_form gamma lambda 5

example
    (gamma lambda : ℝ)
    (hgamma : gamma ≠ 0)
    (hlambda : lambda ≠ 0) :
    causalTraceCoeff gamma lambda 5 ≠ 0 := by
  exact causal_trace_coeff_ne_zero gamma lambda 5 hgamma hlambda

end NarrativeDynamics.TemporalCredit.Tests
