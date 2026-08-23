import NarrativeDynamics.Core.Learning

open NarrativeDynamics

example {old observed η : ℝ}
    (hη : 0 ≤ η)
    (hobs : old ≤ observed) :
    old ≤ learnInstrumentality old observed η := by
  exact learnInstrumentality_moves_up hη hobs

example {old observed η : ℝ}
    (hη0 : 0 ≤ η)
    (hη1 : η ≤ 1)
    (hobs : old ≤ observed) :
    learnInstrumentality old observed η ≤ observed := by
  exact learnInstrumentality_no_overshoot_up hη0 hη1 hobs

example {old observed η : ℝ}
    (hη : 0 ≤ η)
    (hobs : observed ≤ old) :
    learnInstrumentality old observed η ≤ old := by
  exact learnInstrumentality_moves_down hη hobs
