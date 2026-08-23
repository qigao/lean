import Mathlib

namespace NarrativeDynamics

/-- Prediction-error update for an agent's learned estimate of how
instrumental a candidate goal is for relieving a drive. -/
def learnInstrumentality (old observed η : ℝ) : ℝ :=
  old + η * (observed - old)

/-- With a nonnegative learning rate, a higher observed outcome moves the
estimate upward. -/
theorem learnInstrumentality_moves_up {old observed η : ℝ}
    (hη : 0 ≤ η) (hobs : old ≤ observed) :
    old ≤ learnInstrumentality old observed η := by
  unfold learnInstrumentality
  nlinarith

/-- If the learning rate is at most one, an upward update cannot overshoot
the observed outcome. -/
theorem learnInstrumentality_no_overshoot_up {old observed η : ℝ}
    (hη0 : 0 ≤ η) (hη1 : η ≤ 1) (hobs : old ≤ observed) :
    learnInstrumentality old observed η ≤ observed := by
  unfold learnInstrumentality
  nlinarith

/-- With a nonnegative learning rate, a lower observed outcome moves the
estimate downward. -/
theorem learnInstrumentality_moves_down {old observed η : ℝ}
    (hη : 0 ≤ η) (hobs : observed ≤ old) :
    learnInstrumentality old observed η ≤ old := by
  unfold learnInstrumentality
  nlinarith

end NarrativeDynamics
