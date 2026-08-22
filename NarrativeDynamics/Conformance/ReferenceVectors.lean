import NarrativeDynamics.Core.Belief
import NarrativeDynamics.Core.Drive
import NarrativeDynamics.Core.Learning

namespace NarrativeDynamics.Conformance

/-- Exact rational transport used by the conformance corpus. The denominator
is positive in every declared vector and the emitted form is reduced. -/
structure ExactRat where
  numerator : Int
  denominator : Nat

/-- Interpret one exact reference value in the same real-number domain used by
the formal core definitions. -/
def ExactRat.toReal (value : ExactRat) : ℝ :=
  (value.numerator : ℝ) / (value.denominator : ℝ)

private def exactRat (numerator : Int) (denominator : Nat) : ExactRat :=
  { numerator := numerator, denominator := denominator }

private def jsonQuote (value : String) : String :=
  "\"" ++ value ++ "\""

private def ExactRat.toJson (value : ExactRat) : String :=
  jsonQuote (toString value.numerator ++ "/" ++ toString value.denominator)

structure ReferenceVector where
  id : String
  operation : String
  inputs : List (String × ExactRat)
  expected : ExactRat

private def ReferenceVector.toJson (vector : ReferenceVector) : String :=
  let renderedInputs := vector.inputs.map (fun entry =>
    jsonQuote entry.1 ++ ":" ++ ExactRat.toJson entry.2)
  "{" ++
    jsonQuote "id" ++ ":" ++ jsonQuote vector.id ++ "," ++
    jsonQuote "operation" ++ ":" ++ jsonQuote vector.operation ++ "," ++
    jsonQuote "inputs" ++ ":{" ++
      String.intercalate "," renderedInputs ++ "}," ++
    jsonQuote "expected" ++ ":" ++ ExactRat.toJson vector.expected ++
  "}"

structure BayesCase where
  id : String
  prior : ExactRat
  likelihoodH : ExactRat
  likelihoodNotH : ExactRat
  expected : ExactRat

private def BayesCase.toReferenceVector (value : BayesCase) : ReferenceVector :=
  {
    id := value.id
    operation := "bayes_posterior"
    inputs := [
      ("prior", value.prior),
      ("likelihood_h", value.likelihoodH),
      ("likelihood_not_h", value.likelihoodNotH)
    ]
    expected := value.expected
  }

structure LearningCase where
  id : String
  old : ExactRat
  observed : ExactRat
  rate : ExactRat
  expected : ExactRat

private def LearningCase.toReferenceVector
    (value : LearningCase) : ReferenceVector :=
  {
    id := value.id
    operation := "learn_instrumentality"
    inputs := [
      ("old", value.old),
      ("observed", value.observed),
      ("rate", value.rate)
    ]
    expected := value.expected
  }

structure EffectivePressureCase where
  id : String
  pSelf : ExactRat
  pOther : ExactRat
  boundary : ExactRat
  expected : ExactRat

private def EffectivePressureCase.toReferenceVector
    (value : EffectivePressureCase) : ReferenceVector :=
  {
    id := value.id
    operation := "effective_pressure2"
    inputs := [
      ("p_self", value.pSelf),
      ("p_other", value.pOther),
      ("boundary", value.boundary)
    ]
    expected := value.expected
  }

structure GoalScoreCase where
  id : String
  pressure : ExactRat
  instrumentality : ExactRat
  cost : ExactRat
  risk : ExactRat
  expected : ExactRat

private def GoalScoreCase.toReferenceVector
    (value : GoalScoreCase) : ReferenceVector :=
  {
    id := value.id
    operation := "goal_score_single_drive"
    inputs := [
      ("pressure", value.pressure),
      ("instrumentality", value.instrumentality),
      ("cost", value.cost),
      ("risk", value.risk)
    ]
    expected := value.expected
  }

-- Bayesian posterior vectors. Every denominator is strictly positive.
def bayesNegativeEvidence : BayesCase :=
  {
    id := "bayes_negative_evidence"
    prior := exactRat 3 4
    likelihoodH := exactRat 1 4
    likelihoodNotH := exactRat 3 4
    expected := exactRat 1 2
  }

theorem bayesNegativeEvidence_sound :
    NarrativeDynamics.bayesPosterior
        bayesNegativeEvidence.prior.toReal
        bayesNegativeEvidence.likelihoodH.toReal
        bayesNegativeEvidence.likelihoodNotH.toReal =
      bayesNegativeEvidence.expected.toReal := by
  norm_num [bayesNegativeEvidence, ExactRat.toReal, exactRat,
    NarrativeDynamics.bayesPosterior]

def bayesNonbinaryFraction : BayesCase :=
  {
    id := "bayes_nonbinary_fraction"
    prior := exactRat 1 2
    likelihoodH := exactRat 2 3
    likelihoodNotH := exactRat 1 3
    expected := exactRat 2 3
  }

theorem bayesNonbinaryFraction_sound :
    NarrativeDynamics.bayesPosterior
        bayesNonbinaryFraction.prior.toReal
        bayesNonbinaryFraction.likelihoodH.toReal
        bayesNonbinaryFraction.likelihoodNotH.toReal =
      bayesNonbinaryFraction.expected.toReal := by
  norm_num [bayesNonbinaryFraction, ExactRat.toReal, exactRat,
    NarrativeDynamics.bayesPosterior]

def bayesPositiveEvidence : BayesCase :=
  {
    id := "bayes_positive_evidence"
    prior := exactRat 1 2
    likelihoodH := exactRat 3 4
    likelihoodNotH := exactRat 1 4
    expected := exactRat 3 4
  }

theorem bayesPositiveEvidence_sound :
    NarrativeDynamics.bayesPosterior
        bayesPositiveEvidence.prior.toReal
        bayesPositiveEvidence.likelihoodH.toReal
        bayesPositiveEvidence.likelihoodNotH.toReal =
      bayesPositiveEvidence.expected.toReal := by
  norm_num [bayesPositiveEvidence, ExactRat.toReal, exactRat,
    NarrativeDynamics.bayesPosterior]

def bayesUninformative : BayesCase :=
  {
    id := "bayes_uninformative"
    prior := exactRat 1 4
    likelihoodH := exactRat 1 2
    likelihoodNotH := exactRat 1 2
    expected := exactRat 1 4
  }

theorem bayesUninformative_sound :
    NarrativeDynamics.bayesPosterior
        bayesUninformative.prior.toReal
        bayesUninformative.likelihoodH.toReal
        bayesUninformative.likelihoodNotH.toReal =
      bayesUninformative.expected.toReal := by
  norm_num [bayesUninformative, ExactRat.toReal, exactRat,
    NarrativeDynamics.bayesPosterior]

-- Two-person effective-pressure vectors.
def effectivePressureFullBoundary : EffectivePressureCase :=
  {
    id := "effective_pressure_full_boundary"
    pSelf := exactRat 1 8
    pOther := exactRat 3 8
    boundary := exactRat 1 1
    expected := exactRat 1 2
  }

theorem effectivePressureFullBoundary_sound :
    NarrativeDynamics.effectivePressure
        effectivePressureFullBoundary.pSelf.toReal
        effectivePressureFullBoundary.pOther.toReal
        effectivePressureFullBoundary.boundary.toReal =
      effectivePressureFullBoundary.expected.toReal := by
  norm_num [effectivePressureFullBoundary, ExactRat.toReal, exactRat,
    NarrativeDynamics.effectivePressure]

def effectivePressureHalfBoundary : EffectivePressureCase :=
  {
    id := "effective_pressure_half_boundary"
    pSelf := exactRat 1 4
    pOther := exactRat 3 4
    boundary := exactRat 1 2
    expected := exactRat 5 8
  }

theorem effectivePressureHalfBoundary_sound :
    NarrativeDynamics.effectivePressure
        effectivePressureHalfBoundary.pSelf.toReal
        effectivePressureHalfBoundary.pOther.toReal
        effectivePressureHalfBoundary.boundary.toReal =
      effectivePressureHalfBoundary.expected.toReal := by
  norm_num [effectivePressureHalfBoundary, ExactRat.toReal, exactRat,
    NarrativeDynamics.effectivePressure]

def effectivePressureZeroBoundary : EffectivePressureCase :=
  {
    id := "effective_pressure_zero_boundary"
    pSelf := exactRat 2 3
    pOther := exactRat 5 6
    boundary := exactRat 0 1
    expected := exactRat 2 3
  }

theorem effectivePressureZeroBoundary_sound :
    NarrativeDynamics.effectivePressure
        effectivePressureZeroBoundary.pSelf.toReal
        effectivePressureZeroBoundary.pOther.toReal
        effectivePressureZeroBoundary.boundary.toReal =
      effectivePressureZeroBoundary.expected.toReal := by
  norm_num [effectivePressureZeroBoundary, ExactRat.toReal, exactRat,
    NarrativeDynamics.effectivePressure]

-- One-drive goal-score vectors. Pressures are nonnegative so the Python
-- AgentMotivation boundary is exactly the formal goalScore equation.
def goalScoreCostRiskOnly : GoalScoreCase :=
  {
    id := "goal_score_cost_risk_only"
    pressure := exactRat 1 1
    instrumentality := exactRat 0 1
    cost := exactRat 1 8
    risk := exactRat 1 8
    expected := exactRat (-1) 4
  }

theorem goalScoreCostRiskOnly_sound :
    NarrativeDynamics.goalScore
        goalScoreCostRiskOnly.pressure.toReal
        goalScoreCostRiskOnly.instrumentality.toReal
        goalScoreCostRiskOnly.cost.toReal
        goalScoreCostRiskOnly.risk.toReal =
      goalScoreCostRiskOnly.expected.toReal := by
  norm_num [goalScoreCostRiskOnly, ExactRat.toReal, exactRat,
    NarrativeDynamics.goalScore]

def goalScoreFractional : GoalScoreCase :=
  {
    id := "goal_score_fractional"
    pressure := exactRat 2 3
    instrumentality := exactRat 3 4
    cost := exactRat 1 6
    risk := exactRat 1 12
    expected := exactRat 1 4
  }

theorem goalScoreFractional_sound :
    NarrativeDynamics.goalScore
        goalScoreFractional.pressure.toReal
        goalScoreFractional.instrumentality.toReal
        goalScoreFractional.cost.toReal
        goalScoreFractional.risk.toReal =
      goalScoreFractional.expected.toReal := by
  norm_num [goalScoreFractional, ExactRat.toReal, exactRat,
    NarrativeDynamics.goalScore]

def goalScorePositiveNet : GoalScoreCase :=
  {
    id := "goal_score_positive_net"
    pressure := exactRat 3 2
    instrumentality := exactRat 1 2
    cost := exactRat 1 4
    risk := exactRat 1 8
    expected := exactRat 3 8
  }

theorem goalScorePositiveNet_sound :
    NarrativeDynamics.goalScore
        goalScorePositiveNet.pressure.toReal
        goalScorePositiveNet.instrumentality.toReal
        goalScorePositiveNet.cost.toReal
        goalScorePositiveNet.risk.toReal =
      goalScorePositiveNet.expected.toReal := by
  norm_num [goalScorePositiveNet, ExactRat.toReal, exactRat,
    NarrativeDynamics.goalScore]

-- Prediction-error learning vectors.
def learningFullRate : LearningCase :=
  {
    id := "learning_full_rate"
    old := exactRat 1 8
    observed := exactRat 7 8
    rate := exactRat 1 1
    expected := exactRat 7 8
  }

theorem learningFullRate_sound :
    NarrativeDynamics.learnInstrumentality
        learningFullRate.old.toReal
        learningFullRate.observed.toReal
        learningFullRate.rate.toReal =
      learningFullRate.expected.toReal := by
  norm_num [learningFullRate, ExactRat.toReal, exactRat,
    NarrativeDynamics.learnInstrumentality]

def learningMovesDown : LearningCase :=
  {
    id := "learning_moves_down"
    old := exactRat 3 4
    observed := exactRat 1 4
    rate := exactRat 1 4
    expected := exactRat 5 8
  }

theorem learningMovesDown_sound :
    NarrativeDynamics.learnInstrumentality
        learningMovesDown.old.toReal
        learningMovesDown.observed.toReal
        learningMovesDown.rate.toReal =
      learningMovesDown.expected.toReal := by
  norm_num [learningMovesDown, ExactRat.toReal, exactRat,
    NarrativeDynamics.learnInstrumentality]

def learningMovesUp : LearningCase :=
  {
    id := "learning_moves_up"
    old := exactRat 1 4
    observed := exactRat 3 4
    rate := exactRat 1 2
    expected := exactRat 1 2
  }

theorem learningMovesUp_sound :
    NarrativeDynamics.learnInstrumentality
        learningMovesUp.old.toReal
        learningMovesUp.observed.toReal
        learningMovesUp.rate.toReal =
      learningMovesUp.expected.toReal := by
  norm_num [learningMovesUp, ExactRat.toReal, exactRat,
    NarrativeDynamics.learnInstrumentality]

def learningZeroRate : LearningCase :=
  {
    id := "learning_zero_rate"
    old := exactRat 2 3
    observed := exactRat 1 3
    rate := exactRat 0 1
    expected := exactRat 2 3
  }

theorem learningZeroRate_sound :
    NarrativeDynamics.learnInstrumentality
        learningZeroRate.old.toReal
        learningZeroRate.observed.toReal
        learningZeroRate.rate.toReal =
      learningZeroRate.expected.toReal := by
  norm_num [learningZeroRate, ExactRat.toReal, exactRat,
    NarrativeDynamics.learnInstrumentality]

private def definitions : List String :=
  [
    "NarrativeDynamics.bayesPosterior",
    "NarrativeDynamics.learnInstrumentality",
    "NarrativeDynamics.effectivePressure",
    "NarrativeDynamics.goalScore"
  ]

/-- Canonically ordered corpus. Ordering is part of the committed golden-file
identity and is checked again by the Python loader. -/
def referenceVectors : List ReferenceVector :=
  [
    bayesNegativeEvidence.toReferenceVector,
    bayesNonbinaryFraction.toReferenceVector,
    bayesPositiveEvidence.toReferenceVector,
    bayesUninformative.toReferenceVector,
    effectivePressureFullBoundary.toReferenceVector,
    effectivePressureHalfBoundary.toReferenceVector,
    effectivePressureZeroBoundary.toReferenceVector,
    goalScoreCostRiskOnly.toReferenceVector,
    goalScoreFractional.toReferenceVector,
    goalScorePositiveNet.toReferenceVector,
    learningFullRate.toReferenceVector,
    learningMovesDown.toReferenceVector,
    learningMovesUp.toReferenceVector,
    learningZeroRate.toReferenceVector
  ]

private def stringArrayJson (values : List String) : String :=
  "[" ++ String.intercalate "," (values.map jsonQuote) ++ "]"

/-- One deterministic JSON document consumed by the Python conformance gate. -/
def referenceSuiteJson : String :=
  "{" ++
    jsonQuote "schema_version" ++ ":1," ++
    jsonQuote "generator" ++ ":" ++
      jsonQuote "NarrativeDynamics.Conformance.ReferenceVectors" ++ "," ++
    jsonQuote "numeric_encoding" ++ ":" ++
      jsonQuote "reduced_fraction_v1" ++ "," ++
    jsonQuote "definitions" ++ ":" ++ stringArrayJson definitions ++ "," ++
    jsonQuote "vectors" ++ ":[" ++
      String.intercalate "," (referenceVectors.map ReferenceVector.toJson) ++
    "]}"

end NarrativeDynamics.Conformance

/-- Executable entry point used by CI to regenerate the golden fixture. -/
def main : IO Unit :=
  IO.println NarrativeDynamics.Conformance.referenceSuiteJson
