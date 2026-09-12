namespace YoloFlywire

inductive Split where
  | train
  | validation
  | test
  deriving DecidableEq, Repr

inductive ModelFamily where
  | gru
  | tcn
  | transformer
  | randomGraph
  | rewiredFlywire
  | flywire
  deriving DecidableEq, Repr

inductive TopologyKind where
  | dense
  | randomSparse
  | rewired
  | connectome
  deriving DecidableEq, Repr

structure TrainingBudget where
  epochs : Nat
  maxUpdates : Nat
  parameterCeiling : Nat
  deriving DecidableEq, Repr

inductive EvidenceClaim where
  | temporalModelUseful
  | topologySpecificAdvantage
  | robustnessAdvantage
  deriving DecidableEq, Repr

structure RunArm where
  family : ModelFamily
  topology : TopologyKind
  observationSchemaHash : String
  splitHash : String
  budget : TrainingBudget
  seeds : List Nat
  deriving Repr

structure RunProtocol where
  arms : List RunArm
  finalTestUsedForSelection : Bool
  primaryMetric : String
  successThreshold : Float
  deriving Repr

end YoloFlywire
