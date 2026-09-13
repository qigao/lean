import NarrativeDynamics.Core.FitnessDistribution

namespace NarrativeDynamics.FitnessAttachment.DistributionFixtures

example : Fintype.card (TargetTrace 3 2 0) = 1 := by decide_cbv
example : Fintype.card (TargetTrace 3 2 1) = 6 := by decide_cbv
example : Fintype.card (TargetTrace 2 1 2) = 6 := by decide_cbv

end NarrativeDynamics.FitnessAttachment.DistributionFixtures
