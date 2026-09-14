import YoloFlywire.Contracts

namespace YoloFlywire

structure ValidProtocol where
  protocol : RunProtocol
  finalTestSealed : protocol.finalTestUsedForSelection = false

def claimAdmissible (p : ValidProtocol) (claim : EvidenceClaim) : Prop :=
  match claim with
  | .temporalModelUseful => True
  | .topologySpecificAdvantage => hasRequiredRewiredControl p.protocol
  | .robustnessAdvantage => True

end YoloFlywire
