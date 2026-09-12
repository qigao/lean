import YoloFlywire.Protocol

namespace YoloFlywire

theorem topologyClaim_requires_rewired
    (p : ValidProtocol)
    (h : claimAdmissible p EvidenceClaim.topologySpecificAdvantage) :
    hasRequiredRewiredControl p.protocol := by
  exact h

theorem protocol_final_test_sealed (p : ValidProtocol) :
    p.protocol.finalTestUsedForSelection = false := by
  exact p.finalTestSealed

example (p : ValidProtocol) :
    claimAdmissible p EvidenceClaim.topologySpecificAdvantage →
    hasRequiredRewiredControl p.protocol := by
  intro h
  exact topologyClaim_requires_rewired p h

example (p : ValidProtocol) :
    p.protocol.finalTestUsedForSelection = false := by
  exact protocol_final_test_sealed p

end YoloFlywire
