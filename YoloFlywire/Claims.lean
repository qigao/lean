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

theorem flywire_gru_only_rejects_topology_claim
    (p : ValidProtocol)
    (hOnly : ∀ arm, arm ∈ p.protocol.arms →
      arm.family = ModelFamily.flywire ∨ arm.family = ModelFamily.gru) :
    ¬ claimAdmissible p EvidenceClaim.topologySpecificAdvantage := by
  intro h
  change hasRequiredRewiredControl p.protocol at h
  rcases h with ⟨fly, rewired, hFlyMem, hRewiredMem, hFlyFamily, hRewiredFamily, hObs, hBudget⟩
  have hFamily := hOnly rewired hRewiredMem
  rcases hFamily with hAsFlywire | hAsGru
  · have impossible : ModelFamily.rewiredFlywire = ModelFamily.flywire :=
      hRewiredFamily.symm.trans hAsFlywire
    cases impossible
  · have impossible : ModelFamily.rewiredFlywire = ModelFamily.gru :=
      hRewiredFamily.symm.trans hAsGru
    cases impossible

example (p : ValidProtocol) :
    claimAdmissible p EvidenceClaim.topologySpecificAdvantage →
    hasRequiredRewiredControl p.protocol := by
  intro h
  exact topologyClaim_requires_rewired p h

example (p : ValidProtocol) :
    p.protocol.finalTestUsedForSelection = false := by
  exact protocol_final_test_sealed p

end YoloFlywire
