import NarrativeDynamics.Core.ObservationAdmission

open NarrativeDynamics

example
    {ProofId Agent Event Object Location Institution Concept Node : Type*}
    (w : WorldGraph Agent Event Node)
    (hw : WorldInvariant w)
    (kb : AgentBeliefKB ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop)
    (event : Event) (proof : ProofId)
    (h : ObservationEvidenceAdmissible w kb active event proof) :
    infoReachable w.info (w.eventNode event) (w.agentNode kb.owner) := by
  exact admissible_observation_has_info_path w hw kb active event proof h

example
    {ProofId Agent Event Object Location Institution Concept Node : Type*}
    (w : WorldGraph Agent Event Node)
    (hw : WorldInvariant w)
    (kb : AgentBeliefKB ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop)
    (event : Event) (proof : ProofId)
    (hno : ¬ infoReachable w.info (w.eventNode event) (w.agentNode kb.owner)) :
    ¬ ObservationEvidenceAdmissible w kb active event proof := by
  exact no_info_path_no_admissible_observation w hw kb active event proof hno

example
    {ProofId Agent Event Object Location Institution Concept Node : Type*}
    (w : WorldGraph Agent Event Node)
    (kb : AgentBeliefKB ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop)
    (event : Event) (proof : ProofId)
    (h : ObservationEvidenceAdmissible w kb active event proof) :
    BeliefSupported kb active
      (HNode.event event : HNode Agent Event Object Location Institution Concept) := by
  exact admissible_observation_supports_event_belief w kb active event proof h
