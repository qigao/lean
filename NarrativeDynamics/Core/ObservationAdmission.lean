import NarrativeDynamics.Core.WorldGraph
import NarrativeDynamics.Core.BeliefSupport

namespace NarrativeDynamics

universe uP uA uE uO uL uI uC uN

/-- A private proof may be treated as raw observation evidence only when the
objective world records that the KB owner observed the event, the proof is
currently valid, concludes exactly that event, and is an asserted/root proof
rather than an inference-derived proposition. -/
structure ObservationEvidenceAdmissible
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    (w : WorldGraph Agent Event Node)
    (kb : AgentBeliefKB ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop)
    (event : Event) (proof : ProofId) : Prop where
  observed : w.observed event kb.owner
  proofValid : ProofValid kb.graph active proof
  factMatches : kb.graph.fact proof =
    (HNode.event event : HNode Agent Event Object Location Institution Concept)
  assertedRoot : kb.graph.origin proof = none
  noParents : kb.graph.parents proof = []

/-- In a well-formed world, every admitted observation proof inherits the
world's information-path guarantee. -/
theorem admissible_observation_has_info_path
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    (w : WorldGraph Agent Event Node)
    (hw : WorldInvariant w)
    (kb : AgentBeliefKB ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop)
    (event : Event) (proof : ProofId)
    (h : ObservationEvidenceAdmissible w kb active event proof) :
    infoReachable w.info (w.eventNode event) (w.agentNode kb.owner) := by
  exact hw.2.1 event kb.owner h.observed

/-- No information path means no proof can be admitted as an observation of
that event by this agent, even if some private data structure happens to claim
such a fact. -/
theorem no_info_path_no_admissible_observation
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    (w : WorldGraph Agent Event Node)
    (hw : WorldInvariant w)
    (kb : AgentBeliefKB ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop)
    (event : Event) (proof : ProofId)
    (hno : ¬ infoReachable w.info (w.eventNode event) (w.agentNode kb.owner)) :
    ¬ ObservationEvidenceAdmissible w kb active event proof := by
  intro hadmit
  exact hno (admissible_observation_has_info_path
    w hw kb active event proof hadmit)

/-- An admitted observation proof also supports the corresponding event fact
inside the agent's private symbolic KB. -/
theorem admissible_observation_supports_event_belief
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    (w : WorldGraph Agent Event Node)
    (kb : AgentBeliefKB ProofId Agent Event Object Location Institution Concept)
    (active : ProofId → Prop)
    (event : Event) (proof : ProofId)
    (h : ObservationEvidenceAdmissible w kb active event proof) :
    BeliefSupported kb active
      (HNode.event event : HNode Agent Event Object Location Institution Concept) := by
  have hs := valid_proof_supports_belief kb active proof h.proofValid
  rw [h.factMatches] at hs
  exact hs

end NarrativeDynamics
